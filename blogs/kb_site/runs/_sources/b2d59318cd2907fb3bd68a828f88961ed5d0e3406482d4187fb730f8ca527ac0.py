"""Training loop for this candidate — executed by the ground-truth runner."""
from __future__ import annotations

import math
import hashlib

import torch
import torch.nn.functional as F

from loss import loss_fn
from model import Model
from optimizer import build_optimizer


def _resolve_device(device: str) -> torch.device:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported execution device {device!r}; choose 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable "
            f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda!r})"
        )
    return torch.device(device)


def _test_metrics(model: torch.nn.Module, test_x: torch.Tensor, test_y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(test_x)
        ce = F.cross_entropy(logits, test_y.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == test_y.reshape(-1)).float().mean()
    return float(ce.item()), float(accuracy.item())


def train_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int = 0,
    fail_threshold: float = float("inf"),
    device: str = "cpu",
    progress_callback=None,
    process=None,
) -> dict:
    torch.manual_seed(seed)
    run_device = _resolve_device(device)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Model().to(run_device)
    optimizer = build_optimizer(model)
    train_x = train_x.to(run_device)
    train_y = train_y.to(run_device)
    test_x = test_x.to(run_device)
    test_y = test_y.to(run_device)
    recorder = _ProcessRecorder(model, optimizer, process, seed, (train_x, train_y, test_x, test_y))
    n = train_x.shape[0]
    step_metrics: list[float] = []
    eval_samples: list[int] = []
    failed = False
    progress_interval = max(1, steps // 100)
    final_accuracy = float("nan")

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch_size,), device=run_device)
        logits = model(train_x[idx])
        loss = loss_fn(model, logits, train_y[idx])
        if not torch.isfinite(loss):
            failed = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        observation = recorder.before(step, idx)
        optimizer.step()
        recorder.after(observation)

        ce, accuracy = _test_metrics(model, test_x, test_y)
        if not math.isfinite(ce) or not math.isfinite(accuracy):
            failed = True
            break
        eval_samples.append(step * batch_size)
        step_metrics.append(ce)
        final_accuracy = accuracy
        if progress_callback is not None and (
            step == 1 or step % progress_interval == 0 or step == steps
        ):
            progress_callback(
                {
                    "step": step,
                    "training_steps": steps,
                    "samples_seen": step * batch_size,
                    "total_samples_seen": steps * batch_size,
                    "metric": ce,
                    "accuracy": accuracy,
                }
            )

    final_metric = step_metrics[-1] if step_metrics else float("inf")
    if final_metric > fail_threshold:
        failed = True

    return {
        "process_diagnostics": recorder.finish(model, test_x, test_y, failed),
        "failed": failed,
        "final_test_ce": final_metric,
        "final_test_accuracy": final_accuracy,
        "eval_samples": eval_samples,
        "step_metrics": step_metrics,
    }

def _tensor_pin(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def _norm(value):
    return float(torch.linalg.vector_norm(value.detach().double()).item())


def _ratio(numerator, denominator):
    if denominator == 0:
        return {"value": None, "reason": "zero denominator"}
    return {"value": numerator / denominator, "reason": None}


def _cosine(left, right):
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return {"value": None, "reason": "zero vector"}
    return {"value": float(torch.sum(left.double() * right.double()).item()) / denominator, "reason": None}


class _ProcessRecorder:
    def __init__(self, model, optimizer, request, seed, tensors):
        if request is None or request.get("version") != "adam_ce_v1":
            raise ValueError("recording requires a normalized adam_ce_v1 request")
        if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
            raise ValueError("recording requires canonical single-group coupled Adam")
        self.group = optimizer.param_groups[0]
        if any(self.group.get(k, False) for k in ("amsgrad", "maximize", "capturable", "differentiable", "fused", "decoupled_weight_decay")):
            raise ValueError("unsupported process Adam mode")
        if tensors[2].shape[0] > request["max_eval_samples"] or tensors[2].shape[0] == 0:
            raise ValueError("lab evaluation sample count exceeds requested bound or is empty")
        self.named = list(model.named_parameters())
        if len(self.named) > 256:
            raise ValueError("process parameter count exceeds recorder bound")
        self.head = f"net.{len(model.net) - 1}."
        self.optimizer, self.request, self.seed = optimizer, request, seed
        self.stream = hashlib.sha256()
        self.steps = []
        self.record = {"schema_version": 1, "recorder_version": "adam_ce_v1", "request": request,
                       "seed": seed, "optimizer_type": "Adam", "decay_mode": "coupled",
                       "optimizer_defaults": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in self.group.items() if k != "params"},
                       "inputs": {name: _tensor_pin(value) for name, value in zip(
                       ("train_x", "train_y", "test_x", "test_y"), tensors)}, "steps": self.steps}

    def before(self, step, indices):
        self.stream.update(indices.detach().cpu().contiguous().numpy().tobytes())
        if step not in self.request["steps"]:
            return None
        snapshot = []
        for name, parameter in self.named:
            weight = parameter.detach().clone()
            gradient = parameter.grad.detach().clone() if parameter.grad is not None else None
            state = self.optimizer.state.get(parameter, {})
            entry = {"name": name, "role": "head" if name.startswith(self.head) else "hidden",
                     "shape": list(weight.shape), "active": gradient is not None, "parameter_norm": _norm(weight),
                     "moment_before": {key: _norm(state[key]) if key in state else 0.0
                                       for key in ("exp_avg", "exp_avg_sq")}}
            if gradient is None:
                entry["reason"] = "gradient absent; Adam skips this parameter"
            else:
                decay = self.group["weight_decay"] * weight
                entry.update(data_gradient_norm=_norm(gradient), decay_gradient_norm=_norm(decay),
                             coupled_gradient_norm=_norm(gradient + decay),
                             decay_to_data_ratio=_ratio(_norm(decay), _norm(gradient)),
                             parameter_data_cosine=_cosine(weight, gradient), parameter_decay_cosine=_cosine(weight, decay),
                             data_decay_cosine=_cosine(gradient, decay))
            snapshot.append((parameter, weight, gradient, entry))
        return {"step": step, "minibatch": _tensor_pin(indices), "parameters": snapshot}

    def after(self, snapshot):
        if snapshot is None:
            return
        rows = []
        for parameter, weight, gradient, entry in snapshot["parameters"]:
            descent = weight - parameter.detach()
            entry.update(descent_norm=_norm(descent), relative_descent=_ratio(_norm(descent), _norm(weight)))
            state = self.optimizer.state.get(parameter, {})
            entry["moment_after"] = {key: _norm(state[key]) if key in state else 0.0 for key in ("exp_avg", "exp_avg_sq")}
            if gradient is not None:
                beta1, beta2 = self.group["betas"]
                count = float(state["step"])
                direction = (state["exp_avg"] / (1 - beta1 ** count)) / (
                    torch.sqrt(state["exp_avg_sq"] / (1 - beta2 ** count)) + self.group["eps"])
                predicted = self.group["lr"] * direction
                tolerance = 8 * torch.finfo(weight.dtype).eps * max(1.0, float(weight.abs().max()))
                valid = bool(torch.allclose(descent, predicted, rtol=5e-5, atol=tolerance))
                entry.update(preconditioned_direction_norm=_norm(direction), predicted_descent_norm=_norm(predicted),
                             preconditioned_data_cosine=_cosine(direction, gradient),
                             preconditioned_decay_cosine=_cosine(direction, self.group["weight_decay"] * weight),
                             preconditioned_parameter_cosine=_cosine(direction, weight),
                             descent_data_cosine=_cosine(descent, gradient),
                             descent_decay_cosine=_cosine(descent, self.group["weight_decay"] * weight),
                             descent_parameter_cosine=_cosine(descent, weight),
                             adam_step_check={"passed": valid, "error_norm": _norm(descent - predicted),
                                              "atol": tolerance, "rtol": 5e-5}, moment_step=count)
                if not valid:
                    raise ValueError("recorded Adam direction does not match actual descent")
            elif torch.count_nonzero(descent).item():
                raise ValueError("inactive Adam parameter unexpectedly moved")
            rows.append(entry)
        self.steps.append({"step": snapshot["step"], "minibatch": snapshot["minibatch"], "parameters": rows})

    def finish(self, model, test_x, test_y, failed):
        self.record["status"] = "failed" if failed else "completed"
        self.record["minibatch_stream_sha256"] = self.stream.hexdigest()
        if failed:
            self.record["final_metrics"] = None
            return {"record": self.record, "evaluation": {}}
        model.eval()
        with torch.inference_mode():
            logits = model(test_x).detach().cpu()
            targets = test_y.reshape(-1).detach().cpu()
            predictions = logits.argmax(dim=-1)
            other = logits.clone()
            other.scatter_(1, targets[:, None], float("-inf"))
            margins = logits.gather(1, targets[:, None]).reshape(-1) - other.max(dim=1).values
            self.record["final_metrics"] = {"ce": float(F.cross_entropy(logits, targets)),
                                            "accuracy": float((predictions == targets).float().mean())}
        return {"record": self.record, "evaluation": {"indices": torch.arange(targets.shape[0]),
                "logits": logits, "targets": targets, "predictions": predictions,
                "errors": predictions != targets, "true_class_margins": margins}}
