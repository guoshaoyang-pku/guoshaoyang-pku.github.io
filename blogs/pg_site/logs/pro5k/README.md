# 实验曲线和结果摘要（pro5k 服务历史快照）

这里保存 pro5k 集群六台机器 `pg_pro5k_smallw_20261026` 服务于 2026-11-07 21:43–21:44 导出的 271,496 个任务记录，
按 `节点__任务前缀` 分卷，gzip 压缩的 JSONL，每行一个任务。共 651 卷，约 488 MB（467 MiB）。
导出包含可用曲线及结果摘要，不包含完整原始 result.json 或 stdout/stderr，也不涵盖导出之后的任务。

## 每行格式

```json
{
  "job": "ag_000eef38bc0fa304",
  "node": "pro5kvrdma1",
  "result": {"score": 2.96, "updates": 1283, "status": "complete", "seed": 43,
             "key": "a969d...", "protocol_id": "...", "setting_hash": "...",
             "progress_sha": "...", "step_points": 1283, "step_log": "step_losses.jsonl"},
  "curve":  { ... learning_curve.json 完整内容 ... },
  "steps":  [[1, 0.1498, 6.92873], [2, 0.2996, 6.478267], ...]
}
```

- `curve`：worker 写出的 `learning_curve.json` 完整内容，含 `setting`（完整模型与训练配置）、
  `score`、`updates`、`training_history`（每 50 步 + 15/30/60 秒检查点 + 终点，
  每个点有 `updates` / `actual_training_seconds` / `online_loss`）、`checkpoint_history`、
  `context`（节点、GPU、协议哈希、git 版本）。
- `steps`：**逐步 loss**（每个 optimizer update 一条），只有 2026-11-07 晚之后的新任务才有；
  历史任务没有逐步日志（当时 worker 为省主机同步开销只在 GPU 上保留滑窗），
  最早的粒度就是 `curve.training_history` 的 50 步。
- `result`：`result.json` 的标量字段。`result.json` 与 `curve` 中的分数、步数一致。

## 下载与读取

```bash
# 单卷（在博客 repo 里直接 zcat）
zcat blogs/pg_site/logs/pro5k/pro5kvrdma1__ag.jsonl.gz | head -1 | python3 -m json.tool

# 网页下载（panel「日志中心」也是这个链接）
https://guoshaoyang-pku.github.io/blogs/pg_site/logs/pro5k/<卷名>
```

## 前缀含义

任务 id 前缀对应提交服务/实验系列：`ag` 默认 agent 评测服务（challenge 正式搜索）、
`a5` 新评测服务（v1.4/v1.5 等）、`gs` 表格/大搜索、`t6` 600 秒长预算、
`sv` 监督/参考臂，其余小前缀为历次专项实验。每卷的任务数与字节数见
`<节点>__manifest.json`。

## 更新方式

- 节点原始文件保留在 `pro5kvrdma{1..6}:/data/shared/guoshaoyang/pg_pro5k_smallw_20261026/results/<job>/`
  （`learning_curve.json`、`result.json`、新任务另有 `step_losses.jsonl`）。
- 搜索头 `pro5kvrdma6:/data/shared/guoshaoyang/search_head/outputs/curves/pro5k/` 由
  `tools/curve_harvest.py --loop 1200` 每 20 分钟增量回收（含逐步日志）。
- 重新生成全量分卷：在各节点运行 `tools/export_node_logs.py <节点名>`，
  输出到服务目录 `export_logs/`。
- 此网页是静态快照；搜索头增量回收不会自动更新网页分卷。
