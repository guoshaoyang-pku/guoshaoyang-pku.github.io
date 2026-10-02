// Baseline leaderboard for the observatory.
//
// Every "controlled" row is measured in THIS project's engine only
// (suika/part2 settle loop, K=128 columns, loop merge rule) with greedy /
// deterministic inference via suika_dqn/eval_headtohead.py, so the numbers are
// directly comparable with each other and with the arena curves.
// Community rows are shown twice on purpose: natively (their own env) and
// transplanted into our env.
window.SUIKA_BENCH = {
  "generated": "2026-09-29 11:10",
  "engine": "ole-batting settle (pymunk 6.11.1) · K=128 · loop rule · seeds 0-15",
  "controlled": [
    {
      "name": "w3b_mlp_deep",
      "label": "cluster DQN · Dueling MLP 2048³",
      "cluster": "a100_perm",
      "train": "1.48B env steps · γ=1.0 · batch 64k · PER+mirror+n-step3 · EMA target",
      "seeds": 16,
      "mean": 2066.4, "median": 1903.5, "p25": 1757.2, "max": 3049, "min": 1426,
      "p2000": 0.4375, "melon_rate": null, "moves": 202,
      "note": "当前最强 RL policy（仍在训练）；本环境独立复评，与节点自带 greedy（2060）差 <1%，故训练管线评测可信"
    },
    {
      "name": "w3b_tf_deep",
      "label": "cluster DQN · Set-Transformer (8.2M)",
      "cluster": "a100_t1",
      "train": "256M env steps · 同样的 scale 配方（mlp 的 1/5.8 步数）",
      "seeds": 16,
      "mean": 2018.8, "median": 2034.5, "p25": 1688.2, "max": 3039, "min": 980,
      "p2000": 0.5625, "melon_rate": null, "moves": 198,
      "note": "样本效率明显更高；单局 max 3039 越过社区 3000 精英线，≥2000 局占 9/16"
    },
    {
      "name": "heuristic",
      "label": "启发式 + 1 步真物理前瞻",
      "cluster": "local (part2/ai_agent.py)",
      "train": "无学习，纯搜索",
      "seeds": 16,
      "mean": 1596, "median": null, "p25": null, "max": 2004, "min": 1025,
      "p2000": 0.0, "melon_rate": null, "moves": 170,
      "note": "settle 动力学下很强的非学习基线（5 种子时 mean 1812）"
    },
    {
      "name": "plaindqn_g7",
      "label": "cluster DQN · Dueling MLP 1024² (wave1)",
      "cluster": "a100_perm",
      "train": "51M env steps",
      "seeds": 16,
      "mean": 1309, "median": null, "p25": null, "max": 1947, "min": 780,
      "p2000": 0.0, "melon_rate": null, "moves": 144,
      "note": "wave1 冠军，已被 wave3b 大幅超越"
    },
    {
      "name": "alphazero",
      "label": "AlphaZero (PUCT MCTS + Object-Transformer)",
      "cluster": "local (rl/)",
      "train": "step 8900，~9k 优化步（严重欠训练）",
      "seeds": 5,
      "mean": 814, "median": null, "p25": null, "max": 974, "min": 617,
      "p2000": 0.0, "melon_rate": 0.0, "moves": 104,
      "note": "搜索在运作但网络饥饿：200 visits 只把 814 提到 1004"
    },
    {
      "name": "random",
      "label": "随机基线",
      "cluster": "local",
      "train": "—",
      "seeds": 5,
      "mean": 499, "median": null, "p25": null, "max": null, "min": null,
      "p2000": 0.0, "melon_rate": 0.0, "moves": null,
      "note": "均匀随机列"
    }
  ],
  "community": [
    {
      "name": "mattjacobs30",
      "label": "MattJacobs30/SuikaReinforcement (DQN)",
      "native_mean": 1058.8, "native_seeds": 5,
      "ours_mean": 672.4, "ours_seeds": 5, "ours_max": 916,
      "note": "其 env 每个 seed 不可复现（局内出生走全局 rng），单种子分数只是幸运样本"
    },
    {
      "name": "moonfloof",
      "label": "moonfloof (Matter.js, 640×960)",
      "native_mean": null, "native_seeds": 0,
      "ours_mean": null, "ours_seeds": 0, "ours_max": 1415,
      "note": "不同引擎（Matter.js）；仅作跨引擎参考，不计入受控榜"
    },
    {
      "name": "dsaran-7",
      "label": "dsaran-7/suika-game-ai（空间启发式）",
      "native_mean": null, "native_seeds": 0,
      "ours_mean": null, "ours_seeds": 0, "ours_max": null,
      "note": "作者自称 3000–3500，无独立复现；列为待验证基线"
    }
  ]
};
