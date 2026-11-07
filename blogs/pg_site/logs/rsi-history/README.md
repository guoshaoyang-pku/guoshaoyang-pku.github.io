# H100 / H200 RSI 历史训练曲线摘要

这里是另外两批本地可用的历史记录快照，不是全部历史实验的清单，也不含完整原始 result、stdout/stderr 或运行配置。与 Pro5000 曲线分别归档，不把不同协议的分数混为一谈。

| 文件 | 来源和内容 | 记录数 | SHA-256 |
| --- | --- | ---: | --- |
| `h100-online-curves.jsonl.gz` | 2026-10-25 H100 在线搜索终态归档的 result 摘要及可用训练/验证曲线；其中 8 条没有独立曲线文件 | 3,320 | `462f62c825c5ac68dce8893ba2045622bdbdc53d8d66ca873be7d7798a18cfd7` |
| `h200-shared-curves.jsonl.gz` | 2026-10-25 H200 共享搜索控制器本地快照中的 attempt 结果摘要、setting 和可用训练曲线 | 924 | `e442d244576b4eac79f8fcc565c2875508c3c8d54c371febfe716ef92ed12a7c` |

每行是一个 JSON 对象；部分任务失败、缺失曲线或属于同一候选的不同 attempt，不能把记录数等同于有效候选数。字段只保留结果、实验设置和曲线所需信息，不含运行环境、机器身份和原始执行包。历史未记录的逐步 loss 无法补回。

```bash
zcat runs/logs/rsi-history/h100-online-curves.jsonl.gz | head -1 | python3 -m json.tool
zcat runs/logs/rsi-history/h200-shared-curves.jsonl.gz | head -1 | python3 -m json.tool
```

H100 源是本地 `pg_online_mcmc_20261024/native_v2_20261024/terminal_archive_20261025`；H200 源是本地 `pg_shared_live_20261025/central/attempts`。这些快照不会自动更新，其他旧运行及更新后的数据需要单独盘点和归档。
