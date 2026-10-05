# KB science loop setting

本页只记录运行协议；每个 checkpoint 的 KB、报告、源码和哈希以 run 页面中的实际文件为准。

## Release 与评分

`kb_science_pool_v3` 固定 seed=`20261004`，共 980 行：`arch170` 80、`ranking_v2` 400、`ranking_v3` 300、`dataflip500` 200。按 group 和可追踪的数据集/候选集标识整组划分；实际 train/val/test 为 770/105/105。arch170 与 dataflip500 没有可用的新 holdout，当前宏观验证只覆盖 ranking_v2/v3。

选择题答对得 1 分；排序题按逆序对数 0/1/2/3 得 1/0.75/0.5/0.25，≥4 或无法解析得 0。claim 的 `s += score`、`f += 1-score` 只是引用反馈，不是真实概率。

## 当前运行

`soa_async_v1` 使用 `gpt-6.1-sol` / Codex；每题保存 KB 与 no-KB 对照。解题 batch 为 30，计划 15 轮；答案揭示前使用冻结 KB。当前设置关闭最终 test 和 test-on-accept。

总结者与研究者是持久异步任务，不设 14 次工具调用上限。40 分钟是软窗口：超过窗口后继续原任务，以跨轮 label 标注，保留发起轮次、会话和 checkpoint；进程停下后可恢复。研究报告可覆盖多条 claim，保留测量、竞争解释、可区分预测和未决问题。研究是否 ready 不阻止下一轮读取。

## 评测与发布

每 5 轮做一次 ranking v2/v3 宏观检查，每组 3 次重复；连续两个检查点明显下降才回滚。这个检查是迭代信号，不是最终保留集。

publisher 只发布真实保存的 KB、报告、研究 repo、源码包和日志，沿用 run viewer；失败推送会在后续周期继续核验。页面不添加主页 block 或装饰性统计区。
