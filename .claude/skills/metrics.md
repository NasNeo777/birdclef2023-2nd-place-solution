---
name: metrics
description: 查看 BirdCLEF 训练指标（当前状态、修复对比、历史记录、实时监控）
command: metrics
---

执行 `python3 scripts/metrics.py` 展示训练指标。

## 用法

```bash
# 当前训练状态（默认）
python3 scripts/metrics.py --latest

# 修复前后 sed_v2s 对比
python3 scripts/metrics.py --compare

# 所有训练历史（按模型+session分组）
python3 scripts/metrics.py --history

# 实时监控（每30秒刷新）
python3 scripts/metrics.py --watch
```

用户说 `/metrics` 且无参数时，默认执行 `--latest` 显示当前训练状态。
用户说 `/metrics compare` 时，执行 `--compare` 对比修复前后的指标。
用户说 `/metrics history` 时，执行 `--history` 展示完整训练历史。
用户说 `/metrics watch` 时，执行 `--watch` 实时监控训练进度。

脚本自动从 `wandb/` 目录提取指标，自动识别模型名和阶段，自动检测运行中的任务。
