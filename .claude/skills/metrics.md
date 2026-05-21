---
name: metrics
description: 查看 BirdCLEF 训练指标（当前状态、修复对比、历史记录、实时监控）
command: metrics
---

执行 `python3 scripts/metrics.py` 展示训练指标。

## 用法

```bash
# 当前训练状态 + 预估完成时间（默认）
python3 scripts/metrics.py --latest

# 仅查看预估完成时间
python3 scripts/metrics.py --eta

# 修复前后 sed_v2s 对比
python3 scripts/metrics.py --compare

# 所有训练历史（按模型+session分组）
python3 scripts/metrics.py --history

# 实时监控（每30秒刷新）
python3 scripts/metrics.py --watch
```

用户说 `/metrics` 且无参数时，默认执行 `--latest` 显示当前训练状态 + ETA。
用户说 `/metrics eta` 或问"还有多久跑完"时，执行 `--eta` 计算预估完成时间。
用户说 `/metrics compare` 时，执行 `--compare` 对比修复前后的指标。
用户说 `/metrics history` 时，执行 `--history` 展示完整训练历史。
用户说 `/metrics watch` 时，执行 `--watch` 实时监控训练进度。

脚本自动从 `wandb/` 目录提取指标，自动识别模型名和阶段，自动检测运行中的任务。

## ETA 计算逻辑

- 从当前运行的 W&B run 的 history 日志推断每 epoch 耗时
- CNN 模型按 SED 的 0.75 倍估算（架构更简单）
- finetune 阶段按 1.5 倍估算（clip duration 更长）
- 自动跳过已有 checkpoint 的阶段
