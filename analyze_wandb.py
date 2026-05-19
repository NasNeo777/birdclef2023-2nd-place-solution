import json
import os
import glob
from collections import defaultdict

wandb_dir = "/media/nasneo/AI/PyProject/birdclef2023-2nd-place-solution/wandb"

# Collect all runs
runs = []
for run_path in sorted(glob.glob(os.path.join(wandb_dir, "run-*"))):
    run_name = os.path.basename(run_path)
    summary_file = os.path.join(run_path, "files", "wandb-summary.json")
    meta_file = os.path.join(run_path, "files", "wandb-metadata.json")

    summary = {}
    metadata = {}

    if os.path.exists(summary_file):
        with open(summary_file) as f:
            content = f.read().strip()
            if content:
                summary = json.loads(content)

    if os.path.exists(meta_file):
        with open(meta_file) as f:
            content = f.read().strip()
            if content:
                metadata = json.loads(content)

    # Extract stage and model from metadata CLI args
    args = metadata.get("args", [])
    stage = None
    model_name = None
    for i, arg in enumerate(args):
        if arg == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
        if arg == "--model_name" and i + 1 < len(args):
            model_name = args[i + 1]

    # Get git commit
    git_commit = metadata.get("git", {}).get("commit", "unknown")[:7]
    started_at = metadata.get("startedAt", "unknown")

    val_auc = summary.get("val_roc_auc", None)
    val_loss = summary.get("val_loss", None)
    epoch = summary.get("epoch", None)
    runtime = summary.get("_runtime", None)
    train_loss = summary.get("train_loss", None)

    # Determine if this is SoftAUCLoss or original CE/BCE
    # Original runs: May 13-16, SoftAUCLoss: May 17+ (commit c58110f)
    loss_type = "unknown"
    if started_at and started_at != "unknown":
        date_str = started_at[:10]  # "2026-05-13"
        if date_str <= "2026-05-16":
            loss_type = "CE/BCE"
        else:
            loss_type = "SoftAUCLoss"

    # Detect broken runs
    is_broken = (val_auc is not None and val_auc <= 0.51) or (epoch is not None and epoch < 5)

    runs.append({
        "run_name": run_name,
        "model": model_name or "unknown",
        "stage": stage or "unknown",
        "val_auc": val_auc,
        "val_loss": val_loss,
        "epoch": epoch,
        "runtime": runtime,
        "train_loss": train_loss,
        "loss_type": loss_type,
        "git_commit": git_commit,
        "started_at": started_at,
        "is_broken": is_broken,
    })

# === ANALYSIS ===

# 1. Group by model + stage, find best AUC per group for each loss type
print("# BirdCLEF 2026 训练日志分析报告\n")

print(f"## 总览：{len(runs)} 个 wandb run\n")

# Count by loss type
ce_runs = [r for r in runs if r["loss_type"] == "CE/BCE"]
soft_runs = [r for r in runs if r["loss_type"] == "SoftAUCLoss"]
broken_runs = [r for r in runs if r["is_broken"]]
print(f"- CE/BCE loss（5/13-5/16）：{len(ce_runs)} 个 run")
print(f"- SoftAUCLoss（5/17-5/19）：{len(soft_runs)} 个 run")
print(f"- 异常 run（AUC≤0.51 或 epoch<5）：{len(broken_runs)} 个")

# 2. Best AUC by model + stage for each loss type
print("\n## 各模型 x Stage 最佳 AUC 对比\n")

models_order = ["sed_v2s", "sed_b3ns", "sed_seresnext26t", "cnn_v2s", "cnn_resnet34d", "cnn_b3ns", "cnn_b0ns"]
stages_order = ["pretrain_ce", "pretrain_bce", "train_ce", "train_bce", "finetune", "soft_loss"]

# Build comparison table
print("| 模型 | Stage | CE/BCE AUC | SoftAUCLoss AUC | Δ AUC | 备注 |")
print("|------|-------|------------|-----------------|-------|------|")

for model in models_order:
    for stage in stages_order:
        ce_best = max(
            (r for r in runs if r["model"] == model and r["stage"] == stage and r["loss_type"] == "CE/BCE" and not r["is_broken"]),
            key=lambda r: r["val_auc"] or 0,
            default=None
        )
        soft_best = max(
            (r for r in runs if r["model"] == model and r["stage"] == stage and r["loss_type"] == "SoftAUCLoss" and not r["is_broken"]),
            key=lambda r: r["val_auc"] or 0,
            default=None
        )

        if ce_best or soft_best:
            ce_auc = f"{ce_best['val_auc']:.4f}" if ce_best and ce_best["val_auc"] else "-"
            soft_auc = f"{soft_best['val_auc']:.4f}" if soft_best and soft_best["val_auc"] else "-"

            delta = ""
            note = ""
            if ce_best and soft_best and ce_best["val_auc"] and soft_best["val_auc"]:
                d = soft_best["val_auc"] - ce_best["val_auc"]
                delta = f"{d:+.4f}"
                if d < -0.05:
                    note = "⚠️ 显著下降"
                elif d < -0.02:
                    note = "📉 下降"
                elif d > 0.02:
                    note = "📈 上升"
                else:
                    note = "≈ 持平"
            elif ce_best and not soft_best:
                note = "缺 SoftAUCLoss 数据"
            elif soft_best and not ce_best:
                note = "缺 CE/BCE 数据"

            if ce_best and ce_best["is_broken"]:
                note += " [CE异常]"
            if soft_best and soft_best["is_broken"]:
                note += " [Soft异常]"

            print(f"| {model} | {stage} | {ce_auc} | {soft_auc} | {delta} | {note} |")

# 3. Detailed listing per loss type
print("\n## CE/BCE 组详细（5/13-5/16）\n")

for model in models_order:
    model_runs = [r for r in ce_runs if r["model"] == model]
    if not model_runs:
        continue
    print(f"### {model}\n")
    print("| Stage | Run | AUC | Val Loss | Epoch | Runtime |")
    print("|-------|-----|-----|----------|-------|---------|")
    for r in sorted(model_runs, key=lambda x: stages_order.index(x["stage"]) if x["stage"] in stages_order else 99):
        auc_str = f"{r['val_auc']:.4f}" if r["val_auc"] else "N/A"
        loss_str = f"{r['val_loss']:.2f}" if r["val_loss"] else "N/A"
        ep_str = str(r["epoch"]) if r["epoch"] else "N/A"
        rt_str = f"{r['runtime']/3600:.1f}h" if r["runtime"] else "N/A"
        flag = " ❌" if r["is_broken"] else ""
        print(f"| {r['stage']} | {r['run_name'][:30]} | {auc_str} | {loss_str} | {ep_str} | {rt_str}{flag} |")
    print()

print("## SoftAUCLoss 组详细（5/17-5/19）\n")

for model in models_order:
    model_runs = [r for r in soft_runs if r["model"] == model]
    if not model_runs:
        continue
    print(f"### {model}\n")
    print("| Stage | Run | AUC | Val Loss | Epoch | Runtime |")
    print("|-------|-----|-----|----------|-------|---------|")
    for r in sorted(model_runs, key=lambda x: stages_order.index(x["stage"]) if x["stage"] in stages_order else 99):
        auc_str = f"{r['val_auc']:.4f}" if r["val_auc"] else "N/A"
        loss_str = f"{r['val_loss']:.2f}" if r["val_loss"] else "N/A"
        ep_str = str(r["epoch"]) if r["epoch"] else "N/A"
        rt_str = f"{r['runtime']/3600:.1f}h" if r["runtime"] else "N/A"
        flag = " ❌" if r["is_broken"] else ""
        print(f"| {r['stage']} | {r['run_name'][:30]} | {auc_str} | {loss_str} | {ep_str} | {rt_str}{flag} |")
    print()

# 4. Broken runs
print("## 异常 Run（AUC ≤ 0.51 或 epoch < 5）\n")
if broken_runs:
    print("| Run | Model | Stage | AUC | Epoch | Loss Type |")
    print("|-----|-------|-------|-----|-------|-----------|")
    for r in broken_runs:
        print(f"| {r['run_name']} | {r['model']} | {r['stage']} | {r['val_auc']} | {r['epoch']} | {r['loss_type']} |")
else:
    print("无异常 run")

# 5. Currently active runs
active_runs = [r for r in runs if r["epoch"] and r["epoch"] < 5 and r["val_auc"] and r["val_auc"] > 0.51]
if active_runs:
    print("\n## 疑似正在运行的 Run\n")
    for r in active_runs:
        print(f"- {r['run_name']}: {r['model']} {r['stage']} epoch={r['epoch']} AUC={r['val_auc']:.4f}")

# 6. Summary and recommendations
print("\n## 总结与建议\n")

# Calculate average delta for stages that have both CE and SoftAUCLoss
deltas = []
for model in models_order:
    for stage in stages_order:
        ce_best = max(
            (r for r in runs if r["model"] == model and r["stage"] == stage and r["loss_type"] == "CE/BCE" and not r["is_broken"]),
            key=lambda r: r["val_auc"] or 0,
            default=None
        )
        soft_best = max(
            (r for r in runs if r["model"] == model and r["stage"] == stage and r["loss_type"] == "SoftAUCLoss" and not r["is_broken"]),
            key=lambda r: r["val_auc"] or 0,
            default=None
        )
        if ce_best and soft_best and ce_best["val_auc"] and soft_best["val_auc"]:
            deltas.append(soft_best["val_auc"] - ce_best["val_auc"])

if deltas:
    avg_delta = sum(deltas) / len(deltas)
    print(f"- 有 CE/BCE 和 SoftAUCLoss 双数据的 stage 共 {len(deltas)} 个")
    print(f"- **SoftAUCLoss 平均 AUC 变化：{avg_delta:+.4f}**")
    print(f"- 其中下降的有 {sum(1 for d in deltas if d < -0.01)} 个，上升的有 {sum(1 for d in deltas if d > 0.01)} 个")
    print(f"- 最大降幅：{min(deltas):+.4f}，最大升幅：{max(deltas):+.4f}")

    print("\n### 建议")
    if avg_delta < -0.05:
        print("- 🔴 **强烈建议回退到 CE/BCE loss**，SoftAUCLoss 导致 AUC 显著下降")
        print("- 原因：SoftAUCLoss 的硬阈值（0.5）与 Mixup 软标签不兼容")
        print("- 如果一定要用 SoftAUCLoss，需要：")
        print("  1. 关闭 Mixup（mixup=False, mixup2=False）")
        print("  2. 或修改 SoftAUCLoss 支持连续权重（去掉 pair_mask 硬阈值）")
    else:
        print("- 影响不大，可以继续观察")
