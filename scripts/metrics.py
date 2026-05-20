#!/usr/bin/env python3
"""Extract and display training metrics from W&B runs.

Usage:
    python3 scripts/metrics.py               # show all runs summary
    python3 scripts/metrics.py --latest      # show only the latest training session
    python3 scripts/metrics.py --compare     # compare pre-fix vs post-fix runs
    python3 scripts/metrics.py --watch       # continuously monitor current training
"""

import os, sys, json, glob, time
from datetime import datetime
from argparse import ArgumentParser

WANDB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wandb")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Model order for display
MODEL_ORDER = ["sed_v2s", "sed_b3ns", "sed_seresnext26t", "cnn_v2s", "cnn_resnet34d", "cnn_b3ns", "cnn_b0ns"]
STAGE_ORDER = ["pretrain_ce", "pretrain_bce", "train_ce", "train_bce", "finetune", "soft_loss"]

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_run(run_dir):
    """Load summary + metadata from a W&B run directory."""
    summary_path = os.path.join(run_dir, "files", "wandb-summary.json")
    meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        summary = json.load(f)

    info = {
        "run_id": os.path.basename(run_dir),
        "epoch": summary.get("epoch"),
        "step": summary.get("trainer/global_step"),
        "train_loss": summary.get("train_loss"),
        "val_loss": summary.get("val_loss"),
        "val_auc": summary.get("val_roc_auc"),
        "model": "?",
        "stage": "?",
    }

    ts = summary.get("_timestamp", 0)
    if ts > 10000000000:
        ts /= 1000
    try:
        info["dt"] = datetime.fromtimestamp(ts)
    except (OSError, ValueError):
        info["dt"] = None

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        args = meta.get("args", [])
        for i, arg in enumerate(args):
            if arg == "--model_name" and i + 1 < len(args):
                info["model"] = args[i + 1]
            if arg == "--stage" and i + 1 < len(args):
                info["stage"] = args[i + 1]

    return info


def get_all_runs():
    """Return all runs sorted by timestamp."""
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(WANDB_DIR, "run-*"))):
        info = load_run(run_dir)
        if info is not None and info["dt"] is not None:
            runs.append(info)
    runs.sort(key=lambda r: r["dt"])
    return runs


def check_ckpt_exists(model, stage):
    """Check if a checkpoint exists for this model/stage."""
    ckpt_path = os.path.join(OUTPUT_DIR, model, "pytorch", stage, "last.ckpt")
    return os.path.exists(ckpt_path)


def format_auc(val):
    if val is None:
        return "     -"
    s = f"{val:>7.4f}"
    if val >= 0.90:
        return f"{GREEN}{s}{RESET}"
    elif val >= 0.85:
        return f"{BOLD}{s}{RESET}"
    elif val < 0.5:
        return f"{RED}{s}{RESET}"
    return s


def format_loss(val):
    if val is None:
        return "       -"
    s = f"{val:>8.3f}"
    if val is not None and val > 100:
        return f"{RED}{s}{RESET}"
    return s


def iso(dt):
    return dt.strftime("%m-%d %H:%M") if dt else "??"


def print_latest_session():
    """Show only the current/latest training session with checkpoints."""
    runs = get_all_runs()
    # Find runs that have checkpoints
    ckpt_runs = [r for r in runs if check_ckpt_exists(r["model"], r["stage"])]

    if not ckpt_runs:
        print("No completed checkpoints found.")
        return

    # Group by model
    models_seen = {}
    for r in ckpt_runs:
        if r["model"] not in models_seen or r["dt"] > models_seen[r["model"]]["dt"]:
            models_seen[r["model"]] = r

    # Also check running jobs
    running = get_running_job()

    print(f"\n{BOLD}=== 当前训练状态 ({datetime.now().strftime('%m-%d %H:%M')}) ==={RESET}\n")
    header = f"{'模型':<22} {'pretrain_ce':>10} {'pretrain_bce':>10} {'train_ce':>10} {'train_bce':>10} {'finetune':>10}"
    print(header)
    print("-" * 72)

    for model in MODEL_ORDER:
        row = f"{model:<22}"
        for stage in STAGE_ORDER[:5]:
            match = [r for r in ckpt_runs if r["model"] == model and r["stage"] == stage]
            # Prefer latest per model+stage
            match.sort(key=lambda r: r["dt"], reverse=True)
            if match:
                row += format_auc(match[0]["val_auc"])
            else:
                # Check if running
                if running and running["model"] == model and running["stage"] == stage:
                    row += f" {YELLOW}run{RESET}  "
                else:
                    row += "     -   "
        print(row)

    if running:
        print(f"\n{YELLOW}● 正在运行: {running['model']}/{running['stage']} epoch={running['epoch']} AUC={running['val_auc']:.4f}{RESET}")


def get_running_job():
    """Detect currently running training job from W&B."""
    runs = sorted(glob.glob(os.path.join(WANDB_DIR, "run-*")))
    for run_dir in reversed(runs):
        meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("state") == "running":
            info = load_run(run_dir)
            if info:
                return info
    return None


def print_compare():
    """Compare pre-fix vs post-fix training results."""
    runs = get_all_runs()

    # Find the two most recent complete sed_v2s sessions
    sed_v2s_runs = [r for r in runs if r["model"] == "sed_v2s"]

    # Pre-fix: first attempt (May 13 11:06 - 22:12)
    pre_fix = [r for r in sed_v2s_runs if r["dt"] and r["dt"].day == 13 and r["dt"].hour >= 11]
    # Post-fix: latest (May 19-20)
    post_fix = [r for r in sed_v2s_runs if r["dt"] and r["dt"].day >= 19]

    if not pre_fix or not post_fix:
        print("Cannot find both pre-fix and post-fix runs for comparison.")
        return

    print(f"\n{BOLD}=== sed_v2s 修复前后对比 ==={RESET}\n")
    header = f"{'Stage':<15} {'修复前 AUC':>10} {'修复后 AUC':>10} {'Δ':>8}"
    print(header)
    print("-" * 48)

    for stage in STAGE_ORDER[:5]:
        pre = max([r for r in pre_fix if r["stage"] == stage], key=lambda r: r["dt"], default=None)
        post = max([r for r in post_fix if r["stage"] == stage], key=lambda r: r["dt"], default=None)
        pre_auc = pre["val_auc"] if pre else None
        post_auc = post["val_auc"] if post else None
        if pre_auc and post_auc:
            delta = post_auc - pre_auc
            sign = "+" if delta >= 0 else ""
            print(f"{stage:<15} {pre_auc:>10.4f} {post_auc:>10.4f} {sign}{delta:>+7.4f}")


def print_full_history():
    """Print complete training history grouped by model."""
    runs = get_all_runs()

    print(f"\n{BOLD}=== 完整训练历史 ({runs[0]['dt'].strftime('%m-%d')} ~ {runs[-1]['dt'].strftime('%m-%d')}) ==={RESET}\n")

    # Group by model, then by time period
    for model in MODEL_ORDER:
        model_runs = [r for r in runs if r["model"] == model and r["val_auc"] is not None]
        if not model_runs:
            continue

        # Group into sessions (gaps > 4 hours = new session)
        sessions = []
        session = []
        for r in sorted(model_runs, key=lambda x: x["dt"]):
            if session and (r["dt"] - session[-1]["dt"]).total_seconds() > 14400:
                sessions.append(session)
                session = []
            session.append(r)
        if session:
            sessions.append(session)

        # Take the best AUC per stage per session
        print(f"{BOLD}{model}{RESET}  ({len(sessions)} sessions)")
        header = f"  {'Session':<20}"
        for s in STAGE_ORDER[:5]:
            header += f" {s:>10}"
        print(header)
        print("  " + "-" * 72)

        for i, sess in enumerate(sessions):
            t0 = sess[0]["dt"].strftime("%m-%d %H:%M")
            tf = sess[-1]["dt"].strftime("%H:%M")
            label = f"{t0}~{tf}"
            row = f"  {label:<20}"
            best_per_stage = {}
            for r in sess:
                stage = r["stage"]
                if stage in STAGE_ORDER[:5]:
                    if stage not in best_per_stage or (r["val_auc"] or 0) > (best_per_stage[stage].get("val_auc") or 0):
                        best_per_stage[stage] = {"val_auc": r["val_auc"]}
            for stage in STAGE_ORDER[:5]:
                auc = best_per_stage.get(stage, {}).get("val_auc")
                row += format_auc(auc)
            # Mark sessions with loss explosion
            has_explosion = any(r.get("train_loss", 0) and r["train_loss"] > 100 for r in sess)
            if has_explosion:
                row += f" {RED}LOSS爆炸{RESET}"
            print(row)
        print()


def print_watch():
    """Continuously monitor current training with live refresh."""
    print(f"{BOLD}监控训练进度 (Ctrl+C 退出){RESET}")
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(f"{BOLD}=== 训练监控 ({datetime.now().strftime('%m-%d %H:%M:%S')}) ==={RESET}\n")

            running = get_running_job()
            if running:
                print(f"{YELLOW}● 运行中: {running['model']}/{running['stage']}{RESET}")
                print(f"  Epoch: {running['epoch']} | Step: {running['step']}")
                print(f"  Train Loss: {running['train_loss']:.4f}" if running['train_loss'] else "  Train Loss: -")
                print(f"  Val Loss:   {running['val_loss']:.4f}" if running['val_loss'] else "  Val Loss:   -")
                print(f"  Val AUC:    {running['val_auc']:.4f}" if running['val_auc'] else "  Val AUC:    -")

            # Show checkpoint progress
            for model in MODEL_ORDER:
                ckpts = []
                for stage in STAGE_ORDER[:5]:
                    if check_ckpt_exists(model, stage):
                        ckpts.append(f"{GREEN}{stage}{RESET}")
                    elif running and running["model"] == model and running["stage"] == stage:
                        ckpts.append(f"{YELLOW}{stage}{RESET}")
                    else:
                        ckpts.append(stage)
                done = sum(1 for s in STAGE_ORDER[:5] if check_ckpt_exists(model, s))
                bar = "█" * done + "░" * (5 - done)
                print(f"  {model:<20} [{bar}] {done}/5  {' '.join(ckpts)}")

            time.sleep(30)
    except KeyboardInterrupt:
        print("\n停止监控。")


def main():
    parser = ArgumentParser(description="BirdCLEF training metrics viewer")
    parser.add_argument("--latest", action="store_true", help="Show only current training session")
    parser.add_argument("--compare", action="store_true", help="Compare pre-fix vs post-fix")
    parser.add_argument("--history", action="store_true", help="Show full training history")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor live training")
    args = parser.parse_args()

    # Default to --latest if no args
    if not any([args.latest, args.compare, args.history, args.watch]):
        args.latest = True

    if args.watch:
        print_watch()
    elif args.compare:
        print_compare()
        print_latest_session()
    elif args.history:
        print_full_history()
    elif args.latest:
        print_latest_session()


if __name__ == "__main__":
    main()
