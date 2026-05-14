#!/usr/bin/env python3
"""Pre-download all timm model weights needed for training, with retries."""
import timm
import time
import sys
import os
import torch

CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")

# Mapping from timm model name to cache filename (timm naming convention)
MODELS = [
    "tf_efficientnetv2_s_in21k",
    "tf_efficientnet_b3_ns",
    "seresnext26t_32x4d",
    "resnet34d",
    "tf_efficientnet_b0_ns",
]


def clear_corrupted_cache(model_name):
    """Remove any cached file for this model so we don't reuse a truncated download."""
    # Try to find the cache file by creating model with pretrained=False, then
    # looking at its pretrained_cfg to get the URL filename.
    # Simpler: list cache dir and match by model name prefix.
    prefix = model_name.replace("_", "-")  # rough, but works for these models
    if os.path.isdir(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            if fname.startswith(model_name[:20]) and fname.endswith(".pth"):
                fpath = os.path.join(CACHE_DIR, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                if size_mb < 1:
                    print(f"  Removing corrupted cache: {fname} ({size_mb:.1f} MB)")
                    os.remove(fpath)


failed = []
for model_name in MODELS:
    for attempt in range(5):
        try:
            print(f"Downloading {model_name} (attempt {attempt+1})...")
            clear_corrupted_cache(model_name)
            timm.create_model(model_name, pretrained=True)
            print(f"  OK: {model_name}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 4:
                time.sleep(10)
            else:
                print(f"  GAVE UP on {model_name} after 5 attempts")
                failed.append(model_name)

if failed:
    print(f"\nFailed to download: {failed}")
    sys.exit(1)
else:
    print("\nAll weights downloaded successfully.")
