"""
patch_checkpoint_global_stats.py -- inject the missing `global_stats`
(per-channel R/G/B mean+std) into existing PathMamba checkpoints, WITHOUT
retraining or touching a GPU.

Why this works without retraining:
    train_pathmamba_v2_final.py seeds `random.seed(42)` / `np.random.seed(42)`
    at MODULE LOAD time (before main() runs), and the held-out split uses its
    own private `random.Random(42)` instance rather than the global `random`
    module. That means the global random state at the point main() calls
    `sample_rgb_stats(all_pairs)` is untouched by anything before it -- it's
    simply "right after random.seed(42)". So as long as we rebuild `all_pairs`
    in the exact same content/order main() does (same run_data_audit() +
    held-out split + empty-mask filter), calling sample_rgb_stats(all_pairs)
    here reproduces the *exact* global_stats value your original training run
    computed and used -- just without running any of the GPU training itself.

Usage:
    python patch_checkpoint_global_stats.py \
        /home/outputs/stage_a_rgbfinal_checkpoint.pth \
        /home/outputs/stage_a0_rgbfinal_checkpoint.pth \
        [...more checkpoint paths...]

Each checkpoint is backed up to <path>.bak before being overwritten. Safe to
re-run: if a checkpoint already has global_stats, it's left untouched (unless
--force is passed).

IMPORTANT: run this from wherever `import route_resilience` (or
`train_pathmamba_v2_final` directly) resolves for you, with DATA_ROOT set to
the SAME data directory your original training run used -- otherwise
run_data_audit() will build a different all_pairs list and the recomputed
stats won't match what the checkpoint was actually trained with.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

try:
    from route_resilience.train_pathmamba_v2_final import run_data_audit, sample_rgb_stats
except ImportError:
    # fallback if run from inside the route_resilience directory itself
    from train_pathmamba_v2_final import run_data_audit, sample_rgb_stats


def _held_out_split(pairs, n_held_out, seed=42):
    """Exact copy of the closure inside main() -- must stay in sync with it."""
    import random
    if not pairs:
        return [], []
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = min(n_held_out, len(shuffled))
    return shuffled[:n], shuffled[n:]


def _filter_empty_masks(pairs, name):
    """Exact copy of the closure inside main() -- must stay in sync with it."""
    kept = []
    n_dropped = 0
    for img_path, mask_path in pairs:
        mask = np.array(Image.open(mask_path).convert("L"))
        if (mask > 127).any():
            kept.append((img_path, mask_path))
        else:
            n_dropped += 1
    print(f"  Empty-mask filter ({name}): dropped {n_dropped} of {len(pairs)} training pairs")
    return kept


def recompute_global_stats():
    """Rebuilds all_pairs exactly as main() does, then computes global_stats.
    Must be called before anything else touches the global `random` module,
    same as in main() -- run_data_audit()/held-out split/filtering don't
    consume global random state (see module docstring), so this is safe to
    call standalone."""
    N_HELD_OUT_PER_DATASET = 40

    print("Running data audit (same as training main())...")
    dg_pairs, sn_pairs_all, mass_pairs, mass_train_pairs, urban_binary_pairs, sn5_mumbai_pairs = run_data_audit()

    dg_held_out, dg_train = _held_out_split(dg_pairs, N_HELD_OUT_PER_DATASET)
    sn_held_out, sn_train = _held_out_split(sn_pairs_all, N_HELD_OUT_PER_DATASET)
    urban_binary_held_out, urban_binary_train = _held_out_split(urban_binary_pairs, N_HELD_OUT_PER_DATASET)
    sn5_mumbai_held_out, sn5_mumbai_train = _held_out_split(sn5_mumbai_pairs, N_HELD_OUT_PER_DATASET)

    dg_train = _filter_empty_masks(dg_train, "DeepGlobe")
    sn_train = _filter_empty_masks(sn_train, "SpaceNet")
    if mass_train_pairs:
        mass_train_pairs = _filter_empty_masks(mass_train_pairs, "Massachusetts")
    if urban_binary_train:
        urban_binary_train = _filter_empty_masks(urban_binary_train, "Urban-binary")

    all_pairs = (dg_train + sn_train + mass_train_pairs
                + urban_binary_train + sn5_mumbai_train)
    print(f"all_pairs: {len(all_pairs)} total training pairs")

    print("Computing global normalization stats (must match training exactly)...")
    global_stats = sample_rgb_stats(all_pairs)
    print(f"Recomputed global_stats: {global_stats}")
    return global_stats


def patch_checkpoint(path, global_stats, force=False):
    import torch

    if not os.path.exists(path):
        print(f"[skip] {path} does not exist")
        return

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print(f"[skip] {path} is not a dict-style checkpoint (looks like a bare state_dict) -- "
              f"can't attach global_stats to it. Re-save it as {{'model': state_dict, ...}} first.")
        return

    if ckpt.get("global_stats") is not None and not force:
        print(f"[skip] {path} already has global_stats -- pass --force to overwrite")
        return

    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f"  backed up original to {backup_path}")
    else:
        print(f"  backup already exists at {backup_path} (not overwriting it)")

    ckpt["global_stats"] = global_stats
    torch.save(ckpt, path)
    print(f"[patched] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="Path(s) to .pth checkpoint(s) to patch")
    ap.add_argument("--force", action="store_true", help="Overwrite global_stats even if already present")
    args = ap.parse_args()

    global_stats = recompute_global_stats()

    print()
    for ckpt_path in args.checkpoints:
        patch_checkpoint(ckpt_path, global_stats, force=args.force)


if __name__ == "__main__":
    main()
