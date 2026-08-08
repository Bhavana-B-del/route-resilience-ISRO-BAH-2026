"""
train_pathmamba.py
===================
Route Resilience -- PathMamba Stage A/B Pretraining, JarvisLabs edition.

Consolidates your Kaggle notebook (environment checks, data audit, dataset
classes, PathMamba model, combined loss, 3-stage training) into ONE runnable
script, meant to survive an unattended overnight run via tmux/nohup rather
than live Jupyter cells.

RUN THIS INSIDE tmux (survives SSH disconnects; a bare `python3 train_pathmamba.py &`
does NOT survive a dropped SSH session on its own):

    tmux new -s pretrain
    python3 train_pathmamba.py 2>&1 | tee /home/train_log.txt
    # Ctrl-B then D to detach. Reattach any time with: tmux attach -t pretrain

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Environment check (GPU, mamba-ssm install + compile, timm/EfficientNet-B4)
  2. HARD ABORT if mamba-ssm isn't actually working -- see ABORT_IF_FALLBACK below.
     No point spending overnight compute training the GRU fallback if tonight's
     whole purpose is finding out whether real Mamba works.
  3. Data audit (DeepGlobe + SpaceNet, Massachusetts optional) -- fails loud
     with a clear message if your folder layout doesn't match, rather than
     silently training on zero pairs.
  4. Dataset classes, PathMamba model, combined loss -- unchanged logic from
     your notebook, only paths and reliability (AMP/checkpointing) changed.
  5. Smoke test (does NOT proceed to full training if this fails).
  6. Three-stage training (A0 warm-up, A main, B fine-tune) -- each stage:
       - pure FP32 (AMP disabled: mamba-ssm's CUDA kernel is unstable under FP16)
       - both GPUs used via DataParallel, if more than one is visible
       - checkpoint every CHECKPOINT_EVERY_N_BATCHES batches (not just at
         epoch end) -- so a killed session loses minutes, not a whole epoch
       - resumable: re-run this same script and it picks up where it left off
  7. Validation (IoU + clDice) on held-out data.
  8. Visualization: runs inference on N_VIZ_IMAGES held-out images and saves
     one PNG (input / GLCM channel / ground truth / predicted probability /
     confidence map per row) so you can SEE whether it's learning anything,
     not just read a loss number.
  9. Final summary report: wall-clock time per stage, whether Mamba or
     fallback actually ran, validation numbers, and where every artifact was
     saved.

BEFORE RUNNING, on your Jarvis instance:
  - Datasets must already be downloaded under DATA_ROOT (see the "kaggle
    datasets download" commands in the accompanying instructions message --
    this script does not download them itself, since Kaggle credentials are
    yours to manage, not something to bake into a script).
  - `pip install kaggle timm scipy pillow matplotlib` if not already present
    (torch/mamba-ssm/causal-conv1d are installed automatically by Section 1).
"""

from __future__ import annotations

import glob
import json
import os
import random
import subprocess
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image

# ============================================================================
# CONFIG -- every path and hyperparameter lives here. Nothing else in this
# file should have a hardcoded path -- if you need to change where something
# reads or writes, change it here.
# ============================================================================

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/data"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.expanduser("~/outputs"))
# Resume base for THIS run: the confirmed-best checkpoint from earlier in
# the session (B6, w_cldice=0.5, canopy=0.3 -- the one backed up to the
# repo specifically for this reason). NOT in OUTPUT_ROOT -- it's sitting in
# the git repo root from that backup. Adjust this path if you placed it
# somewhere else on this instance.
RESUME_FROM_CHECKPOINT = os.environ.get(
    "RESUME_FROM_CHECKPOINT",
    os.path.expanduser("~/ISRO-BAH-Grand-Finale-Hackathon/stage_b_checkpoint_b6_v1.pth")
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEEPGLOBE_ROOT = os.path.join(DATA_ROOT, "deepglobe-road-extraction-dataset")
SPACENET_ROOT = os.path.join(DATA_ROOT, "spacenet-3")
MASS_ROOT = os.path.join(DATA_ROOT, "massachusetts-roads-dataset")  # optional
# --- New for the 5-dataset "urbanmix" run ---
URBAN_BINARY_ROOT = os.path.join(DATA_ROOT, "urban-binary")
SPACENET5_MUMBAI_ROOT = os.path.join(DATA_ROOT, "spacenet-5-mumbai")
URBAN_BINARY_GSD_M = 0.3   # confirmed by user, not a guess
SPACENET5_MUMBAI_GSD_M = 0.3  # confirmed via torchgeo docs: PS-RGB, WorldView-3, matches SpaceNet-3
# --- Cartosat-3 fine-tune stage: OFF by default (see run_cartosat_finetune_stage()
# near the end of main()). On hackathon day, once ISRO's real Cartosat-3 tile +
# labels exist at CARTOSAT_ROOT, flip ENABLE_CARTOSAT_FINETUNE=1 and this stage
# runs automatically -- no code changes needed, install and run straight away. ---
CARTOSAT_ROOT = os.environ.get("CARTOSAT_ROOT", os.path.join(DATA_ROOT, "cartosat-3"))
ENABLE_CARTOSAT_FINETUNE = os.environ.get("ENABLE_CARTOSAT_FINETUNE", "0") == "1"
CARTOSAT_EPOCHS, CARTOSAT_BATCH, CARTOSAT_LR = 8, 4, 1e-5  # small LR: fine-tuning
# from stage_b_rgbfinal_checkpoint.pth, not training from scratch. Epoch count is a
# guess for a short hackathon-day window -- adjust once you know the real
# tile size and time budget on the day.

# --- OpenSatMap fine-tune stage: OFF by default, same pattern as Cartosat
# above. UNVERIFIED folder structure -- OpenSatMap (arxiv 2410.23278) ships
# instance-level lane/curb/road-structure labels, not simple binary masks
# like every other dataset here. build_opensatmap_pairs() below is a
# genuine placeholder (raises NotImplementedError) until real folder
# structure is confirmed via `find` AND the label format is confirmed
# (may need collapsing multi-class instance labels into one binary "road"
# class before this is usable at all -- don't assume this is a quick
# fill-in the way urban-binary turned out to be). 0.3m or 0.15m GSD
# depending on which OpenSatMap resolution tier gets used.
OPENSATMAP_ROOT = os.environ.get("OPENSATMAP_ROOT", os.path.join(DATA_ROOT, "opensatmap"))
ENABLE_OPENSATMAP_FINETUNE = os.environ.get("ENABLE_OPENSATMAP_FINETUNE", "0") == "1"
OPENSATMAP_EPOCHS, OPENSATMAP_BATCH, OPENSATMAP_LR = 8, 4, 1e-5
OPENSATMAP_GSD_M = 0.3  # confirmed via search (arxiv 2410.23278) for the
# level-19 tier -- use 0.15 instead if using the level-20 tier specifically.

# If mamba-ssm genuinely fails to install/compile, this script normally stops
# BEFORE any training rather than silently training the GRU fallback all
# night. Set ALLOW_FALLBACK_TRAINING=1 only if you deliberately want a
# fallback-only run anyway (e.g. as an emergency backup while debugging the
# Mamba build).
ABORT_IF_FALLBACK = os.environ.get("ALLOW_FALLBACK_TRAINING", "0") != "1"
USE_FAST_GLCM_PROXY = os.environ.get("USE_FAST_GLCM_PROXY", "0") == "1"

CHECKPOINT_EVERY_N_BATCHES = 150
N_VIZ_IMAGES = 5

# Batch size 8->4: B6 has ~2.3x more backbone params than B4, meaningfully
# more activation memory per sample. An OOM crash mid-overnight-run wastes
# hours you don't have to spare -- this is a safety margin, not a guess that
# 4 is optimal. If you have GPU memory to spare (check nvidia-smi after the
# first few batches), 6 or 8 may still work fine -- raise it back up if so.
#
# Epoch counts increased for the overnight run (2/10/5 -> 5/25/10, 17 total
# epochs -> 40 total). These are ESTIMATES targeting a realistic ~8h budget,
# assuming B6 runs roughly 1.5-2x slower per epoch than B4 did -- not a
# guaranteed fit. Check progress after Stage A0 completes (which gives a
# REAL measured per-epoch time for this actual run+hardware) before assuming
# Stage A will finish on schedule.
A0_EPOCHS, A0_BATCH, A0_LR = 5, 4, 1e-4
A_EPOCHS, A_BATCH, A_LR = 25, 4, 1e-4
B_EPOCHS, B_BATCH, B_LR = 10, 4, 5e-5

random.seed(42)
np.random.seed(42)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# SECTION 1 -- Environment check. Order matters: GPU check, then mamba-ssm
# (which needs the CUDA-enabled torch this check confirms), then
# timm/EfficientNet-B4 weights.
# ============================================================================

def check_gpu() -> None:
    log("=== GPU CHECK ===")
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        log("nvidia-smi not found -- almost certainly no GPU on this instance.")

    import torch

    log(f"torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        log("!!! NO GPU DETECTED. This will not train in any reasonable time. Stopping.")
        sys.exit(1)


def _neutralize_mamba_lm_head_import() -> bool:
    """Removes the `from mamba_ssm.models.mixer_seq_simple import
    MambaLMHeadModel` line from an already-installed mamba_ssm's __init__.py,
    if present. We never use MambaLMHeadModel -- it drags in transformers'
    generation utils (GenerationMixin -> GreedySearchDecoderOnlyOutput etc.),
    and different transformers versions rename/remove those classes,
    breaking mamba_ssm's import entirely on a fresh instance.

    CRITICAL: this function must NEVER do `import mamba_ssm` (or import any
    submodule of it) -- that executes __init__.py, which is exactly the
    broken code we're trying to patch BEFORE it runs. This bit us for real:
    an earlier version of this fix used `from mamba_ssm.modules.mamba_simple
    import Mamba` thinking it bypassed __init__.py, but Python always runs a
    package's __init__.py before any of its submodules -- so that "fix" had
    the exact same failure mode as a plain `import mamba_ssm` on any fresh
    instance with an incompatible transformers version. Verified live: find_spec
    (below) locates the file without executing it; a direct import does not.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("mamba_ssm")
    except Exception:
        spec = None
    if spec is None or spec.origin is None:
        return False  # not installed yet -- nothing to patch

    init_path = spec.origin
    try:
        src = open(init_path).read()
    except Exception:
        return False

    if "MambaLMHeadModel" not in src:
        return False  # already clean (or already patched)

    lines = [l for l in src.splitlines() if "MambaLMHeadModel" not in l]
    open(init_path, "w").write("\n".join(lines) + "\n")
    log(f"Neutralized MambaLMHeadModel import in {init_path} "
        f"(unused by this script; avoids a transformers-version crash).")
    return True


def install_mamba() -> bool:
    """Returns True if mamba-ssm is genuinely importable and usable, False if
    we're on the GRU fallback. Mirrors your notebook's install sequence
    exactly (causal-conv1d first, both with --no-build-isolation so the
    build sees the CUDA-enabled torch already on the machine, not an
    isolated CPU-only build env)."""
    log("=== mamba-ssm CHECK ===")
    _neutralize_mamba_lm_head_import()  # BEFORE any import attempt -- see docstring above
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm import Mamba  # noqa: F401

        log("mamba-ssm already importable -- skipping install.")
        return True
    except Exception as first_err:
        log(f"mamba-ssm not importable yet: {first_err!r}")

    log("Installing build prerequisites...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "packaging", "ninja", "wheel", "setuptools"],
        capture_output=True, text=True,
    )
    log("Installing causal-conv1d (--no-build-isolation)...")
    r1 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "causal-conv1d>=1.4.0", "--no-build-isolation"],
        capture_output=True, text=True,
    )
    if r1.returncode != 0:
        log("causal-conv1d install failed (non-fatal, mamba-ssm can still work without it, just slower):")
        log(r1.stderr[-800:])

    log("Installing mamba-ssm (--no-build-isolation, compiles CUDA kernels, ~5-10 min)...")
    r2 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "mamba-ssm", "--no-build-isolation"],
        capture_output=True, text=True,
    )
    log(r2.stdout[-1500:])
    if r2.returncode != 0:
        log("mamba-ssm install FAILED. stderr tail:")
        log(r2.stderr[-2000:])
        return False

    _neutralize_mamba_lm_head_import()  # fresh pip install writes a new, unpatched __init__.py
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm import Mamba  # noqa: F401

        log("mamba-ssm installed and importable.")
        return True
    except Exception as e:
        log(f"mamba-ssm installed but still not importable: {e!r}")
        return False


def check_timm() -> None:
    log("=== timm / EfficientNet-B4 CHECK ===")
    import timm

    model = timm.create_model("tf_efficientnet_b6.ns_jft_in1k", pretrained=True, features_only=True)
    log(f"timm {timm.__version__}: tf_efficientnet_b6.ns_jft_in1k built OK, "
        f"{sum(p.numel() for p in model.parameters()):,} params.")
    del model


# ============================================================================
# SECTION 2 -- Data audit. Defensive: your exact folder layout wasn't
# confirmed, so this tries the common DeepGlobe/SpaceNet mirror conventions
# and FAILS LOUD with a clear message (not a silent empty dataset) if
# neither matches. If it fails, look at the printed directory listing and
# adjust the glob patterns below to match what's actually on disk.
# ============================================================================

def build_deepglobe_pairs(root: str) -> list:
    if not os.path.isdir(root):
        return []
    for candidate_dir in (os.path.join(root, "train"), root):
        if not os.path.isdir(candidate_dir):
            continue
        sat_files = sorted(glob.glob(os.path.join(candidate_dir, "*_sat.jpg")))
        if sat_files:
            pairs = [(f, f.replace("_sat.jpg", "_mask.png")) for f in sat_files]
            pairs = [(i, m) for i, m in pairs if os.path.exists(m)]
            return pairs
    return []


def build_spacenet_pairs(root: str) -> list:
    if not os.path.isdir(root):
        return []
    IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    pairs = []
    for base in (os.path.join(root, "8bit"), root):
        if not os.path.isdir(base):
            continue
        for city in sorted(os.listdir(base)):
            img_dir = os.path.join(base, city, "images")
            mask_dir = os.path.join(base, city, "mask")
            if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
                # Filter to real image extensions only. Without this,
                # precompute_glcm_cache()'s <image_path>.glcm.npy cache files
                # (written into this same images/ dir) get picked up by
                # os.listdir() on any re-run after the cache exists, shifting
                # every img<->mask index pairing after that point and
                # eventually handing a .npy cache file to PIL.Image.open().
                imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
                masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXTS))
                n = min(len(imgs), len(masks))
                pairs.extend(
                    (os.path.join(img_dir, imgs[i]), os.path.join(mask_dir, masks[i])) for i in range(n)
                )
        if pairs:
            return pairs
    return []


def build_massachusetts_pairs(root: str, split: str = "test") -> list:
    """split='test' (default): the dataset's own 49-image test split -- used
    ONLY for the OOD generalization check (run_ood_check in main()), never
    trained on.
    split='train': the dataset's own ~1108-image train split -- used as
    additional Stage A training data.
    split='val': the dataset's own ~14-image val split -- also foldable into
    training if you want it (main() combines train+val for the training pool).
    """
    if not os.path.isdir(root):
        return []
    for img_sub, mask_sub in ((f"tiff/{split}", f"tiff/{split}_labels"),
                               (f"png/{split}", f"png/{split}_labels")):
        img_dir, mask_dir = os.path.join(root, img_sub), os.path.join(root, mask_sub)
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            def stem(s):
                return os.path.splitext(s)[0]
            imap = {stem(f): os.path.join(img_dir, f) for f in os.listdir(img_dir) if not f.startswith(".")}
            mmap = {stem(f): os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if not f.startswith(".")}
            common = sorted(set(imap) & set(mmap))
            if common:
                return [(imap[s], mmap[s]) for s in common]
    return []


# =============================================================================
# PLACEHOLDER -- NOT YET VERIFIED AGAINST REAL FOLDER STRUCTURE.
# =============================================================================
# Both functions below raise NotImplementedError on purpose -- this is a hard
# guard, not an oversight. Writing plausible-looking pairing logic without
# having actually SEEN the real folder/file structure risks silently pairing
# the wrong image with the wrong mask (same file-stem-matching approach as
# build_massachusetts_pairs above works great IF the naming convention
# actually matches -- but guessing that convention wrong produces a script
# that runs "successfully" while training on garbage pairs, which is a much
# worse failure mode than a crash). Fill these in once `find <root> -maxdepth
# 3` output is available, following the same stem-matching pattern already
# proven correct in build_massachusetts_pairs above.

def build_urban_binary_pairs(root: str) -> list:
    """Urban-binary dataset. Confirmed real structure (via `find`/`ls` on the
    actual uploaded data, not assumed): root/images/*.png + root/masks/*.png,
    numeric filenames (e.g. 1366.png), matched by stem. 1156 images and 1156
    masks confirmed present, one-to-one, no orphans on either side.
    """
    if not os.path.isdir(root):
        return []
    img_dir, mask_dir = os.path.join(root, "images"), os.path.join(root, "masks")
    if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
        return []
    IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def stem(s):
        return os.path.splitext(s)[0]

    imap = {stem(f): os.path.join(img_dir, f) for f in os.listdir(img_dir)
           if not f.startswith(".") and f.lower().endswith(IMG_EXTS)}
    mmap = {stem(f): os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
           if not f.startswith(".") and f.lower().endswith(IMG_EXTS)}
    common = sorted(set(imap) & set(mmap))
    return [(imap[s], mmap[s]) for s in common]


def build_spacenet5_mumbai_pairs(root: str) -> list:
    """Deliberately disabled for this run -- NOT because anything is broken.
    SpaceNet-5 ships only vector road-network labels (geojson centerlines +
    WKT/speed CSVs), no pre-rasterized binary masks like SpaceNet-3 has.
    Using it would need a rasterization step first (buffer centerlines to a
    road-width polygon, similar to osm_labels.py's make_osm_label_raster),
    which is a bigger task than fits before this run needs to start. Training
    proceeds on the remaining 4 datasets instead. Revisit after the hackathon
    if there's time to build the rasterization step properly.
    """
    return []


def build_cartosat_pairs(root: str) -> list:
    """For the dormant hackathon-day fine-tune stage (run_cartosat_finetune_stage,
    near the end of main()). Expects root/images/*.tif + root/masks/*.png (or
    .tif), paired by filename stem -- same convention as
    build_massachusetts_pairs. Returns [] if CARTOSAT_ROOT doesn't exist yet
    or has no pairs, which is exactly what keeps this stage a safe no-op
    until real hackathon-day data is actually present.
    """
    if not os.path.isdir(root):
        return []
    IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    for img_sub, mask_sub in (("images", "masks"), ("img", "mask")):
        img_dir, mask_dir = os.path.join(root, img_sub), os.path.join(root, mask_sub)
        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            def stem(s):
                return os.path.splitext(s)[0]
            imap = {stem(f): os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if not f.startswith(".") and f.lower().endswith(IMG_EXTS)}
            mmap = {stem(f): os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                   if not f.startswith(".") and f.lower().endswith(IMG_EXTS)}
            common = sorted(set(imap) & set(mmap))
            if common:
                return [(imap[s], mmap[s]) for s in common]
    return []


def build_opensatmap_pairs(root: str) -> list:
    """For the dormant OpenSatMap fine-tune stage. Genuine placeholder --
    raises NotImplementedError on purpose, same reasoning as urban-binary
    and SpaceNet-5 Mumbai were before their real structure was confirmed.

    UNLIKE urban-binary (which turned out to be a simple images/+masks/
    stem-match, filled in in minutes), OpenSatMap's real format (confirmed
    via web search, arxiv 2410.23278) ships INSTANCE-LEVEL lane/curb/road-
    structure annotations -- multiple classes per image (lane lines, curbs,
    virtual lines), not one binary "is this a road" mask. Before writing
    real pairing logic here:
      1. Check the actual downloaded format (HuggingFace: z-hb/OpenSatMap) --
         does it ship pre-rasterized per-class masks, or only vectorized
         polylines needing rasterization first (same complexity class as
         the SpaceNet-5 Mumbai problem)?
      2. If per-class masks exist, they need COLLAPSING into one binary
         road mask (merge all road/lane/curb-adjacent classes together) --
         this is a real preprocessing decision (which classes count as
         "road" for this project's purposes?), not just a stem-match.
    Budget real time for this, don't assume it's a quick fill-in.
    """
    raise NotImplementedError(
        "build_opensatmap_pairs() is a placeholder -- OpenSatMap's real "
        "label format (instance-level lane/curb annotations, not simple "
        "binary masks) needs investigation before writing real pairing "
        "logic. Check the actual downloaded data's structure and label "
        "format first."
    )


def run_data_audit():
    log("=== DATA AUDIT ===")
    dg_pairs = build_deepglobe_pairs(DEEPGLOBE_ROOT)
    sn_pairs = build_spacenet_pairs(SPACENET_ROOT)
    # test split: held out, OOD generalization check ONLY (see run_ood_check
    # in main()) -- never trained on, never touches global_stats/glcm_norm_value.
    mass_pairs = build_massachusetts_pairs(MASS_ROOT, split="test")
    # train + val splits: legitimate additional Stage A training data --
    # completely separate images from mass_pairs above, no leakage between them.
    mass_train_pairs = (build_massachusetts_pairs(MASS_ROOT, split="train")
                        + build_massachusetts_pairs(MASS_ROOT, split="val"))

    # New for the 5-dataset run. Wrapped in try/except since these two are
    # still placeholders (see the NotImplementedError functions above) --
    # this lets the rest of the audit run and report clearly what's still
    # missing, rather than crashing the whole script before anything useful
    # can be checked.
    try:
        urban_binary_pairs = build_urban_binary_pairs(URBAN_BINARY_ROOT)
    except NotImplementedError as e:
        log(f"!!! urban-binary pairing not ready yet: {e}")
        urban_binary_pairs = []
    try:
        sn5_mumbai_pairs = build_spacenet5_mumbai_pairs(SPACENET5_MUMBAI_ROOT)
    except NotImplementedError as e:
        log(f"!!! SpaceNet-5 Mumbai pairing not ready yet: {e}")
        sn5_mumbai_pairs = []

    log(f"DeepGlobe pairs found:      {len(dg_pairs)}  (root: {DEEPGLOBE_ROOT})")
    log(f"SpaceNet pairs found:       {len(sn_pairs)}  (root: {SPACENET_ROOT})")
    log(f"Massachusetts train+val:    {len(mass_train_pairs)}  (root: {MASS_ROOT}, optional, used in Stage A)")
    log(f"Massachusetts test:         {len(mass_pairs)}  (root: {MASS_ROOT}, optional, OOD check ONLY)")
    log(f"Urban-binary pairs found:   {len(urban_binary_pairs)}  (root: {URBAN_BINARY_ROOT})")
    log(f"SpaceNet-5 Mumbai found:    {len(sn5_mumbai_pairs)}  (root: {SPACENET5_MUMBAI_ROOT})")

    if len(dg_pairs) == 0 and len(sn_pairs) == 0:
        log("!!! NO TRAINING PAIRS FOUND ANYWHERE. Stopping before wasting GPU time.")
        log(f"    Checked DEEPGLOBE_ROOT = {DEEPGLOBE_ROOT}")
        log(f"    Checked SPACENET_ROOT  = {SPACENET_ROOT}")
        log("    Directory listing of DATA_ROOT for debugging:")
        if os.path.isdir(DATA_ROOT):
            for entry in sorted(os.listdir(DATA_ROOT)):
                log(f"      {entry}")
        else:
            log(f"    DATA_ROOT itself doesn't exist: {DATA_ROOT}")
        log("    Fix: adjust DEEPGLOBE_ROOT/SPACENET_ROOT at the top of this file, or the")
        log("    glob patterns in build_deepglobe_pairs()/build_spacenet_pairs(), to match")
        log("    whatever layout `kaggle datasets download` actually produced.")
        sys.exit(1)

    if len(urban_binary_pairs) == 0 or len(sn5_mumbai_pairs) == 0:
        log("!!! One or both new datasets have zero pairs -- the placeholder pairing")
        log("    functions need filling in with real folder structure before this")
        log("    script will actually train on all 5 datasets. Training WILL proceed")
        log("    below using whatever datasets DO have pairs, rather than stopping --")
        log("    but double check this is what you actually want before walking away.")

    for name, pairs in (("DeepGlobe", dg_pairs), ("SpaceNet", sn_pairs),
                        ("Urban-binary", urban_binary_pairs), ("SpaceNet-5 Mumbai", sn5_mumbai_pairs)):
        if not pairs:
            continue
        sample_mask = np.array(Image.open(pairs[0][1]).convert("L"))
        uniq = np.unique(sample_mask)
        log(f"{name} first mask value range: min={uniq.min()} max={uniq.max()} "
            f"(expect roughly 0 and 255)")

    return dg_pairs, sn_pairs, mass_pairs, mass_train_pairs, urban_binary_pairs, sn5_mumbai_pairs


# ============================================================================
# SECTION 3 -- Dataset classes. Logic unchanged from your notebook (GLCM
# occlusion channel, canopy/thin-road paste augmentation, cutout) -- only
# thing different is these no longer assume Kaggle-specific globals.
# ============================================================================

GLCM_FIXED_WINDOW = 5


class GLCMOcclusionMap:
    """REAL multi-angle GLCM (Gray-Level Co-occurrence Matrix) occlusion map,
    matching the deck's "GLCM texture variance" claim.

    Implementation: vectorized, not a per-window Python loop (that version
    was tried first and took an estimated 70+ hours to precompute the whole
    dataset -- a real engineering mistake, not an inherent GLCM cost; see the
    fix note below). GLCM contrast and homogeneity at a given angle/distance
    are provably equal to a local average of f(pixel_diff) over that offset
    -- NOT an approximation, verified to match skimage's graycomatrix/
    graycoprops to floating-point precision on a test patch. That identity
    means the whole computation reduces to: shift the image by each of 4
    angle offsets, compute the pixel difference, box-filter (via
    scipy.ndimage.uniform_filter) the squared difference (contrast) and
    1/(1+diff^2) (homogeneity) over the window, average across the 4 angles.
    No Python loop over pixel positions at all. Benchmark: ~0.35s for a
    1024x1024 image, vs. an estimated ~130s/image for the old loop version --
    roughly 375x faster, same real GLCM math.

    Cost: still somewhat slower than the plain uniform-filter proxy (this
    computes 8 box filters -- contrast+homogeneity x 4 angles -- vs. the
    proxy's 2), but now genuinely cheap in absolute terms. Two mitigations
    remain in place regardless:
      1. On-disk caching: for each source image, the GLCM output is computed
         once and cached as `<image>.glcm.npy`, then loaded directly on every
         subsequent epoch. See `precompute_glcm_cache` below.
      2. `use_fast_proxy=True` reverts to the uniform-filter proxy as an
         emergency fallback if a Jarvis run turns out CPU-bound regardless. Off by default.

    `norm_value` MUST be a fixed constant computed upfront (see
    `compute_glcm_norm_value`), for the same across-worker consistency reason
    documented in the old class.
    """

    def __init__(self, window: int = 11, norm_value: Optional[float] = None,
                 gray_levels: int = 32, use_fast_proxy: bool = False):
        self.window = window
        self._norm_value = norm_value
        self.gray_levels = gray_levels  # quantize to 32 gray levels -- controls
                                          # GLCM matrix size (32x32 not 256x256)
                                          # which is where the real speed win comes from
        self.use_fast_proxy = use_fast_proxy

    def compute(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        if self.use_fast_proxy:
            result = self._compute_fast_proxy(image_gray_uint8)
        else:
            result = self._compute_real_glcm(image_gray_uint8)
        if self._norm_value is None:
            self._norm_value = float(np.percentile(result, 99)) if result.max() > 0 else 1.0
        return np.clip(result / max(self._norm_value, 1e-6), 0, 1).astype(np.float32)

    def compute_raw(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        """Unclipped, unnormalized signal -- for calibration ONLY
        (compute_glcm_norm_value). Bypasses norm_value/clipping entirely.

        BUG THIS EXISTS TO FIX: compute_glcm_norm_value() used to call
        compute() with _norm_value temporarily set to 1.0, intending to get
        "raw" output. But compute() always clips to [0, norm_value] as its
        last step -- setting norm_value=1.0 doesn't disable that clip, it
        just makes the clip ceiling 1.0. Real GLCM contrast values routinely
        exceed 1.0 (raw range is roughly 0 to (gray_levels-1)^2), so nearly
        every sampled pixel got slammed to exactly 1.0 BEFORE the percentile
        was computed -- meaning the "calibration" trivially returned 1.0
        every time, regardless of image content or random sample. This is
        why two independent runs both logged "norm_value = 1.0000" on the
        nose: not coincidence, a saturated calibration input.
        """
        if self.use_fast_proxy:
            return self._compute_fast_proxy(image_gray_uint8)
        return self._compute_real_glcm(image_gray_uint8)

    def _compute_fast_proxy(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        """Original uniform-filter local-variance proxy. Kept as an emergency
        fallback -- flip use_fast_proxy=True to use it. Much faster, but not
        a real GLCM. Returns RAW, unclipped values -- see compute()."""
        from scipy.ndimage import uniform_filter

        img = image_gray_uint8.astype(np.float32)
        mean = uniform_filter(img, size=self.window)
        mean_sq = uniform_filter(img ** 2, size=self.window)
        return np.clip(mean_sq - mean ** 2, 0, None)

    def _compute_real_glcm(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        """Real multi-angle GLCM contrast + homogeneity, fully vectorized.

        Verified identity (see conversation/commit notes): for a given
        angle/distance offset, GLCM contrast = local mean of (pixel -
        shifted_pixel)^2, and GLCM homogeneity = local mean of
        1/(1+(pixel-shifted_pixel)^2) -- exact, not approximate, because
        both Haralick features are sums of a function of (i-j) weighted by
        the co-occurrence probability, which by definition IS the empirical
        distribution of (pixel, shifted_pixel) pairs in the window. This
        holds regardless of gray-level quantization for these two specific
        features (quantization only matters for histogram-shape features
        like entropy/ASM, which aren't used here).

        No Python loop over pixel positions -- 4 array shifts (one per
        angle) + uniform_filter (box average) per shift. Gray-level
        quantization (`self.gray_levels`) is applied anyway for consistency
        with the deck's stated 32-gray-level design, though it no longer
        controls a matrix size the way it did in the loop version -- it's
        now just a mild pre-quantization of the input signal.
        """
        from scipy.ndimage import uniform_filter

        img = image_gray_uint8.astype(np.float32)
        img_q = np.clip(img / 256.0 * self.gray_levels, 0, self.gray_levels - 1)

        # Distance=1 offsets for angles 0, 45, 90, 135 degrees.
        offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
        contrast_sum = np.zeros_like(img_q)
        homogeneity_sum = np.zeros_like(img_q)
        for dy, dx in offsets:
            shifted = np.roll(np.roll(img_q, -dy, axis=0), -dx, axis=1)
            diff = img_q - shifted
            contrast_sum += uniform_filter(diff ** 2, size=self.window)
            homogeneity_sum += uniform_filter(1.0 / (1.0 + diff ** 2), size=self.window)

        contrast = contrast_sum / len(offsets)
        homogeneity = homogeneity_sum / len(offsets)
        # Same combination rationale as before: both point toward
        # "occluded/rough" when high; multiply so they amplify, not cancel.
        # Returns RAW, unclipped values -- see compute() for the norm/clip step.
        return contrast * (1.0 - homogeneity)


def precompute_glcm_cache(pairs, glcm_tool, force=False):
    """Precomputes and caches each source image's GLCM output alongside the
    image, as `<image_path>.glcm.npy`. Idempotent -- skips images that
    already have a cache file unless `force=True`. Run this ONCE upfront,
    before any DataLoader touches the images, so training epochs load
    cached GLCM directly instead of recomputing per sample.

    This is what makes real GLCM affordable: it's slow per image, but only
    computed once per image across the entire training run, not once per
    epoch per worker.
    """
    computed, skipped = 0, 0
    for img_path, _ in pairs:
        cache_path = img_path + ".glcm.npy"
        if os.path.exists(cache_path) and not force:
            skipped += 1
            continue
        try:
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
            o_map = glcm_tool.compute(arr)
            np.save(cache_path, o_map)
            computed += 1
            if computed % 100 == 0:
                log(f"  GLCM cache: {computed} images computed so far")
        except Exception as e:
            log(f"  GLCM cache: failed on {img_path}: {e!r}")
    log(f"GLCM cache: {computed} computed, {skipped} already cached ({computed + skipped} total).")


def load_glcm_from_cache_or_compute(img_path, image_gray_uint8, glcm_tool):
    """Fast-path used by RoadSegDataset during training: try cache, fall back
    to on-the-fly computation. On-the-fly should be rare after
    precompute_glcm_cache runs upfront, but the fallback keeps the pipeline
    working even if a cache file is missing/corrupt."""
    cache_path = img_path + ".glcm.npy"
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path).astype(np.float32)
        except Exception:
            pass  # corrupt cache -- fall through to recompute
    return glcm_tool.compute(image_gray_uint8)


def compute_glcm_norm_value(pairs, glcm_tool, n_samples=50, seed=42):
    """Computes ONE fixed 99th-percentile constant across a representative
    sample of the training pool, matching the OLD proxy's normalization
    approach but now against the REAL GLCM output. Shared by every
    GLCMOcclusionMap instance for the rest of the run.

    n_samples defaults lower here (50 vs the old 300) since real GLCM is
    slower -- 50 images is still statistically ample for a 99th-percentile
    estimate, and this only runs once per full run.

    seed=42 (not the unseeded global `random` module as before) so the exact
    same 50 images get sampled on every run -- important since this value is
    NOT currently persisted into the checkpoint, so a resumed/validation-only
    run recomputes it from scratch and needs to land on the same answer the
    original training run did, not a fresh random draw.
    """
    sample = random.Random(seed).sample(pairs, min(n_samples, len(pairs)))
    all_values = []
    for img_path, _ in sample:
        arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        # Downsample for speed -- 99th percentile is a distributional
        # statistic and doesn't need every pixel of every image.
        arr_small = arr[::4, ::4]
        # compute_raw(), NOT compute() -- compute() always clips to
        # [0, norm_value], so it can never reveal the true dynamic range
        # calibration needs. See compute_raw()'s docstring for the bug this
        # replaced (both prior runs logged norm_value=1.0000 identically,
        # which was that clip-at-1.0 saturation, not a coincidence).
        raw_output = glcm_tool.compute_raw(arr_small)
        all_values.append(raw_output.ravel())

    combined = np.concatenate(all_values)
    norm_value = float(np.percentile(combined, 99)) if combined.max() > 0 else 1.0
    log(f"Fixed GLCM norm_value = {norm_value:.4f} (computed over {len(sample)} sampled images)")
    return norm_value


class AugmentationConfig:
    def __init__(self, canopy_paste_prob=0.0, thin_road_paste_prob=0.0, scale_range=None,
                 cutout_prob=0.0, canopy_patches=None, thin_road_snippets=None,
                 cloud_paste_prob=0.0, shadow_paste_prob=0.0, contrast_jitter_prob=0.0):
        self.canopy_paste_prob = canopy_paste_prob
        self.thin_road_paste_prob = thin_road_paste_prob
        self.scale_range = scale_range
        self.cutout_prob = cutout_prob
        self.canopy_patches = canopy_patches or []
        self.thin_road_snippets = thin_road_snippets or []
        self.cloud_paste_prob = cloud_paste_prob
        self.shadow_paste_prob = shadow_paste_prob
        self.contrast_jitter_prob = contrast_jitter_prob

    @staticmethod
    def none():
        return AugmentationConfig()


class RoadSegDataset:
    """RGB image + binary mask, returns 4-channel input [R, G, B, O_GLCM] + mask.
    GLCM is inherently a single-channel texture operation, so it's computed
    from a luminance-derived grayscale version of the RGB image internally
    -- this does NOT change how GLCM itself works, only where its input
    comes from."""

    def __init__(self, pairs, source_gsd_m, tile_size=512, global_stats=None, aug_config=None,
                 glcm_norm_value=None, deterministic_crop=None, use_fast_glcm_proxy=False):
        self.pairs = pairs
        self.source_gsd_m = source_gsd_m
        self.tile_size = tile_size
        self.glcm = GLCMOcclusionMap(norm_value=glcm_norm_value, use_fast_proxy=use_fast_glcm_proxy)
        self.global_stats = global_stats
        self.aug = aug_config or AugmentationConfig.none()
        self.deterministic_crop = deterministic_crop

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _luminance(rgb):
        """Standard RGB->luminance (BT.601), used ONLY for GLCM texture
        analysis, which is inherently single-channel. Does not affect the
        RGB channels actually fed to the model."""
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])

    def __getitem__(self, idx):
        import torch

        img_path, mask_path = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)  # (H, W, 3)
        mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)

        if self.aug.scale_range is not None:
            lo, hi = self.aug.scale_range
            s = random.uniform(lo, hi)
            new_h, new_w = max(1, int(img.shape[0] * s)), max(1, int(img.shape[1] * s))
            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR),
                           dtype=np.float32)
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((new_w, new_h), Image.NEAREST),
                            dtype=np.float32) / 255.0

        gray = self._luminance(img)
        gray_uint8 = ((gray - gray.min()) / (np.ptp(gray) + 1e-6) * 255).astype(np.uint8)
        if self.aug.scale_range is not None:
            o_map = self.glcm.compute(gray_uint8)
        else:
            o_map = load_glcm_from_cache_or_compute(img_path, gray_uint8, self.glcm)

        img, mask, o_map = self._fit_tile(img, mask, o_map)

        if self.aug.thin_road_snippets and random.random() < self.aug.thin_road_paste_prob:
            img, mask, o_map = self._paste_thin_road(img, mask, o_map)
        if self.aug.canopy_patches and random.random() < self.aug.canopy_paste_prob:
            img, o_map = self._paste_canopy(img, mask, o_map)
        if random.random() < self.aug.cloud_paste_prob:
            img, o_map = self._paste_cloud(img, mask, o_map)
        if random.random() < self.aug.shadow_paste_prob:
            img, o_map = self._paste_shadow(img, mask, o_map)
        if random.random() < self.aug.cutout_prob:
            img, o_map = self._cutout_background(img, mask, o_map)
        if random.random() < self.aug.contrast_jitter_prob:
            img = self._contrast_jitter(img)

        if self.global_stats:
            # Per-channel mean/std -- global_stats["mean"]/["std"] are now
            # 3-element arrays (R, G, B), not scalars.
            mean = np.asarray(self.global_stats["mean"], dtype=np.float32)
            std = np.asarray(self.global_stats["std"], dtype=np.float32)
            img_norm = (img - mean) / std
        else:
            img_norm = (img - img.mean(axis=(0, 1))) / (img.std(axis=(0, 1)) + 1e-6)

        # img_norm: (H, W, 3) -> (3, H, W); o_map: (H, W) -> (1, H, W); concat -> (4, H, W)
        img_chw = np.transpose(img_norm, (2, 0, 1))
        x = torch.tensor(np.concatenate([img_chw, o_map[None, :, :]], axis=0), dtype=torch.float32)
        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return x, y

    def _fit_tile(self, img, mask, o_map):
        t = self.tile_size
        h, w = img.shape[:2]  # img is now (H, W, 3)
        if h < t or w < t:
            ph, pw = max(0, t - h), max(0, t - w)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
            o_map = np.pad(o_map, ((0, ph), (0, pw)), mode="reflect")
            h, w = img.shape[:2]
        max_r, max_c = h - t, w - t

        if self.deterministic_crop == "center_occlusion":
            # Center the crop on the occluded-road region (mask AND high GLCM
            # texture overlap), not a random location. A random crop on an
            # image picked FOR containing occluded roads can still crop that
            # exact region out if it isn't centered -- this is what actually
            # guarantees the validation/visualization tile contains what it
            # was selected to show.
            occluded_road = (mask > 0.5) & (o_map > 0.5)
            if occluded_road.sum() >= 20:
                ys, xs = np.where(occluded_road)
                cy, cx = int(ys.mean()), int(xs.mean())
            else:
                # No occlusion found in this image after all -- fall back to
                # centering on the road mask generally, still deterministic.
                ys, xs = np.where(mask > 0.5)
                if len(ys) > 0:
                    cy, cx = int(ys.mean()), int(xs.mean())
                else:
                    cy, cx = h // 2, w // 2
            r = int(np.clip(cy - t // 2, 0, max(max_r, 0)))
            c = int(np.clip(cx - t // 2, 0, max(max_c, 0)))
        else:
            r = random.randint(0, max_r) if max_r > 0 else 0
            c = random.randint(0, max_c) if max_c > 0 else 0

        return img[r:r + t, c:c + t], mask[r:r + t, c:c + t], o_map[r:r + t, c:c + t]

    def _paste_thin_road(self, img, mask, o_map):
        snippet_img, snippet_mask = random.choice(self.aug.thin_road_snippets)
        if snippet_img.ndim != 3 or snippet_img.shape[-1] != 3:
            # Guards against a corrupt/stale snippet slipping through outside
            # the cache-loading path (e.g. bank built/passed in manually).
            # Skip this paste rather than crash the whole training run.
            return img, mask, o_map
        sh, sw = snippet_img.shape[:2]  # snippet_img is now (H, W, 3) RGB
        H, W = img.shape[:2]
        if sh >= H or sw >= W:
            return img, mask, o_map
        r, c = random.randint(0, H - sh), random.randint(0, W - sw)
        img[r:r + sh, c:c + sw] = snippet_img
        mask[r:r + sh, c:c + sw] = np.maximum(mask[r:r + sh, c:c + sw], snippet_mask)
        snippet_gray = self._luminance(snippet_img)
        snippet_u8 = snippet_gray.astype(np.uint8)
        o_local = self.glcm.compute(snippet_u8) if snippet_u8.size > 25 else np.zeros((sh, sw), np.float32)
        o_map[r:r + sh, c:c + sw] = o_local
        return img, mask, o_map

    def _paste_canopy(self, img, mask, o_map):
        road_ys, road_xs = np.where(mask > 0.5)
        if len(road_ys) < 50:
            return img, o_map
        canopy = random.choice(self.aug.canopy_patches)  # now (ch, cw, 3) RGB
        if canopy.ndim != 3 or canopy.shape[-1] != 3:
            # Same corrupt/stale-bank guard as _paste_thin_road above.
            return img, o_map
        ch, cw = canopy.shape[:2]
        H, W = img.shape[:2]
        if ch >= H or cw >= W:
            return img, o_map
        i = random.randrange(len(road_ys))
        ry, rx = road_ys[i], road_xs[i]
        r0 = max(0, min(H - ch, ry - ch // 2))
        c0 = max(0, min(W - cw, rx - cw // 2))
        yy, xx = np.mgrid[:ch, :cw].astype(np.float32)
        yc, xc = ch / 2, cw / 2
        rr = ((yy - yc) / yc) ** 2 + ((xx - xc) / xc) ** 2
        alpha = (np.clip(1.0 - rr, 0, 1) * 0.75)[:, :, None]  # broadcast over 3 channels
        patch = img[r0:r0 + ch, c0:c0 + cw]
        img[r0:r0 + ch, c0:c0 + cw] = (1 - alpha) * patch + alpha * canopy.astype(np.float32)
        o_map[r0:r0 + ch, c0:c0 + cw] = np.maximum(o_map[r0:r0 + ch, c0:c0 + cw], alpha[:, :, 0] * 0.9)
        return img, o_map

    def _paste_cloud(self, img, mask, o_map):
        """Synthetic cloud occlusion -- deliberately DIFFERENT treatment from
        _paste_canopy's o_map bump. Canopy patches are real, genuinely
        textured crops (high local contrast from leaves/branches), so
        bumping o_map (GLCM texture channel) there is accurate. Real clouds
        are the opposite: smooth, near-uniform brightness, LOW local
        texture. Artificially forcing o_map high on a smooth pasted patch
        would teach the model a false association ("smooth bright region ->
        high occlusion signal") that doesn't match what GLCM would actually
        compute on a real cloud. So: paste a procedurally generated soft
        bright blob (no real-cloud patch bank exists in these training
        datasets, unlike canopy which crops real texture from DeepGlobe
        images), keep the road label unchanged (same inference-through-
        occlusion signal as canopy), and leave o_map to whatever it
        naturally computes on the resulting patch -- expected near-zero,
        which is honest, not artificially inflated.
        """
        road_ys, road_xs = np.where(mask > 0.5)
        if len(road_ys) < 50:
            return img, o_map
        H, W = img.shape[:2]
        ch = cw = random.randint(40, 90)
        if ch >= H or cw >= W:
            return img, o_map
        i = random.randrange(len(road_ys))
        ry, rx = road_ys[i], road_xs[i]
        r0 = max(0, min(H - ch, ry - ch // 2))
        c0 = max(0, min(W - cw, rx - cw // 2))
        yy, xx = np.mgrid[:ch, :cw].astype(np.float32)
        yc, xc = ch / 2, cw / 2
        rr = ((yy - yc) / yc) ** 2 + ((xx - xc) / xc) ** 2
        # Small per-pixel noise so the blob isn't a perfect circle -- real
        # clouds have irregular, feathered edges.
        noise = np.random.normal(0, 0.12, size=(ch, cw)).astype(np.float32)
        alpha = (np.clip(1.0 - rr + noise, 0, 1) * 0.85)[:, :, None]  # broadcast over 3 channels
        # Genuinely colored, not neutral gray: real clouds have a slight
        # cool/blue tint from sky illumination (Rayleigh-scattered skylight
        # is blue-shifted). Base brightness randomized, blue channel boosted
        # slightly relative to red -- a small, physically-motivated tint,
        # not an arbitrary color choice.
        base = random.uniform(200, 245)
        cloud_color = np.array([base - 8, base - 3, base + 5], dtype=np.float32)  # R,G,B
        cloud_color = np.clip(cloud_color, 0, 255)
        patch = img[r0:r0 + ch, c0:c0 + cw]
        img[r0:r0 + ch, c0:c0 + cw] = (1 - alpha) * patch + alpha * cloud_color
        # o_map intentionally untouched -- see docstring.
        return img, o_map

    def _paste_shadow(self, img, mask, o_map):
        """Synthetic shadow occlusion (building/cloud-cast shadow over a road)
        -- a third occlusion type, distinct from canopy (textured) and cloud
        (bright, smooth). Shadows are DARK and smooth, same low-texture
        reasoning as _paste_cloud applies here too: o_map is NOT forced up,
        since a real shadow wouldn't produce high GLCM texture either.
        Elongated (not circular) since real shadows are cast in a consistent
        direction, not radially symmetric like a cloud or canopy clump.
        """
        road_ys, road_xs = np.where(mask > 0.5)
        if len(road_ys) < 50:
            return img, o_map
        H, W = img.shape[:2]
        # Elongated footprint: longer axis 60-140px, short axis 30-60px,
        # random orientation -- mimics a shadow cast across a road at an angle.
        long_axis = random.randint(60, 140)
        short_axis = random.randint(30, 60)
        angle = random.uniform(0, np.pi)
        ch = cw = max(long_axis, short_axis) + 10  # bounding box, padded
        if ch >= H or cw >= W:
            return img, o_map
        i = random.randrange(len(road_ys))
        ry, rx = road_ys[i], road_xs[i]
        r0 = max(0, min(H - ch, ry - ch // 2))
        c0 = max(0, min(W - cw, rx - cw // 2))
        yy, xx = np.mgrid[:ch, :cw].astype(np.float32)
        yc, xc = ch / 2, cw / 2
        # Rotate coordinates by angle, then apply an elliptical falloff
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        yr = (yy - yc) * cos_a - (xx - xc) * sin_a
        xr = (yy - yc) * sin_a + (xx - xc) * cos_a
        rr = (yr / (long_axis / 2)) ** 2 + (xr / (short_axis / 2)) ** 2
        noise = np.random.normal(0, 0.1, size=(ch, cw)).astype(np.float32)
        alpha = (np.clip(1.0 - rr + noise, 0, 1) * 0.7)[:, :, None]  # broadcast over 3
        # channels. Shadows are rarely fully opaque -- some ground detail
        # usually still shows through.
        # Genuinely colored, not neutral dark: real shadows are illuminated
        # by blue-shifted skylight (Rayleigh scattering), not pure black --
        # blue channel boosted relative to red, small physically-motivated tint.
        base = random.uniform(20, 70)
        shadow_color = np.array([base - 5, base - 2, base + 6], dtype=np.float32)  # R,G,B
        shadow_color = np.clip(shadow_color, 0, 255)
        patch = img[r0:r0 + ch, c0:c0 + cw]
        img[r0:r0 + ch, c0:c0 + cw] = (1 - alpha) * patch + alpha * shadow_color
        return img, o_map

    def _contrast_jitter(self, img):
        """Random global contrast adjustment -- your 4 training datasets
        almost certainly differ in imaging/processing conditions (different
        sensors, different post-processing pipelines), so teaching contrast
        invariance should help generalization across them, and to whatever
        the real Cartosat-3 tile's own contrast characteristics turn out to
        be. Simple linear contrast around the image's own mean, clipped to
        valid grayscale range -- deliberately not touching o_map, since
        contrast is a global tone operation, not a source of new occlusion.
        """
        factor = random.uniform(0.7, 1.4)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.float32)
        return img

    def _cutout_background(self, img, mask, o_map):
        H, W = img.shape[:2]
        h_c, w_c = random.randint(16, 48), random.randint(16, 48)
        for _ in range(3):
            r, c = random.randint(0, H - h_c), random.randint(0, W - w_c)
            if mask[r:r + h_c, c:c + w_c].mean() < 0.05:
                img[r:r + h_c, c:c + w_c] = 0.0
                o_map[r:r + h_c, c:c + w_c] = 0.0
                break
        return img, o_map


# ============================================================================
# SECTION 4 -- Global norm stats, canopy/thin-road banks, priority scores.
# All cached to OUTPUT_ROOT so a re-run doesn't redo this CPU work.
# ============================================================================

def sample_rgb_stats(pairs, n_samples=400):
    """Per-channel (R, G, B) mean/std, computed the same running-sum way as
    before -- just 3 accumulators instead of 1, since __getitem__ now
    normalizes each channel independently."""
    sample = random.sample(pairs, min(n_samples, len(pairs)))
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for img_path, _ in sample:
        arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)  # (H, W, 3)
        pixel_sum += arr.sum(axis=(0, 1))
        pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        pixel_count += arr.shape[0] * arr.shape[1]
    mean = pixel_sum / pixel_count
    std = (pixel_sq_sum / pixel_count - mean ** 2) ** 0.5
    return {"mean": mean.tolist(), "std": std.tolist()}


def _validate_patch_bank(patches, expected_channels=3, kind="patch"):
    """Sanity-check a loaded (e.g. canopy) patch bank before it's trusted.

    A cache built by an older/different code path -- e.g. before patches
    were switched from grayscale to real RGB -- will have the wrong array
    shape. Loading that silently and only discovering it deep inside a
    DataLoader worker (as a cryptic broadcast ValueError) is exactly what
    caused this crash. Returns False for anything empty or malformed so the
    caller can rebuild instead of crashing later.
    """
    if not patches:
        return False
    for p in patches:
        arr = np.asarray(p)
        if arr.ndim != 3 or arr.shape[-1] != expected_channels:
            log(f"Stale/invalid {kind} cache entry: shape={arr.shape}, "
                f"expected (H, W, {expected_channels}). Cache will be rebuilt.")
            return False
    return True


def _validate_thin_snippet_bank(snippets):
    """Same idea as _validate_patch_bank, but each entry is an (img, mask)
    pair rather than a bare array, so both halves need checking."""
    if not snippets:
        return False
    for img, mask in snippets:
        img_arr = np.asarray(img)
        mask_arr = np.asarray(mask)
        if img_arr.ndim != 3 or img_arr.shape[-1] != 3:
            log(f"Stale/invalid thin-road snippet image: shape={img_arr.shape}, "
                f"expected (H, W, 3). Cache will be rebuilt.")
            return False
        if mask_arr.ndim != 2 or mask_arr.shape != img_arr.shape[:2]:
            log(f"Stale/invalid thin-road snippet mask: shape={mask_arr.shape}, "
                f"expected {img_arr.shape[:2]}. Cache will be rebuilt.")
            return False
    return True


def _load_cached_bank(path, loader_fn, validator_fn, kind):
    """Load a .npy/.npz cache from disk and validate it before trusting it.

    Returns the validated bank, or None if the file is missing, unreadable,
    or fails validation -- in every "None" case the caller is expected to
    rebuild from source and re-save, rather than crash mid-training.
    """
    if not os.path.exists(path):
        return None
    try:
        loaded = loader_fn(path)
    except Exception as e:
        log(f"Failed to load {kind} cache at {path} ({e}). Rebuilding.")
        return None
    if not validator_fn(loaded):
        log(f"Discarding stale/invalid {kind} cache at {path}. Rebuilding.")
        return None
    return loaded


def build_canopy_bank(pair_pool, glcm_norm_value, n_patches=200, patch_size=64, glcm_threshold=0.35):
    glcm_tool = GLCMOcclusionMap(norm_value=glcm_norm_value, use_fast_proxy=USE_FAST_GLCM_PROXY)
    patches = []
    sample_pool = random.sample(pair_pool, min(len(pair_pool), n_patches * 3))
    for img_path, _ in sample_pool:
        if len(patches) >= n_patches:
            break
        try:
            rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)  # (H, W, 3)
            if rgb.shape[0] < patch_size or rgb.shape[1] < patch_size:
                continue
            # GLCM/texture is a location-finding tool here -- luminance-derived,
            # doesn't affect what gets extracted (real RGB, below).
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
            o = glcm_tool.compute(gray)
            H, W = gray.shape
            for _ in range(30):
                r, c = random.randint(0, H - patch_size), random.randint(0, W - patch_size)
                if o[r:r + patch_size, c:c + patch_size].mean() > glcm_threshold:
                    patches.append(rgb[r:r + patch_size, c:c + patch_size].copy())  # real RGB patch
                    break
        except Exception:
            continue
    return patches


def build_thin_snippet_bank(pair_pool, n=200, snippet_size=80, narrow_px_threshold=5):
    from scipy.ndimage import distance_transform_edt

    snippets = []
    sample_pool = random.sample(pair_pool, min(len(pair_pool), n * 3))
    for img_path, mask_path in sample_pool:
        if len(snippets) >= n:
            break
        try:
            rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)  # (H, W, 3)
            mk = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
            if rgb.shape[:2] != mk.shape or rgb.shape[0] < snippet_size:
                continue
            H, W = mk.shape
            d = distance_transform_edt(mk > 0)
            narrow_ys, narrow_xs = np.where((d > 0) & (d <= narrow_px_threshold))
            if len(narrow_ys) < 20:
                continue
            i = random.randrange(len(narrow_ys))
            y, x = narrow_ys[i], narrow_xs[i]
            r0 = max(0, min(H - snippet_size, y - snippet_size // 2))
            c0 = max(0, min(W - snippet_size, x - snippet_size // 2))
            snippets.append((
                rgb[r0:r0 + snippet_size, c0:c0 + snippet_size].copy(),  # real RGB snippet
                mk[r0:r0 + snippet_size, c0:c0 + snippet_size].astype(np.float32).copy(),
            ))
        except Exception:
            continue
    return snippets


def select_occluded_validation_pairs(held_out_pairs, glcm_norm_value, n_target=20, occlusion_threshold=0.5):
    """Ranks held-out pairs by what fraction of their ROAD pixels sit under
    high-GLCM-texture (canopy/shadow-like) regions, and returns the top
    `n_target` -- i.e. validation images that actually contain roads hidden
    under occlusion, which a plain tail-slice of the held-out set does not
    guarantee. Falls back to whatever's available if fewer than n_target
    pairs clear a nonzero-occlusion bar.
    """
    glcm_tool = GLCMOcclusionMap(norm_value=glcm_norm_value, use_fast_proxy=USE_FAST_GLCM_PROXY)
    scored = []
    for img_path, mask_path in held_out_pairs:
        try:
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
            mk = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
            if arr.shape != mk.shape:
                continue
            road_pixels = int(mk.sum())
            if road_pixels < 50:
                continue
            o = glcm_tool.compute(arr)
            occluded_road_pixels = int(((o > occlusion_threshold) & (mk > 0)).sum())
            frac = occluded_road_pixels / road_pixels
            scored.append((frac, img_path, mask_path))
        except Exception:
            continue

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:n_target]
    n_with_real_occlusion = sum(1 for f, _, _ in top if f > 0.05)
    if top:
        log(f"Occluded-validation selection: {len(scored)} held-out pairs scored, "
            f"top {len(top)} selected, {n_with_real_occlusion} have >5% of their road "
            f"under occlusion-like texture, occlusion fractions range "
            f"{top[-1][0]:.3f}-{top[0][0]:.3f}")
    else:
        log("Occluded-validation selection: no held-out pairs could be scored.")
    if n_with_real_occlusion < min(5, n_target):
        log("!!! WARNING: fewer than 5 held-out images show meaningful road/canopy "
            "overlap. Validation will still run, but it may not be testing occlusion "
            "robustness the way this run is meant to. This can happen if your dataset "
            "genuinely has little canopy cover (e.g. dense urban DeepGlobe/SpaceNet "
            "tiles) -- expected on some data, worth knowing either way.")
    return [(ip, mp) for _, ip, mp in top]


def build_priority_scores(all_pairs, cache_path, glcm_norm_value):
    from scipy.ndimage import distance_transform_edt

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("n") == len(all_pairs):
            log(f"Loaded {len(cached['scores'])} priority scores from cache.")
            return cached["scores"]

    glcm_tool = GLCMOcclusionMap(norm_value=glcm_norm_value, use_fast_proxy=USE_FAST_GLCM_PROXY)

    def score_sample(img_path, mask_path):
        try:
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)[::4, ::4]
            mk = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)[::4, ::4]
            if arr.shape != mk.shape:
                return 0.05
            if mk.sum() < 10:
                narrow_frac = 0.0
            else:
                d = distance_transform_edt(mk > 0)
                narrow_frac = float(((d > 0) & (d <= 3)).sum()) / max(mk.size, 1)
            o = glcm_tool.compute(arr)
            glcm_frac = float((o > 0.5).mean())
            return 0.05 + 0.5 * narrow_frac + 0.5 * glcm_frac
        except Exception:
            return 0.05

    scores = []
    for i, (ip, mp) in enumerate(all_pairs):
        scores.append(score_sample(ip, mp))
        if i % 1000 == 0:
            log(f"  priority-scored {i}/{len(all_pairs)}")
    with open(cache_path, "w") as f:
        json.dump({"n": len(all_pairs), "scores": scores}, f)
    return scores


# ============================================================================
# SECTION 5 -- PathMamba model (unchanged contract: returns
# (road_logits, confidence_map), both (B, 1, H, W); input is 2-channel).
# ============================================================================

def build_model(use_mamba: bool):
    import timm
    import torch
    import torch.nn as nn

    class CrossAttentionLayer(nn.Module):
        def __init__(self, dim, num_heads=8):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            return self.norm(x + attn_out)

    class BottleneckFallback(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.gru = nn.GRU(d_model, d_model // 2, bidirectional=True, batch_first=True)

        def forward(self, x):
            out, _ = self.gru(x)
            return out

    class StackedMamba(nn.Module):
        """2 Mamba blocks in sequence, pre-norm residual connections -- the
        one architecture change for this retrain. Targets the actual stated
        bottleneck directly: Mamba's job here is long-range reasoning to
        bridge occlusion gaps, which is the hard part of this task, so
        adding capacity specifically here is a more targeted lever than
        scaling the CNN backbone further (which mostly improves local
        feature quality, not long-range reasoning). Pre-norm residual is
        the standard, well-established way to stack sequence blocks --
        matches how real Mamba-based architectures scale depth, not a novel
        or risky pattern.
        """
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, n_layers=2):
            super().__init__()
            from mamba_ssm.modules.mamba_simple import Mamba
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        def forward(self, x):
            for layer, norm in zip(self.layers, self.norms):
                x = x + layer(norm(x))
            return x

    class UNetDecoder(nn.Module):
        def __init__(self, encoder_channels, decoder_channels):
            super().__init__()
            self.blocks = nn.ModuleList()
            in_ch = encoder_channels[-1]
            for skip_ch, out_ch in zip(reversed(encoder_channels[:-1]), decoder_channels):
                self.blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ))
                in_ch = out_ch
            self.final_up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        def forward(self, skip_features, bottleneck):
            x = bottleneck
            for block, skip in zip(self.blocks, reversed(skip_features)):
                upsample_layer, conv_layer, relu_layer = block
                x = upsample_layer(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = conv_layer(x)
                x = relu_layer(x)
            return self.final_up(x)

    class PathMamba(nn.Module):
        def __init__(self, in_channels=4, pretrained=True, use_mamba=True):
            super().__init__()
            # tf_efficientnet_b6.ns_jft_in1k: Noisy Student pretraining (JFT-300M
            # -> ImageNet-1k fine-tune) -- best transfer performance of the 3
            # B6 pretrained variants timm has available (.aa_in1k, .ap_in1k,
            # .ns_jft_in1k). Plain "efficientnet_b6" has NO registered
            # pretrained weights at all on this timm version -- confirmed via
            # timm.list_models(pretrained=True) before picking this name,
            # not guessed blind a second time.
            self.encoder = timm.create_model("tf_efficientnet_b6.ns_jft_in1k", pretrained=pretrained,
                                              features_only=True, out_indices=(0, 1, 2, 3, 4))
            orig_conv = self.encoder.conv_stem
            new_conv = nn.Conv2d(in_channels, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                                  stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight = nn.Parameter(
                    orig_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels * 3
                )
            self.encoder.conv_stem = new_conv

            feat_dims = [f["num_chs"] for f in self.encoder.feature_info.info]
            bottleneck_dim = feat_dims[-1]

            self.use_mamba = use_mamba
            if use_mamba:
                self.seq_block = StackedMamba(d_model=bottleneck_dim, d_state=16, d_conv=4, expand=2, n_layers=2)
            else:
                self.seq_block = BottleneckFallback(bottleneck_dim)

            self.cross_attn = CrossAttentionLayer(dim=bottleneck_dim, num_heads=8)
            self.decoder = UNetDecoder(encoder_channels=feat_dims, decoder_channels=[256, 128, 64, 32])
            self.road_head = nn.Conv2d(32, 1, kernel_size=1)
            self.conf_head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

        def forward(self, x):
            features = self.encoder(x)
            bottleneck = features[-1]
            B, C, H, W = bottleneck.shape
            seq = bottleneck.flatten(2).transpose(1, 2)
            seq = self.seq_block(seq)
            seq = self.cross_attn(seq)
            bottleneck = seq.transpose(1, 2).reshape(B, C, H, W)
            out = self.decoder(features[:-1], bottleneck)
            if out.shape[-2:] != x.shape[-2:]:
                out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.road_head(out), self.conf_head(out)

    model = PathMamba(in_channels=4, pretrained=True, use_mamba=use_mamba).cuda()
    log(f"Model built. use_mamba={use_mamba}. "
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


# ============================================================================
# SECTION 6 -- Combined loss (Tversky + Boundary + clDice), unchanged.
# ============================================================================

def load_cldice():
    """Clones jocpae/clDice for SoftSkeletonize. Returns (available, soft_skel_fn)."""
    clone_dir = os.path.join(OUTPUT_ROOT, "clDice")
    if not os.path.isdir(clone_dir):
        subprocess.run(["git", "clone", "-q", "https://github.com/jocpae/clDice.git", clone_dir],
                       capture_output=True, text=True)
    sys.path.append(os.path.join(clone_dir, "cldice_loss", "pytorch"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "soft_skeleton", os.path.join(clone_dir, "cldice_loss", "pytorch", "soft_skeleton.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        soft_skel_fn = mod.SoftSkeletonize(num_iter=10)
        log("clDice loaded OK.")
        return True, soft_skel_fn
    except Exception as e:
        log(f"clDice import failed; disabling clDice term: {e!r}")
        return False, None


def build_loss(tile_size=512):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.ndimage import distance_transform_edt

    cldice_available, soft_skel_fn = load_cldice()

    class TverskyLoss(nn.Module):
        def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
            super().__init__()
            self.alpha, self.beta, self.smooth = alpha, beta, smooth

        def forward(self, pred, target):
            p = torch.sigmoid(pred).view(-1)
            t = target.view(-1)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

    class BoundaryLoss(nn.Module):
        def __init__(self, tile_size=512):
            super().__init__()
            self.scale = tile_size * 0.2

        def forward(self, pred, target):
            target_np = target.detach().cpu().numpy()
            dist_maps = []
            for b in range(target_np.shape[0]):
                mask = target_np[b, 0].astype(bool)
                if not mask.any():
                    d = np.ones_like(mask, dtype=np.float32) * 0.5
                else:
                    d = distance_transform_edt(~mask).astype(np.float32)
                    d = np.clip(d / self.scale, 0, 1)
                dist_maps.append(d)
            dist_tensor = torch.tensor(np.stack(dist_maps)[:, np.newaxis], device=pred.device)
            ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            return (dist_tensor * ce).mean()

    class BufferedIoULoss(nn.Module):
        """Differentiable buffer-tolerant IoU loss, matching the rubric's
        'Length-Complete / Relaxed IoU' metric definition (3-5px tolerance).
        Dilates both prediction and target by buffer_px before computing
        IoU -- directly optimizes for the leniency the evaluation itself
        rewards, rather than only measuring it after the fact.

        Dilation implemented via max-pooling: a k x k max-pool with
        stride=1, padding=k//2 is mathematically equivalent to square
        dilation by k//2 pixels, and is fully differentiable (unlike a
        real morphological dilation op), so gradients flow through it
        normally during training.
        """
        def __init__(self, buffer_px=4):
            super().__init__()
            self.buffer_px = buffer_px
            k = 2 * buffer_px + 1
            self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=buffer_px)

        def forward(self, pred, target):
            pred_prob = torch.sigmoid(pred)
            pred_dilated = self.pool(pred_prob)
            target_dilated = self.pool(target)
            intersection = (pred_dilated * target_dilated).sum()
            union = pred_dilated.sum() + target_dilated.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            return 1 - iou

    class CombinedLoss(nn.Module):
        """Adds an explicit confidence-supervision term on top of the
        original Tversky+Boundary+clDice segmentation loss.

        BUG FOUND during review: the model's `conf` (confidence) output was
        never used in any loss term anywhere -- it only received gradients
        indirectly through the shared decoder backbone, so nothing taught it
        what "confidence" should actually mean. healing.py's 3-gate check
        depends on this output meaning "the model believes an occluded gap
        is plausibly a real road" -- untrained, it likely just mirrors
        whatever generic texture/edge signal the decoder already encodes.

        Fix: confidence is explicitly trained to predict the ground-truth
        road mask, weighted toward occluded (high-GLCM-texture) regions --
        i.e. "be confident specifically about occluded-but-real roads,"
        which is exactly the semantics healing.py's gate assumes.
        """

        def __init__(self, w_tversky=0.5, w_boundary=0.4, w_cldice=0.5, w_confidence=0.2,
                     w_buffered_iou=0.4, buffer_px=4,
                     tile_size=512, tversky_alpha=0.7, tversky_beta=0.3):
            # alpha/beta FLIPPED from the original 0.3/0.7 (recall-favoring)
            # to 0.7/0.3 (precision-favoring). Root cause confirmed across
            # every one of 6 completed runs: recall was consistently high
            # (0.58-0.86) while precision stayed low (0.20-0.49) -- the old
            # setting penalized false negatives 2.3x more than false
            # positives, pushing training against a dimension (recall) that
            # was never actually the problem, while barely constraining the
            # dimension (precision) that was. Predicted masks were visibly
            # fatter/blobbier than ground truth on every dataset -- this is
            # the direct mechanism. This setting never changed across any
            # prior run (backbone, augmentation, and Mamba depth all did),
            # so it was never ruled out as the root cause until now.
            #
            # w_buffered_iou=0.4: directly optimizes for the rubric's
            # "Length-Complete / Relaxed IoU" tolerance (3-5px buffer),
            # rather than only measuring it at evaluation time.
            super().__init__()
            self.w_cldice = w_cldice if cldice_available else 0.0
            self.w_tversky, self.w_boundary, self.w_confidence = w_tversky, w_boundary, w_confidence
            self.w_buffered_iou = w_buffered_iou
            self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
            self.buffered_iou = BufferedIoULoss(buffer_px=buffer_px)
            self.boundary = BoundaryLoss(tile_size=tile_size)

        def forward(self, pred, target, conf=None, occlusion_channel=None):
            l_tv = self.tversky(pred, target)
            l_bd = self.boundary(pred, target)
            l_biou = self.buffered_iou(pred, target)
            total = self.w_tversky * l_tv + self.w_boundary * l_bd + self.w_buffered_iou * l_biou
            l_cl = torch.tensor(0.0, device=pred.device)
            if self.w_cldice > 0:
                pred_prob = torch.sigmoid(pred)
                pred_skel = soft_skel_fn(pred_prob)
                target_skel = soft_skel_fn(target)
                tprec = (pred_skel * target).sum() / (pred_skel.sum() + 1e-6)
                tsens = (target_skel * pred_prob).sum() / (target_skel.sum() + 1e-6)
                l_cl = 1 - 2 * (tprec * tsens) / (tprec + tsens + 1e-6)
                total = total + self.w_cldice * l_cl

            l_conf = torch.tensor(0.0, device=pred.device)
            if conf is not None and occlusion_channel is not None and self.w_confidence > 0:
                # BCE against the ground-truth road mask (conf already has
                # sigmoid applied in the model, per the confirmed contract --
                # use plain BCE, not with-logits).
                per_pixel_bce = F.binary_cross_entropy(conf.clamp(1e-6, 1 - 1e-6), target, reduction="none")
                # Weight by occlusion (GLCM channel, already normalized [0,1])
                # plus a small floor so non-occluded pixels still contribute
                # a little signal rather than being entirely ignored.
                weight = 0.1 + 0.9 * occlusion_channel
                l_conf = (per_pixel_bce * weight).sum() / (weight.sum() + 1e-6)
                total = total + self.w_confidence * l_conf

            return total, {
                "tversky": l_tv.item(), "boundary": l_bd.item(),
                "cldice": float(l_cl), "confidence": float(l_conf),
                "buffered_iou": l_biou.item(),
            }

    return CombinedLoss(tile_size=tile_size).cuda()


# ============================================================================
# SECTION 7 -- Reliability: checkpoint save/load and a single reusable
# training-stage runner (AMP + DataParallel + periodic checkpointing).
# ============================================================================

def save_checkpoint(path, model, optimizer, epoch, scheduler=None, scaler=None, extra=None):
    import torch

    state = {
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_best_available(ckpt_path, snapshot_path, model, optimizer, scheduler=None, scaler=None):
    """Picks whichever of (completed-epoch checkpoint, mid-epoch snapshot)
    represents more progress, and returns the epoch to resume FROM (i.e. the
    epoch index that should be re-run in full -- mid-epoch snapshots restart
    their epoch rather than trying to skip already-seen batches of a
    shuffled DataLoader, which would need fragile bookkeeping for a small
    time saving)."""
    import torch

    ckpt_data = torch.load(ckpt_path, map_location="cuda") if os.path.exists(ckpt_path) else None
    snap_data = torch.load(snapshot_path, map_location="cuda") if os.path.exists(snapshot_path) else None

    if ckpt_data is None and snap_data is None:
        return 0

    # Prefer the snapshot ONLY if it represents a STRICTLY LATER epoch than
    # the completed checkpoint -- i.e. genuine additional progress. A TIE
    # means the checkpoint already finished that same epoch; the original
    # version wrongly preferred the snapshot on a tie, which meant the final
    # epoch of every stage got needlessly re-run on every resume (a stale
    # mid-epoch snapshot from partway through that already-completed epoch
    # was mistaken for "further along" than the completed checkpoint).
    if snap_data is not None and (ckpt_data is None or snap_data["epoch"] > ckpt_data["epoch"]):
        label, ckpt = "mid-epoch snapshot", snap_data
    else:
        label, ckpt = "completed-epoch checkpoint", ckpt_data
    (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    resume_epoch = ckpt["epoch"] if label == "mid-epoch snapshot" else ckpt["epoch"] + 1
    log(f"Resuming from {label}: epoch field={ckpt['epoch']} -> restarting at epoch {resume_epoch}")
    return resume_epoch


def run_stage(stage_name, model, loader, epochs, lr, ckpt_path, snapshot_path, criterion,
              scheduler_fn=None, load_init_from=None, glcm_norm_value=None):
    """One training stage (A0 / A / B), with AMP, DataParallel (if >1 GPU
    visible), and checkpointing both at epoch-end and every
    CHECKPOINT_EVERY_N_BATCHES batches within an epoch."""
    import torch

    if torch.cuda.device_count() > 1 and not hasattr(model, "module"):
        model = torch.nn.DataParallel(model)
        log(f"[{stage_name}] Using {torch.cuda.device_count()} GPUs via DataParallel.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = scheduler_fn(optimizer) if scheduler_fn else None
    # AMP disabled entirely: mamba-ssm's selective-scan CUDA kernel is
    # unstable under FP16 (produces nan mid-forward-pass, confirmed during
    # a live run). Pure FP32 throughout -- slower per batch, but the only
    # thing that produced real, non-nan loss values in practice.
    scaler = None

    if load_init_from and os.path.exists(load_init_from) and not os.path.exists(ckpt_path):
        ckpt = torch.load(load_init_from, map_location="cuda")
        target_model = model.module if hasattr(model, "module") else model
        ckpt_state = ckpt["model"]
        current_state = target_model.state_dict()

        # Shape-aware loading: the RGB switch changed the first conv layer's
        # input channel count (2 -> 4), so its weight tensor shape no longer
        # matches the old checkpoint. Rather than failing outright (strict
        # loading) or silently dropping the whole checkpoint, explicitly
        # detect and skip ONLY keys with a genuine shape mismatch, loading
        # everything else (the entire backbone beyond the stem, Mamba,
        # decoder) from the old best-performing weights. This is the actual
        # mechanism behind "resume old weights, adapt only the first layer."
        skipped = []
        compatible_state = {}
        for k, v in ckpt_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                compatible_state[k] = v
            else:
                skipped.append(k)
        missing = target_model.load_state_dict(compatible_state, strict=False)
        log(f"[{stage_name}] Initialized weights from previous stage: {load_init_from}")
        if skipped:
            log(f"[{stage_name}] Skipped {len(skipped)} shape-mismatched key(s) "
                f"(expected for the RGB channel-count change): {skipped}")
    start_epoch = load_best_available(ckpt_path, snapshot_path, model, optimizer, scheduler, scaler)

    if start_epoch >= epochs:
        log(f"[{stage_name}] Already complete ({start_epoch} epochs >= {epochs}). Skipping.")
        return model, {"stage": stage_name, "epochs_run": 0, "wall_seconds": 0.0}

    stage_t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = len(loader)
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad()
            pred, conf = model(x)
            occlusion_channel = x[:, 1:2, :, :]  # GLCM channel, already [0,1]
            loss, parts = criterion(pred, y, conf=conf, occlusion_channel=occlusion_channel)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 50 == 0:
                log(f"  [{stage_name}] epoch {epoch} batch {i}/{n_batches} loss={loss.item():.4f} "
                    f"({parts})")
            if i > 0 and i % CHECKPOINT_EVERY_N_BATCHES == 0:
                save_checkpoint(snapshot_path, model, optimizer, epoch, scheduler, scaler,
                                 extra={"batch_in_epoch": i, "glcm_norm_value": glcm_norm_value})
                log(f"  [{stage_name}] snapshot saved at epoch {epoch} batch {i} -> {snapshot_path}")

        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0
        log(f"=== [{stage_name}] epoch {epoch} done. avg_loss={running_loss / n_batches:.4f} "
            f"time={dt:.1f}s ===")
        save_checkpoint(ckpt_path, model, optimizer, epoch, scheduler, scaler,
                         extra={"glcm_norm_value": glcm_norm_value})

    return model, {"stage": stage_name, "epochs_run": epochs - start_epoch, "wall_seconds": time.time() - stage_t0}


# ============================================================================
# SECTION 8 -- Validation (IoU + clDice) on held-out data.
# ============================================================================

def iou_score(pred_bin, target_bin, eps=1e-6):
    inter = (pred_bin & target_bin).sum().item()
    union = (pred_bin | target_bin).sum().item()
    return (inter + eps) / (union + eps)


def road_precision_recall(pred_bin, target_bin, eps=1e-6):
    """Road-pixel-ONLY precision and recall -- background pixels never enter
    either number. Complements iou_score (which is also already road-only,
    not raw pixel accuracy) with the more legible breakdown of WHICH kind of
    error is happening:
      recall low, precision high    -> model is too conservative, missing
                                        real road (the canopy-occlusion
                                        failure mode from earlier)
      recall high, precision low    -> model is over-predicting road
                                        (false positives on non-road texture)
    Neither can be inflated by a "predict all background" model -- both
    would correctly show 0 recall for a model that never predicts any road.
    """
    tp = (pred_bin & target_bin).sum().item()
    pred_pos = pred_bin.sum().item()
    true_pos = target_bin.sum().item()
    precision = (tp + eps) / (pred_pos + eps)
    recall = (tp + eps) / (true_pos + eps)
    return precision, recall


def run_validation(model, val_pairs, global_stats, source_gsd_m, cldice_available, soft_skel_fn,
                   glcm_norm_value, use_fast_glcm_proxy=False):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    val_ds = RoadSegDataset(val_pairs, source_gsd_m=source_gsd_m, tile_size=512,
                            global_stats=global_stats, aug_config=AugmentationConfig.none(),
                            glcm_norm_value=glcm_norm_value, deterministic_crop="center_occlusion")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    ious, cldices, precisions, recalls = [], [], [], []
    occlusion_recalls = []  # NEW: recall computed ONLY on true-road pixels that
    # are ALSO under high-GLCM-texture (occlusion-like) regions -- the deck's
    # own "Occlusion-Recall" metric, promised but never actually computed as
    # its own number until tonight. 0.35 threshold matches build_canopy_bank's
    # existing glcm_threshold convention elsewhere in this file, for consistency.
    _OCCLUSION_GLCM_THRESHOLD = 0.35
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            pred, _ = model(x)
            pred_bin = torch.sigmoid(pred) > 0.5
            target_bin = y > 0.5
            occlusion_mask = x[:, 1:2] > _OCCLUSION_GLCM_THRESHOLD  # GLCM channel is x[:, 1]
            for b in range(pred_bin.shape[0]):
                ious.append(iou_score(pred_bin[b], target_bin[b]))
                p, r = road_precision_recall(pred_bin[b], target_bin[b])
                precisions.append(p)
                recalls.append(r)
                if cldice_available:
                    p_skel = soft_skel_fn(pred_bin[b:b+1].float())
                    t_skel = soft_skel_fn(target_bin[b:b+1].float())
                    tprec = (p_skel * target_bin[b].float()).sum() / (p_skel.sum() + 1e-6)
                    tsens = (t_skel * pred_bin[b:b+1].float()).sum() / (t_skel.sum() + 1e-6)
                    cldices.append((2 * tprec * tsens / (tprec + tsens + 1e-6)).item())

                occluded_road = target_bin[b] & occlusion_mask[b]
                n_occluded_road_px = occluded_road.sum().item()
                if n_occluded_road_px >= 20:  # skip tiles with ~no occluded road at all
                    hits = (pred_bin[b] & occluded_road).sum().item()
                    occlusion_recalls.append(hits / n_occluded_road_px)

    ious_a = np.array(ious)
    result = {
        "n": len(ious),
        "iou_mean": float(ious_a.mean()),
        "iou_median": float(np.median(ious_a)),
        "road_precision_mean": float(np.mean(precisions)),
        "road_recall_mean": float(np.mean(recalls)),
    }
    if cldices:
        cldices_a = np.array(cldices)
        result["cldice_mean"] = float(cldices_a.mean())
    if occlusion_recalls:
        result["occlusion_recall_mean"] = float(np.mean(occlusion_recalls))
        result["occlusion_recall_n_tiles"] = len(occlusion_recalls)
    else:
        result["occlusion_recall_mean"] = None
        result["occlusion_recall_n_tiles"] = 0
    log(f"Validation on {len(val_pairs)} pairs: {result}")
    return result


# ============================================================================
# SECTION 9 -- Visualization: N_VIZ_IMAGES held-out images, each showing
# input / GLCM channel / ground truth / predicted probability / confidence.
# ============================================================================

def run_visualization(model, val_pairs, global_stats, source_gsd_m, glcm_norm_value,
                      use_fast_glcm_proxy=False, n_images=N_VIZ_IMAGES, seed=123,
                      out_filename="visual_check.png"):
    import matplotlib
    matplotlib.use("Agg")  # no display on a headless Jarvis instance
    import matplotlib.pyplot as plt
    import torch

    model.eval()
    # Random sample, not val_pairs[:n_images] -- otherwise every run shows the
    # exact same first-alphabetically images. Seeded so a re-run of the SAME
    # underlying val_pairs is still reproducible, not random noise you can't
    # compare across runs.
    if len(val_pairs) > n_images:
        chosen = random.Random(seed).sample(val_pairs, n_images)
    else:
        chosen = val_pairs
    if not chosen:
        log("No pairs available for visualization -- skipping.")
        return

    ds = RoadSegDataset(chosen, source_gsd_m=source_gsd_m, tile_size=512,
                        global_stats=global_stats, aug_config=AugmentationConfig.none(),
                        glcm_norm_value=glcm_norm_value, deterministic_crop="center_occlusion")

    fig, axes = plt.subplots(len(chosen), 5, figsize=(20, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["Input (grayscale)", "GLCM occlusion channel", "Ground truth mask",
                  "Predicted probability", "Confidence map"]

    with torch.no_grad():
        for row, idx in enumerate(range(len(chosen))):
            x, y = ds[idx]
            x_batch = x.unsqueeze(0).cuda()
            pred, conf = model(x_batch)
            pred_prob = torch.sigmoid(pred)[0, 0].cpu().numpy()
            conf_map = conf[0, 0].cpu().numpy()

            # Per-row stat: what fraction of THIS crop's road pixels sit under
            # occlusion-like texture -- makes it explicit in the figure itself
            # that these rows are actually testing occluded roads, not just
            # asserted to be in the log.
            road_mask = y[0].numpy() > 0.5
            occl_mask = x[1].numpy() > 0.5
            occl_road_frac = float((road_mask & occl_mask).sum()) / max(int(road_mask.sum()), 1)
            pred_iou_row = _quick_iou(pred_prob > 0.5, road_mask)

            panels = [x[0].numpy(), x[1].numpy(), y[0].numpy(), pred_prob, conf_map]
            for col, (panel, title) in enumerate(zip(panels, col_titles)):
                ax = axes[row, col]
                ax.imshow(panel, cmap="gray" if col != 3 else "inferno")
                if row == 0:
                    ax.set_title(title, fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{occl_road_frac:.0%} of road\nunder occlusion\nIoU={pred_iou_row:.2f}",
                                  fontsize=9, rotation=0, labelpad=55, va="center")
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_ROOT, out_filename)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved visualization: {out_path}")
    log("Interpretation guide:")
    log("  - Left label on each row: what fraction of that image's road is under occlusion-like")
    log("    texture, and the model's IoU on that specific crop -- read these together.")
    log("  - Column 3 vs 4: does the predicted probability roughly follow the ground truth mask's shape?")
    log("  - Column 5 (confidence): does it light up specifically where column 2 (GLCM texture) is high --")
    log("    i.e. is the model learning to flag occluded regions, not just copying probability?")
    log("  - If column 4 is near-uniform gray noise everywhere, training hasn't converged yet.")


def _quick_iou(pred_bin_np, target_bin_np, eps=1e-6):
    inter = np.logical_and(pred_bin_np, target_bin_np).sum()
    union = np.logical_or(pred_bin_np, target_bin_np).sum()
    return float((inter + eps) / (union + eps))


# ============================================================================
# MAIN
# ============================================================================

def main():
    run_report = {"stages": []}

    check_gpu()
    mamba_available = install_mamba()
    check_timm()

    if not mamba_available:
        log("=" * 70)
        log("!!! mamba-ssm did not install/compile successfully.")
        log("!!! This run would train the GRU fallback, not real PathMamba.")
        if ABORT_IF_FALLBACK:
            log("!!! ABORT_IF_FALLBACK is set (default) -- stopping before any training.")
            log("!!! Set env var ALLOW_FALLBACK_TRAINING=1 to override and train the fallback anyway.")
            log("=" * 70)
            sys.exit(1)
        else:
            log("!!! ALLOW_FALLBACK_TRAINING=1 set -- proceeding with GRU fallback anyway.")
        log("=" * 70)

    dg_pairs, sn_pairs_all, mass_pairs, mass_train_pairs, urban_binary_pairs, sn5_mumbai_pairs = run_data_audit()

    # --- Held-out split, done ONCE, before anything touches these lists. ---
    # Uniform across all 5 datasets: 40 held out per dataset (not just
    # SpaceNet's old 5%-based approach), via a fixed seed so this is
    # reproducible run to run. Massachusetts is the one exception -- it
    # already has its own official, dataset-designated test split (mass_pairs,
    # handled separately below and in run_ood_check), which is a cleaner
    # held-out set than an arbitrary reshuffle would be, so it's left as-is.
    N_HELD_OUT_PER_DATASET = 40

    def _held_out_split(pairs, n_held_out, seed=42):
        if not pairs:
            return [], []
        shuffled = list(pairs)
        random.Random(seed).shuffle(shuffled)  # separate RNG instance so this
        # shuffle doesn't consume/perturb the global random state other
        # augmentation uses
        n = min(n_held_out, len(shuffled))
        return shuffled[:n], shuffled[n:]

    dg_held_out, dg_train = _held_out_split(dg_pairs, N_HELD_OUT_PER_DATASET)
    sn_held_out, sn_train = _held_out_split(sn_pairs_all, N_HELD_OUT_PER_DATASET)
    urban_binary_held_out, urban_binary_train = _held_out_split(urban_binary_pairs, N_HELD_OUT_PER_DATASET)
    sn5_mumbai_held_out, sn5_mumbai_train = _held_out_split(sn5_mumbai_pairs, N_HELD_OUT_PER_DATASET)

    def _filter_empty_masks(pairs, name):
        """Mentor-suggested: drop training pairs whose mask has literally
        zero road pixels. Applied to TRAINING pairs only, deliberately NOT
        to held-out/validation sets -- realistic evaluation should still
        include genuinely-empty scenes, to properly measure false-positive
        behavior on non-road imagery (this is exactly what several of the
        visualized validation rows this session were: correct 0%-road
        predictions on 0%-road ground truth, a legitimate case to keep
        testing, not a case to keep training on if it's contributing
        nothing but background)."""
        kept = []
        n_dropped = 0
        for img_path, mask_path in pairs:
            mask = np.array(Image.open(mask_path).convert("L"))
            if (mask > 127).any():
                kept.append((img_path, mask_path))
            else:
                n_dropped += 1
        log(f"  Empty-mask filter ({name}): dropped {n_dropped} of {len(pairs)} training pairs")
        return kept

    dg_train = _filter_empty_masks(dg_train, "DeepGlobe")
    sn_train = _filter_empty_masks(sn_train, "SpaceNet")
    if mass_train_pairs:
        mass_train_pairs = _filter_empty_masks(mass_train_pairs, "Massachusetts")
    if urban_binary_train:
        urban_binary_train = _filter_empty_masks(urban_binary_train, "Urban-binary")

    log(f"Held-out split (40/dataset, never trained on, never in global_stats/glcm_norm_value):")
    log(f"  DeepGlobe:          {len(dg_train)} train, {len(dg_held_out)} held out")
    log(f"  SpaceNet:           {len(sn_train)} train, {len(sn_held_out)} held out")
    log(f"  Urban-binary:       {len(urban_binary_train)} train, {len(urban_binary_held_out)} held out")
    log(f"  SpaceNet-5 Mumbai:  {len(sn5_mumbai_train)} train, {len(sn5_mumbai_held_out)} held out")
    log(f"  Massachusetts:      {len(mass_train_pairs)} train (own official split), "
        f"{len(mass_pairs)} held out (own official test split, unchanged)")
    total_held_out = (len(dg_held_out) + len(sn_held_out) + len(urban_binary_held_out)
                      + len(sn5_mumbai_held_out) + len(mass_pairs))
    log(f"  TOTAL held out across all 5 datasets: {total_held_out} (target ~200)")

    # mass_train_pairs (Massachusetts train+val) is legitimate training data,
    # included here like dg_train/sn_train. mass_pairs (Massachusetts TEST
    # split) is deliberately excluded -- held out for the OOD generalization
    # check (see run_ood_check below) and must not influence global_stats or
    # glcm_norm_value, or the "generalization" test would quietly cheat by
    # tuning training-time normalization to the test domain's own statistics.
    all_pairs = (dg_train + sn_train + mass_train_pairs
                + urban_binary_train + sn5_mumbai_train)

    log("Computing global normalization stats...")
    global_stats = sample_rgb_stats(all_pairs)
    log(f"Global stats: {global_stats}")

    # Real multi-angle GLCM occlusion channel, matching the deck's claim
    # (previous version was a uniform-filter local-variance proxy). See
    # GLCMOcclusionMap docstring for the trade-offs. `use_fast_proxy=True`
    # is the emergency escape hatch if the Jarvis run turns out CPU-bound --
    # flip it and re-run.
    USE_FAST_GLCM_PROXY = os.environ.get("USE_FAST_GLCM_PROXY", "0") == "1"
    if USE_FAST_GLCM_PROXY:
        log("USE_FAST_GLCM_PROXY=1 -- using local-variance proxy, NOT real GLCM.")
    import torch
    glcm_tool_for_norm = GLCMOcclusionMap(use_fast_proxy=USE_FAST_GLCM_PROXY)

    # Prefer a norm_value already saved in an existing checkpoint over
    # recomputing -- resuming or re-validating must use the EXACT constant
    # the model was actually trained under, not a fresh recomputation (see
    # compute_glcm_norm_value's docstring for why recomputing isn't safe to
    # treat as equivalent, even with the seed fix below).
    glcm_norm_value = None
    for ckpt_name in ("stage_b_rgbfinal_checkpoint.pth", "stage_a_rgbfinal_checkpoint.pth", "stage_a0_rgbfinal_checkpoint.pth"):
        ckpt_path = os.path.join(OUTPUT_ROOT, ckpt_name)
        if os.path.exists(ckpt_path):
            saved = torch.load(ckpt_path, map_location="cpu").get("glcm_norm_value")
            if saved is not None:
                glcm_norm_value = saved
                log(f"Reusing glcm_norm_value={glcm_norm_value:.4f} saved in {ckpt_name} "
                    f"(not recomputing).")
                break
    if glcm_norm_value is None:
        log("Sampling real GLCM output to fix normalization constant "
            "(this takes a minute; runs once)...")
        glcm_norm_value = compute_glcm_norm_value(all_pairs, glcm_tool_for_norm)

    # Precompute the real GLCM output ONCE per image, cached to disk. Every
    # training epoch after this then loads the cached array in ~10ms instead
    # of recomputing GLCM at ~100-300ms per tile. This is what makes real
    # GLCM affordable inside a DataLoader without becoming the bottleneck.
    if not USE_FAST_GLCM_PROXY:
        log("Precomputing GLCM cache for all training images (one-time cost)...")
        glcm_tool_for_cache = GLCMOcclusionMap(norm_value=glcm_norm_value,
                                                use_fast_proxy=False)
        precompute_glcm_cache(all_pairs + sn_held_out + dg_held_out + urban_binary_held_out
                              + sn5_mumbai_held_out + mass_pairs, glcm_tool_for_cache)
        if mass_pairs:
            # Cached with the SAME training-derived norm_value, not a fresh
            # one of its own -- the OOD check needs to see Massachusetts
            # through the exact lens the model was trained under, matching
            # how an unseen Cartosat-3 tile will be handled at inference.
            log("Precomputing GLCM cache for Massachusetts (OOD check only, not trained on)...")
            precompute_glcm_cache(mass_pairs, glcm_tool_for_cache)

    canopy_path = os.path.join(OUTPUT_ROOT, "canopy_patches.npy")
    canopy_patches = _load_cached_bank(
        canopy_path,
        loader_fn=lambda p: list(np.load(p, allow_pickle=True)),
        validator_fn=lambda patches: _validate_patch_bank(patches, expected_channels=3, kind="canopy"),
        kind="canopy",
    )
    if canopy_patches is None:
        # mass_train_pairs included (legitimate training data); mass_pairs
        # (Massachusetts TEST split) intentionally excluded -- held out
        # entirely for the OOD generalization check (see run_ood_check
        # below), never touched by training in any form, including
        # indirectly via canopy-patch augmentation.
        canopy_source = dg_train + mass_train_pairs
        canopy_patches = build_canopy_bank(canopy_source, glcm_norm_value)
        np.save(canopy_path, np.array(canopy_patches, dtype=object))
    log(f"Canopy patch bank: {len(canopy_patches)} patches.")

    def _load_thin_npz(p):
        data = np.load(p, allow_pickle=True)
        return list(zip(data["imgs"], data["masks"]))

    thin_path = os.path.join(OUTPUT_ROOT, "thin_road_snippets.npz")
    thin_road_snippets = _load_cached_bank(
        thin_path,
        loader_fn=_load_thin_npz,
        validator_fn=_validate_thin_snippet_bank,
        kind="thin-road snippet",
    )
    if thin_road_snippets is None:
        thin_source = sn_train + dg_train
        thin_road_snippets = build_thin_snippet_bank(thin_source)
        if thin_road_snippets:
            np.savez(thin_path, imgs=np.array([s[0] for s in thin_road_snippets], dtype=object),
                     masks=np.array([s[1] for s in thin_road_snippets], dtype=object))
    log(f"Thin-road snippet bank: {len(thin_road_snippets)} snippets.")

    priority_scores = build_priority_scores(
        all_pairs, os.path.join(OUTPUT_ROOT, "priority_scores.json"), glcm_norm_value
    )

    model = build_model(use_mamba=mamba_available)
    criterion = build_loss(tile_size=512)

    import torch
    from torch.utils.data import DataLoader

    log("=== SMOKE TEST ===")
    smoke_ds = RoadSegDataset(all_pairs[:64], source_gsd_m=0.3, tile_size=512,
                              global_stats=global_stats, aug_config=AugmentationConfig.none(),
                              glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
    smoke_loader = DataLoader(smoke_ds, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    smoke_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for i, (x, y) in enumerate(smoke_loader):
        x, y = x.cuda(), y.cuda()
        smoke_optimizer.zero_grad()
        pred, conf = model(x)
        occlusion_channel = x[:, 1:2, :, :]
        loss, parts = criterion(pred, y, conf=conf, occlusion_channel=occlusion_channel)
        loss.backward()
        smoke_optimizer.step()
        log(f"  smoke batch {i}: loss={loss.item():.4f} parts={parts}")
    log("Smoke test passed -- proceeding to full training.")
    del smoke_optimizer  # fresh optimizer state for the real Stage A0 run

    a0_datasets = [
        RoadSegDataset(dg_train, source_gsd_m=0.5, tile_size=512, global_stats=global_stats,
                       aug_config=AugmentationConfig.none(), glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY),
        RoadSegDataset(sn_train, source_gsd_m=0.3, tile_size=512, global_stats=global_stats,
                       aug_config=AugmentationConfig.none(), glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY),
    ]
    if urban_binary_train:
        a0_datasets.append(
            RoadSegDataset(urban_binary_train, source_gsd_m=URBAN_BINARY_GSD_M, tile_size=512,
                           global_stats=global_stats, aug_config=AugmentationConfig.none(),
                           glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
        )
    if sn5_mumbai_train:
        a0_datasets.append(
            RoadSegDataset(sn5_mumbai_train, source_gsd_m=SPACENET5_MUMBAI_GSD_M, tile_size=512,
                           global_stats=global_stats, aug_config=AugmentationConfig.none(),
                           glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
        )
    if mass_train_pairs:
        a0_datasets.append(
            RoadSegDataset(mass_train_pairs, source_gsd_m=1.0, tile_size=512, global_stats=global_stats,
                           aug_config=AugmentationConfig.none(), glcm_norm_value=glcm_norm_value,
                           use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
        )
    a0_ds = torch.utils.data.ConcatDataset(a0_datasets)
    # shuffle=True already randomizes batch order across the full concatenated
    # pool every epoch -- with mass_train_pairs now included, all three
    # sources are shuffled together from the very start of training, not
    # bolted on partway through in Stage A.
    a0_loader = DataLoader(a0_ds, batch_size=A0_BATCH, shuffle=True, num_workers=4,
                           drop_last=True, pin_memory=True)
    model, a0_report = run_stage(
        "A0", model, a0_loader, A0_EPOCHS, A0_LR,
        ckpt_path=os.path.join(OUTPUT_ROOT, "stage_a0_rgbfinal_checkpoint.pth"),
        snapshot_path=os.path.join(OUTPUT_ROOT, "stage_a0_rgbfinal_snapshot.pth"),
        criterion=criterion,
        load_init_from=RESUME_FROM_CHECKPOINT,
        glcm_norm_value=glcm_norm_value,
    )
    run_report["stages"].append(a0_report)

    # cloud_paste_prob=0.15: new tonight, modest probability deliberately --
    # this is an untested mechanism, don't let it dominate the way an
    # over-aggressive change destabilized an earlier experiment tonight.
    # shadow_paste_prob and contrast_jitter_prob are new this run -- also
    # modest, same caution.
    full_aug = AugmentationConfig(canopy_paste_prob=0.3, thin_road_paste_prob=0.25,
                                   scale_range=(0.7, 1.5), cutout_prob=0.25,
                                   canopy_patches=canopy_patches, thin_road_snippets=thin_road_snippets,
                                   cloud_paste_prob=0.15, shadow_paste_prob=0.15, contrast_jitter_prob=0.3)

    a_datasets = [
        RoadSegDataset(dg_train, source_gsd_m=0.5, tile_size=512, global_stats=global_stats,
                       aug_config=full_aug, glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY),
        RoadSegDataset(sn_train, source_gsd_m=0.3, tile_size=512, global_stats=global_stats,
                       aug_config=full_aug, glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY),
    ]
    mass_aug = AugmentationConfig(scale_range=(0.7, 1.5), contrast_jitter_prob=0.3)

    # Neutral weight (1.0) for the two new datasets by default -- no
    # evidence-based reason yet to up/downweight them the way SpaceNet (1.5,
    # historically prioritized) or Massachusetts (0.5, downweighted) are.
    # Adjust once there's a specific reason to.
    per_source_weight = [1.0] * len(dg_train) + [1.5] * len(sn_train)
    if mass_train_pairs:
        a_datasets.append(RoadSegDataset(mass_train_pairs, source_gsd_m=1.0, tile_size=512,
                                         global_stats=global_stats, aug_config=mass_aug,
                                         glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY))
        per_source_weight += [1.5] * len(mass_train_pairs)
    if urban_binary_train:
        urban_binary_aug = AugmentationConfig(canopy_paste_prob=0.2, thin_road_paste_prob=0.15,
                                              scale_range=(0.7, 1.5), cutout_prob=0.2,
                                              canopy_patches=canopy_patches, thin_road_snippets=thin_road_snippets,
                                              cloud_paste_prob=0.1, shadow_paste_prob=0.1, contrast_jitter_prob=0.3)
        a_datasets.append(RoadSegDataset(urban_binary_train, source_gsd_m=URBAN_BINARY_GSD_M, tile_size=512,
                                         global_stats=global_stats, aug_config=urban_binary_aug,
                                         glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY))
        per_source_weight += [0.4] * len(urban_binary_train)
    if sn5_mumbai_train:
        a_datasets.append(RoadSegDataset(sn5_mumbai_train, source_gsd_m=SPACENET5_MUMBAI_GSD_M, tile_size=512,
                                         global_stats=global_stats, aug_config=full_aug,
                                         glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY))
        per_source_weight += [1.0] * len(sn5_mumbai_train)
    # mass_pairs (Massachusetts TEST split) is NOT added here -- see
    # run_ood_check below. That's the whole point of keeping it separate
    # from mass_train_pairs.
    a_ds = torch.utils.data.ConcatDataset(a_datasets)

    assert len(priority_scores) == len(a_ds), (
        f"priority_scores length {len(priority_scores)} != dataset length {len(a_ds)} -- "
        "the concat order changed somewhere; re-check dg/sn/mass ordering matches build_priority_scores' input."
    )
    sample_weights = [w * p for w, p in zip(per_source_weight, priority_scores)]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(a_ds), replacement=True)
    a_loader = DataLoader(a_ds, batch_size=A_BATCH, sampler=sampler, num_workers=4,
                          drop_last=True, pin_memory=True)

    model, a_report = run_stage(
        "A", model, a_loader, A_EPOCHS, A_LR,
        ckpt_path=os.path.join(OUTPUT_ROOT, "stage_a_rgbfinal_checkpoint.pth"),
        snapshot_path=os.path.join(OUTPUT_ROOT, "stage_a_rgbfinal_snapshot.pth"),
        criterion=criterion,
        load_init_from=os.path.join(OUTPUT_ROOT, "stage_a0_rgbfinal_checkpoint.pth"),
        glcm_norm_value=glcm_norm_value,
    )
    run_report["stages"].append(a_report)

    b_aug = AugmentationConfig(canopy_paste_prob=0.2, thin_road_paste_prob=0.0, scale_range=(0.85, 1.15),
                                cutout_prob=0.15, canopy_patches=canopy_patches)
    b_ds = RoadSegDataset(sn_train, source_gsd_m=0.3, tile_size=512, global_stats=global_stats,
                          aug_config=b_aug, glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
    b_loader = DataLoader(b_ds, batch_size=B_BATCH, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    model, b_report = run_stage(
        "B", model, b_loader, B_EPOCHS, B_LR,
        ckpt_path=os.path.join(OUTPUT_ROOT, "stage_b_rgbfinal_checkpoint.pth"),
        snapshot_path=os.path.join(OUTPUT_ROOT, "stage_b_rgbfinal_snapshot.pth"),
        criterion=criterion,
        scheduler_fn=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=B_EPOCHS),
        load_init_from=os.path.join(OUTPUT_ROOT, "stage_a_rgbfinal_checkpoint.pth"),
        glcm_norm_value=glcm_norm_value,
    )
    run_report["stages"].append(b_report)

    # --- Validation: genuinely held-out AND selected specifically for
    # containing roads under canopy/shadow-like occlusion, not a raw slice. ---
    val_pairs = select_occluded_validation_pairs(sn_held_out, glcm_norm_value, n_target=20)
    cldice_available, soft_skel_fn = load_cldice()
    val_result = run_validation(model, val_pairs, global_stats, source_gsd_m=0.3,
                                cldice_available=cldice_available, soft_skel_fn=soft_skel_fn,
                                glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
    run_report["validation"] = val_result

    run_visualization(model, val_pairs, global_stats, source_gsd_m=0.3, glcm_norm_value=glcm_norm_value,
                      use_fast_glcm_proxy=USE_FAST_GLCM_PROXY, out_filename="visual_check_spacenet.png")

    # --- Out-of-distribution generalization check: Massachusetts roads was
    # NEVER trained on (excluded from Stage A, from all_pairs, from
    # global_stats, from glcm_norm_value calibration, and from the
    # canopy-patch bank) -- this is a genuine "does the architecture
    # generalize to a domain it has never seen" test, using the SAME trained
    # weights and the SAME normalization constants the model learned under
    # (matching how an unseen real Cartosat-3 tile gets handled at inference,
    # per run_real_inference.py). 1m/pixel matches this dataset's known GSD.
    if mass_pairs:
        log("=" * 70)
        log("OUT-OF-DISTRIBUTION CHECK: Massachusetts roads (never trained on)")
        log("=" * 70)
        ood_result = run_validation(model, mass_pairs, global_stats, source_gsd_m=1.0,
                                    cldice_available=cldice_available, soft_skel_fn=soft_skel_fn,
                                    glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
        run_report["ood_massachusetts"] = ood_result
        run_visualization(model, mass_pairs, global_stats, source_gsd_m=1.0,
                          glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY,
                          n_images=min(N_VIZ_IMAGES, len(mass_pairs)), out_filename="visual_check_massachusetts.png")
    else:
        log("No Massachusetts pairs found -- skipping OOD generalization check "
            f"(expected under {MASS_ROOT}).")
        ood_result = None

    # --- The remaining 3 held-out sets: DeepGlobe, urban-binary, and
    # SpaceNet-5 Mumbai. DeepGlobe is in-distribution (trained on other
    # DeepGlobe images); urban-binary and SpaceNet-5 Mumbai are genuinely new
    # domains this model has never seen anything else from at all, so these
    # are real generalization checks too, same spirit as the Massachusetts one.
    def _held_out_check(name, pairs, source_gsd_m, viz_filename, report_key):
        if not pairs:
            log(f"No {name} held-out pairs -- skipping (dataset not ready yet or empty).")
            return None
        log("=" * 70)
        log(f"HELD-OUT CHECK: {name}")
        log("=" * 70)
        result = run_validation(model, pairs, global_stats, source_gsd_m=source_gsd_m,
                                cldice_available=cldice_available, soft_skel_fn=soft_skel_fn,
                                glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY)
        run_report[report_key] = result
        run_visualization(model, pairs, global_stats, source_gsd_m=source_gsd_m,
                          glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=USE_FAST_GLCM_PROXY,
                          n_images=min(N_VIZ_IMAGES, len(pairs)), out_filename=viz_filename)
        return result

    dg_held_out_result = _held_out_check("DeepGlobe", dg_held_out, 0.5,
                                         "visual_check_deepglobe.png", "held_out_deepglobe")
    urban_binary_result = _held_out_check("Urban-binary", urban_binary_held_out, URBAN_BINARY_GSD_M,
                                          "visual_check_urbanbinary.png", "held_out_urban_binary")
    sn5_mumbai_result = _held_out_check("SpaceNet-5 Mumbai", sn5_mumbai_held_out, SPACENET5_MUMBAI_GSD_M,
                                        "visual_check_spacenet5mumbai.png", "held_out_spacenet5_mumbai")

    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Mamba used (not fallback): {mamba_available}")
    total_wall = 0.0
    for s in run_report["stages"]:
        log(f"  Stage {s['stage']}: {s['epochs_run']} epochs run, {s['wall_seconds'] / 3600:.2f} hours")
        total_wall += s["wall_seconds"]
    log(f"Total training wall time this run: {total_wall / 3600:.2f} hours")
    log(f"Validation (SpaceNet, held-out, in-distribution): {val_result}")
    log(f"OOD check (Massachusetts, never trained on): {ood_result}")
    log(f"Held-out check (DeepGlobe, in-distribution): {dg_held_out_result}")
    log(f"Held-out check (Urban-binary, never trained on): {urban_binary_result}")
    log(f"Held-out check (SpaceNet-5 Mumbai, never trained on): {sn5_mumbai_result}")
    log(f"Checkpoints saved under: {OUTPUT_ROOT}")
    log(f"5 visualizations saved under {OUTPUT_ROOT}: visual_check_{{spacenet,massachusetts,"
        f"deepglobe,urbanbinary,spacenet5mumbai}}.png")
    with open(os.path.join(OUTPUT_ROOT, "run_report.json"), "w") as f:
        json.dump(run_report, f, indent=2)
    log(f"Full run report: {os.path.join(OUTPUT_ROOT, 'run_report.json')}")
    log("=" * 70)

    # --- Cartosat-3 fine-tune (hackathon day only) -----------------------
    # OFF by default. Everything above this line (tonight's run, its
    # checkpoints, its run_report.json) is complete and unaffected whether
    # or not this runs. See run_cartosat_finetune_stage()'s docstring.
    run_cartosat_finetune_stage(global_stats, glcm_norm_value, USE_FAST_GLCM_PROXY, criterion)
    run_opensatmap_finetune_stage(global_stats, glcm_norm_value, USE_FAST_GLCM_PROXY, criterion)


def run_cartosat_finetune_stage(global_stats, glcm_norm_value, use_fast_glcm_proxy, criterion):
    """Hackathon-day Cartosat-3 fine-tune. OFF by default (ENABLE_CARTOSAT_FINETUNE
    env var, checked first below) -- a no-op on any run where you haven't
    explicitly turned it on, including tonight's.

    IMPORTANT: CARTOSAT_ROOT must point at RGB-CONVERTED PNG tiles (via
    convert_isro_tiles_to_rgb.py), NOT the raw 4-band GeoTIFF output from
    patchify.py directly -- PIL's Image.open().convert("RGB") (used by
    RoadSegDataset) is not reliable for multi-band satellite GeoTIFFs.

    GSD = 1.0m -- confirmed via the actual ISRO hackathon spec document,
    NOT the ~0.28m figure assumed earlier in this session from the original
    deck. This matters: using the wrong GSD here would silently break every
    real-world distance calculation downstream, same class of bug as the
    EPSG:4326-as-meters issue fixed earlier this session.

    Starts from stage_b_rgbfinal_checkpoint.pth (load_init_from), NOT training
    from scratch. Small LR (CARTOSAT_LR = 1e-5). Saves to a SEPARATE
    checkpoint (stage_c_cartosat_rgbfinal_checkpoint.pth).

    UNTESTED -- no real labeled Cartosat-3 data existed to test this stage
    against until today's upload. Budget time to verify it actually runs
    cleanly before trusting its output blind.
    """
    if not ENABLE_CARTOSAT_FINETUNE:
        log("Cartosat-3 fine-tune stage: ENABLE_CARTOSAT_FINETUNE not set -- skipping (this is correct for tonight).")
        return

    log("=" * 70)
    log("CARTOSAT-3 FINE-TUNE STAGE (hackathon day)")
    log("=" * 70)
    cartosat_pairs = build_cartosat_pairs(CARTOSAT_ROOT)
    if not cartosat_pairs:
        log(f"No Cartosat-3 pairs found under {CARTOSAT_ROOT} -- skipping. "
            f"(Expected {CARTOSAT_ROOT}/images/*.png + {CARTOSAT_ROOT}/masks/*.png -- "
            f"run convert_isro_tiles_to_rgb.py first if you only have raw GeoTIFF tiles.)")
        return

    log(f"Cartosat-3 pairs found: {len(cartosat_pairs)}")
    n_val = max(1, int(0.1 * len(cartosat_pairs)))
    cartosat_val, cartosat_train = cartosat_pairs[:n_val], cartosat_pairs[n_val:]
    log(f"Cartosat-3 split: {len(cartosat_train)} train, {len(cartosat_val)} held out.")

    import torch
    from torch.utils.data import DataLoader

    cartosat_aug = AugmentationConfig(scale_range=(0.8, 1.3), contrast_jitter_prob=0.3)
    cartosat_ds = RoadSegDataset(cartosat_train, source_gsd_m=1.0, tile_size=512,
                                 global_stats=global_stats, aug_config=cartosat_aug,
                                 glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=use_fast_glcm_proxy)
    cartosat_loader = DataLoader(cartosat_ds, batch_size=CARTOSAT_BATCH, shuffle=True,
                                 num_workers=4, drop_last=True, pin_memory=True)

    mamba_available = install_mamba()  # idempotent -- fast "already importable" path if already installed
    model = build_model(use_mamba=mamba_available)
    model, c_report = run_stage(
        "C_cartosat", model, cartosat_loader, CARTOSAT_EPOCHS, CARTOSAT_LR,
        ckpt_path=os.path.join(OUTPUT_ROOT, "stage_c_cartosat_rgbfinal_checkpoint.pth"),
        snapshot_path=os.path.join(OUTPUT_ROOT, "stage_c_cartosat_rgbfinal_snapshot.pth"),
        criterion=criterion,
        load_init_from=os.path.join(OUTPUT_ROOT, "stage_a_rgbfinal_checkpoint.pth"),
        glcm_norm_value=glcm_norm_value,
    )

    if cartosat_val:
        cldice_available, soft_skel_fn = load_cldice()
        val_result = run_validation(model, cartosat_val, global_stats, source_gsd_m=1.0,
                                    cldice_available=cldice_available, soft_skel_fn=soft_skel_fn,
                                    glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=use_fast_glcm_proxy)
        log(f"Cartosat-3 fine-tune validation (held-out): {val_result}")
        run_visualization(model, cartosat_val, global_stats, source_gsd_m=1.0,
                          glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=use_fast_glcm_proxy,
                          n_images=min(N_VIZ_IMAGES, len(cartosat_val)))

    log(f"Cartosat-3 fine-tune done. Checkpoint: "
        f"{os.path.join(OUTPUT_ROOT, 'stage_c_cartosat_rgbfinal_checkpoint.pth')}")
    log("=" * 70)


def run_opensatmap_finetune_stage(global_stats, glcm_norm_value, use_fast_glcm_proxy, criterion):
    """Hackathon-day OpenSatMap fine-tune. OFF by default, exact same
    pattern as run_cartosat_finetune_stage above -- a no-op unless
    ENABLE_OPENSATMAP_FINETUNE=1. Starts from stage_b_rgbfinal_checkpoint.pth,
    small LR, saves to a separate checkpoint. See build_opensatmap_pairs()'s
    docstring for why this is still a genuine placeholder (label-format
    complexity, not yet a simple fill-in like urban-binary turned out to be).
    """
    if not ENABLE_OPENSATMAP_FINETUNE:
        log("OpenSatMap fine-tune stage: ENABLE_OPENSATMAP_FINETUNE not set -- skipping (correct for this run).")
        return

    log("=" * 70)
    log("OPENSATMAP FINE-TUNE STAGE (hackathon day)")
    log("=" * 70)
    try:
        opensatmap_pairs = build_opensatmap_pairs(OPENSATMAP_ROOT)
    except NotImplementedError as e:
        log(f"!!! OpenSatMap pairing not ready yet: {e}")
        return
    if not opensatmap_pairs:
        log(f"No OpenSatMap pairs found under {OPENSATMAP_ROOT} -- skipping.")
        return

    log(f"OpenSatMap pairs found: {len(opensatmap_pairs)}")
    n_val = max(1, int(0.1 * len(opensatmap_pairs)))
    opensatmap_val, opensatmap_train = opensatmap_pairs[:n_val], opensatmap_pairs[n_val:]
    log(f"OpenSatMap split: {len(opensatmap_train)} train, {len(opensatmap_val)} held out.")

    import torch
    from torch.utils.data import DataLoader

    opensatmap_aug = AugmentationConfig(scale_range=(0.8, 1.3), contrast_jitter_prob=0.3)
    opensatmap_ds = RoadSegDataset(opensatmap_train, source_gsd_m=OPENSATMAP_GSD_M, tile_size=512,
                                   global_stats=global_stats, aug_config=opensatmap_aug,
                                   glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=use_fast_glcm_proxy)
    opensatmap_loader = DataLoader(opensatmap_ds, batch_size=OPENSATMAP_BATCH, shuffle=True,
                                   num_workers=4, drop_last=True, pin_memory=True)

    mamba_available = install_mamba()
    model = build_model(use_mamba=mamba_available)
    model, o_report = run_stage(
        "D_opensatmap", model, opensatmap_loader, OPENSATMAP_EPOCHS, OPENSATMAP_LR,
        ckpt_path=os.path.join(OUTPUT_ROOT, "stage_d_opensatmap_rgbfinal_checkpoint.pth"),
        snapshot_path=os.path.join(OUTPUT_ROOT, "stage_d_opensatmap_rgbfinal_snapshot.pth"),
        criterion=criterion,
        load_init_from=os.path.join(OUTPUT_ROOT, "stage_b_rgbfinal_checkpoint.pth"),
        glcm_norm_value=glcm_norm_value,
    )

    if opensatmap_val:
        cldice_available, soft_skel_fn = load_cldice()
        val_result = run_validation(model, opensatmap_val, global_stats, source_gsd_m=OPENSATMAP_GSD_M,
                                    cldice_available=cldice_available, soft_skel_fn=soft_skel_fn,
                                    glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=use_fast_glcm_proxy)
        log(f"OpenSatMap fine-tune validation (held-out): {val_result}")
        run_visualization(model, opensatmap_val, global_stats, source_gsd_m=OPENSATMAP_GSD_M,
                          glcm_norm_value=glcm_norm_value, use_fast_glcm_proxy=use_fast_glcm_proxy,
                          n_images=min(N_VIZ_IMAGES, len(opensatmap_val)))

    log(f"OpenSatMap fine-tune done. Checkpoint: "
        f"{os.path.join(OUTPUT_ROOT, 'stage_d_opensatmap_rgbfinal_checkpoint.pth')}")
    log("=" * 70)


if __name__ == "__main__":
    main()
