"""
Simplified Noiseprint-style image tampering detection PoC.

Noise residual backend (selected automatically at runtime):
  1. Official Noiseprint CNN (preferred):
       Requires Noiseprint.py + utilityRead.py + pretrained_weights/ in the
       working directory, cloned from:
         https://github.com/RonyAbecidan/noiseprint-pytorch
       Install extra dependency:  pip install torch
       The correct weight file is chosen per-image from the JPEG quality
       factor (QF 51-100); PNG images use qf101 weights.
  2. Gaussian high-pass fallback (automatic if CNN files are absent):
       residual = image - gaussian_lowpass(image)
       No camera-discriminative learning; provided for convenience only.

Directory layout expected:
    data/src/<CameraModel>_<OriginalName>.jpg   -- reference images (fingerprint source)
    data/dst/<CameraModel>_<OriginalName>.jpg   -- images to be inspected for tampering
    output/                                      -- results written here

Filename convention follows the VISION dataset:
    <CameraModelName>_<OriginalFileName>.jpg
    e.g.  iPhone_16_IMG001.jpg  ->  camera model = "iPhone_16"

Requirements:
    Python >= 3.10
    pip install numpy scipy Pillow matplotlib          # always required
    pip install torch                                  # required for CNN backend
    # clone model + weights into working directory:
    #   git clone https://github.com/RonyAbecidan/noiseprint-pytorch
    #   cp noiseprint-pytorch/Noiseprint.py .
    #   cp noiseprint-pytorch/utilityRead.py .
    #   cp -r noiseprint-pytorch/pretrained_weights .
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, zoom

try:
    from Noiseprint import getNoiseprint as _getNoiseprint  # type: ignore
    NOISEPRINT_CNN_AVAILABLE = True
except ImportError:
    NOISEPRINT_CNN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SRC_DIR    = Path("data/src")   # reference images for fingerprint extraction
DST_DIR    = Path("data/dst")   # images to inspect
OUTPUT_DIR = Path("output")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

GAUSSIAN_SIGMA  = 2.0   # high-pass sigma for noise residual extraction
NCC_PATCH_SIZE  = 48    # local NCC patch size (pixels)
NCC_STEP        = 16    # NCC sliding window step (pixels)
SIZE_OUTLIER_RATIO = 0.5  # reject images whose size differs from majority


# ---------------------------------------------------------------------------
# Image loading with EXIF transpose normalization
# ---------------------------------------------------------------------------

def load_gray(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load image as float32 grayscale, applying EXIF orientation first.
    Returns (array, (width, height)) after transpose.
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)   # normalize orientation
        img = img.convert("L")               # grayscale
        arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    return arr, (w, h)


def collect_image_paths(directory: Path) -> list[Path]:
    """Return all supported image files in directory (non-recursive)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Size-consistency filtering (per camera model)
# ---------------------------------------------------------------------------

def filter_by_size_per_camera(
    paths: list[Path], label: str
) -> tuple[list[Path], list[Path]]:
    """
    For each camera model (first underscore-separated token of the filename),
    determine the majority (width, height) among images of that camera and
    reject any image whose size differs from that majority.

    Images from different camera models are checked independently -- a size
    valid for camera A does not constrain camera B.

    Returns (accepted, rejected) path lists and prints warnings for rejected files.
    """
    # Group paths by camera model
    by_camera: dict[str, list[Path]] = defaultdict(list)
    unreadable: list[Path] = []

    for p in paths:
        by_camera[parse_camera_model(p.name)].append(p)

    accepted: list[Path] = []
    rejected: list[Path] = list(unreadable)

    for cam, cam_paths in by_camera.items():
        # Read sizes for all images of this camera
        size_of: dict[Path, tuple[int, int]] = {}
        for p in cam_paths:
            try:
                with Image.open(p) as img:
                    img = ImageOps.exif_transpose(img)
                    size_of[p] = img.size   # (width, height)
            except Exception as e:
                print(f"  [WARNING] [{label}] Cannot open {p.name} ({e}) -- skipped")
                rejected.append(p)

        if not size_of:
            continue

        # Determine majority size within this camera group
        size_counts: dict[tuple[int, int], int] = defaultdict(int)
        for s in size_of.values():
            size_counts[s] += 1
        maj = max(size_counts, key=lambda s: size_counts[s])

        for p, size in size_of.items():
            if size == maj:
                accepted.append(p)
            else:
                rejected.append(p)
                print(
                    f"  [WARNING] [{label}] {p.name} rejected: "
                    f"size {size} differs from majority size {maj} "
                    f"for camera '{cam}'"
                )

    return accepted, rejected


# ---------------------------------------------------------------------------
# VISION-style filename parsing
# ---------------------------------------------------------------------------

def parse_camera_model(filename: str) -> str:
    """
    Extract the camera model prefix from a VISION-style filename.
    Format: <CameraModelName>_<OriginalFileName>.ext
    Only the first underscore-separated token is used as the camera model.

    Examples:
        'Apple_iPhone6_IMG_0001.jpg'  -> 'Apple'
        'Canon_EOS_img001.jpg'        -> 'Canon'
        'SamsungGalaxyS7_photo.jpg'   -> 'SamsungGalaxyS7'
    """
    stem = Path(filename).stem          # strip extension
    parts = stem.split("_")
    return parts[0]


# ---------------------------------------------------------------------------
# Noise residual extraction
# ---------------------------------------------------------------------------

def extract_noise_residual(
    img_gray: np.ndarray,
    sigma: float = GAUSSIAN_SIGMA,
    image_path: Path | None = None,
) -> np.ndarray:
    """
    Extract noise residual from an image using one of two backends:

    1. Official Noiseprint CNN (used when NOISEPRINT_CNN_AVAILABLE is True
       and image_path is provided):
         - Calls getNoiseprint(image_path) from RonyAbecidan/noiseprint-pytorch,
           which loads the pretrained DnCNN weight file matching the image's
           JPEG quality factor (model_qf{QF}.pth, QF 51-100; PNG -> qf101).
         - getNoiseprint() reads the file independently from img_gray, so
           img_gray is unused in this branch but kept for API consistency.
         - Returns the CNN residual as float32.
         - NOTE: getNoiseprint() performs its own internal image read and does
           NOT apply EXIF transpose. If source images have orientation metadata,
           pre-normalise them on disk before running this script.

    2. Gaussian high-pass fallback (used when CNN is unavailable or
       image_path is None):
         - residual = image - gaussian_lowpass(image, sigma)
         - Captures general high-frequency noise but is NOT camera-
           discriminative in the deep-learning sense.
    """
    if NOISEPRINT_CNN_AVAILABLE and image_path is not None:
        try:
            _, residual = _getNoiseprint(str(image_path))
            return residual.astype(np.float32)
        except Exception as e:
            print(
                f"  [WARNING] Noiseprint CNN failed for "
                f"{Path(image_path).name} ({e}) -- falling back to Gaussian residual"
            )
    # Gaussian high-pass fallback
    smooth = gaussian_filter(img_gray, sigma=sigma)
    return img_gray - smooth


# ---------------------------------------------------------------------------
# Per-camera fingerprint aggregation
# ---------------------------------------------------------------------------

def build_camera_fingerprint(residuals: list[np.ndarray]) -> np.ndarray:
    """
    Average multiple noise residuals from the same camera to suppress
    scene content and retain the stable camera-specific pattern.
    Equivalent to the PRNU estimation step in classical forensics.
    """
    stack = np.stack(residuals, axis=0)
    return stack.mean(axis=0)


# ---------------------------------------------------------------------------
# Local NCC heatmap
# ---------------------------------------------------------------------------

def local_ncc(
    ref: np.ndarray,
    tgt: np.ndarray,
    patch: int = NCC_PATCH_SIZE,
    step: int  = NCC_STEP,
) -> np.ndarray:
    """
    Compute a sliding-window Normalized Cross-Correlation map between
    two noise residual images of the same spatial size.

    NCC near +1  -> consistent noise (same camera source)
    NCC near  0  -> inconsistent noise (possible tampering / different source)
    """
    h, w = ref.shape
    rows = list(range(0, h - patch, step))
    cols = list(range(0, w - patch, step))
    hmap = np.zeros((len(rows), len(cols)), dtype=np.float32)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            a = ref[r:r+patch, c:c+patch].ravel()
            b = tgt[r:r+patch, c:c+patch].ravel()
            a -= a.mean();  sa = a.std()
            b -= b.mean();  sb = b.std()
            if sa < 1e-6 or sb < 1e-6:
                hmap[i, j] = 0.0
            else:
                hmap[i, j] = float(np.dot(a, b) / (len(a) * sa * sb))
    return hmap


# ---------------------------------------------------------------------------
# Per-image detection
# ---------------------------------------------------------------------------

def detect(
    tgt_path: Path,
    camera_fingerprints: dict[str, np.ndarray],
    tgt_gray: np.ndarray,
) -> dict:
    """
    Run detection for one target image against all available camera fingerprints.
    Returns a result dict with per-camera NCC heatmaps and a best-match summary.
    """
    tgt_residual = extract_noise_residual(tgt_gray, image_path=tgt_path)
    tgt_camera   = parse_camera_model(tgt_path.name)

    results_per_camera: dict[str, np.ndarray] = {}
    for cam_model, fingerprint in camera_fingerprints.items():
        if fingerprint.shape != tgt_gray.shape:
            print(
                f"  [WARNING] Fingerprint shape mismatch for camera '{cam_model}' "
                f"vs target '{tgt_path.name}' -- skipped"
            )
            continue
        hmap = local_ncc(fingerprint, tgt_residual)
        results_per_camera[cam_model] = hmap

    # Best-match camera = highest mean NCC
    if results_per_camera:
        best_cam = max(results_per_camera, key=lambda c: results_per_camera[c].mean())
        best_mean_ncc = float(results_per_camera[best_cam].mean())
        # Mean NCC across all cameras OTHER than the best match
        other_scores = [
            float(hmap.mean())
            for cam, hmap in results_per_camera.items()
            if cam != best_cam
        ]
        not_match_mean_ncc = float(sum(other_scores) / len(other_scores)) if other_scores else float("nan")
    else:
        best_cam, best_mean_ncc, not_match_mean_ncc = None, float("nan"), float("nan")

    return {
        "path":               tgt_path,
        "expected_camera":    tgt_camera,
        "best_match_camera":  best_cam,
        "best_mean_ncc":      best_mean_ncc,
        "not_match_mean_ncc": not_match_mean_ncc,
        "heatmaps":           results_per_camera,
        "tgt_gray":           tgt_gray,
        "tgt_residual":       tgt_residual,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _upscale_hmap(hmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    sh, sw = hmap.shape
    th, tw = target_shape
    return zoom(hmap, (th / sh, tw / sw), order=1)


def save_detection_figure(result: dict, output_dir: Path) -> Path:
    """
    Save a per-image detection figure to output_dir.
    Files are written as: output/<target_stem>_detection.png
    """
    tgt_path    = result["path"]
    tgt_gray    = result["tgt_gray"]
    heatmaps    = result["heatmaps"]
    best_cam    = result["best_match_camera"]
    exp_cam     = result["expected_camera"]

    n_cams = len(heatmaps)
    if n_cams == 0:
        return None

    ncols = min(n_cams + 2, 6)   # original + residual + one col per camera (cap at 6)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5), facecolor="#12121f")
    if ncols == 1:
        axes = [axes]

    # Column 0: target image
    axes[0].imshow(tgt_gray, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Target\n{tgt_path.name}", color="white", fontsize=8)
    axes[0].axis("off")

    # Column 1: noise residual
    axes[1].imshow(result["tgt_residual"], cmap="RdBu", interpolation="nearest")
    axes[1].set_title("Noise residual", color="white", fontsize=8)
    axes[1].axis("off")

    # Remaining columns: NCC heatmaps per camera
    for idx, (cam, hmap) in enumerate(heatmaps.items()):
        ax_idx = idx + 2
        if ax_idx >= ncols:
            break
        ax = axes[ax_idx]
        hup = _upscale_hmap(hmap, tgt_gray.shape)
        ax.imshow(tgt_gray, cmap="gray", alpha=0.55, interpolation="nearest")
        im = ax.imshow(1 - hup, cmap="hot", alpha=0.55, vmin=0, vmax=1,
                       interpolation="bilinear")
        marker = " [BEST]" if cam == best_cam else ""
        ax.set_title(f"NCC vs {cam}{marker}\nmean={hmap.mean():.7f}",
                     color="white", fontsize=7.5)
        ax.axis("off")

    match_label = (
        "MATCH" if best_cam == exp_cam else "MISMATCH"
    )
    match_color = "#55dd88" if best_cam == exp_cam else "#ff6666"
    fig.suptitle(
        f"{tgt_path.name}  |  Expected: {exp_cam}  |  Best match: {best_cam}  [{match_label}]",
        color=match_color, fontsize=9, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = output_dir / f"{tgt_path.stem}_detection.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path



def save_fingerprint_figures(
    camera_fingerprints: dict[str, np.ndarray],
    camera_residuals: dict[str, list[np.ndarray]],
    output_dir: Path,
) -> list[Path]:
    """
    Save a visualization figure for each camera's aggregated noise fingerprint.

    Each figure contains:
      - Row 0: all individual noise residuals used to build the fingerprint
      - Row 1 (spanning full width): the averaged fingerprint with statistics

    Files are written as: output/fingerprint_<CameraModel>.png
    """
    saved: list[Path] = []

    for cam, fingerprint in camera_fingerprints.items():
        residuals   = camera_residuals.get(cam, [])
        n_src       = len(residuals)

        # Layout: top row = individual residuals (up to 8 shown),
        #         bottom row = averaged fingerprint + stats panel
        max_show    = min(n_src, 8)
        ncols       = max(max_show, 2)   # at least 2 cols for the bottom row
        fig         = plt.figure(figsize=(max(10, ncols * 2.6), 6.5), facecolor="#12121f")

        # --- Top row: individual residuals ---
        for i in range(max_show):
            ax = fig.add_subplot(2, ncols, i + 1)
            ax.imshow(residuals[i], cmap="RdBu", interpolation="nearest",
                      vmin=-10, vmax=10)
            ax.set_title(f"src residual {i+1}", color="#aaaacc", fontsize=7)
            ax.axis("off")

        # Fill remaining top-row slots if fewer than ncols sources
        for i in range(max_show, ncols):
            ax = fig.add_subplot(2, ncols, i + 1)
            ax.axis("off")

        # --- Bottom row: averaged fingerprint (spans ncols - 1 cols) ---
        ax_fp = fig.add_subplot(2, ncols, (ncols + 1, 2 * ncols - 1))
        im = ax_fp.imshow(fingerprint, cmap="RdBu", interpolation="nearest",
                          vmin=-10, vmax=10)
        ax_fp.set_title(
            f"Averaged fingerprint  ({n_src} source image{'s' if n_src != 1 else ''})",
            color="white", fontsize=9,
        )
        ax_fp.axis("off")
        plt.colorbar(im, ax=ax_fp, fraction=0.03, pad=0.02).ax.tick_params(
            colors="white", labelsize=7
        )

        # --- Bottom-right: statistics panel ---
        ax_st = fig.add_subplot(2, ncols, 2 * ncols)
        ax_st.axis("off")
        fp_flat = fingerprint.ravel()
        stats_text = "\n".join([
            f"Camera:  {cam}",
            f"Shape:   {fingerprint.shape[1]} x {fingerprint.shape[0]} px",
            f"Sources: {n_src} image{'s' if n_src != 1 else ''}",
            "",
            f"Mean:    {fp_flat.mean():+.4f}",
            f"Std:     {fp_flat.std():.4f}",
            f"Min:     {fp_flat.min():+.4f}",
            f"Max:     {fp_flat.max():+.4f}",
            f"SNR:     {fp_flat.mean() / (fp_flat.std() + 1e-9):.4f}",
        ])
        ax_st.text(
            0.08, 0.92, stats_text,
            transform=ax_st.transAxes,
            color="white", fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e1e3a", edgecolor="#444466"),
        )

        fig.suptitle(
            f"Noise fingerprint — {cam}",
            color="#88aaff", fontsize=11, fontweight="bold",
        )
        fig.tight_layout()

        out_path = output_dir / f"fingerprint_{cam}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(out_path)

    return saved

def save_summary_figure(all_results: list[dict], output_dir: Path) -> Path:
    """
    Save a summary bar chart of best-match mean NCC scores across all target images.
    Files are written as: output/summary.png
    """
    names       = [r["path"].name for r in all_results]
    ncc_scores  = [r["best_mean_ncc"] for r in all_results]
    colors      = [
        "#55dd88" if r["best_match_camera"] == r["expected_camera"] else "#ff6666"
        for r in all_results
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5), facecolor="#12121f")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(range(len(names)), ncc_scores, color=colors, edgecolor="#333355",
                  linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7, color="white")
    ax.set_ylabel("Best mean NCC", color="white", fontsize=9)
    ax.set_ylim(-0.1, 1.0)
    ax.tick_params(axis="y", colors="white")
    ax.spines[:].set_color("#333355")
    ax.set_title("Summary: best-match NCC per target image  (green=expected camera, red=mismatch)",
                 color="white", fontsize=9)
    fig.tight_layout()

    out_path = output_dir / "summary.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def write_summary_txt(all_results: list[dict], output_dir: Path) -> Path:
    """Write a plain-text summary report."""
    out_path = output_dir / "summary.txt"
    lines = [
        "Noiseprint PoC -- Detection Summary",
        "=" * 60,
        "",
        f"{'Target image':<40} {'Expected camera':<22} {'Best match':<22} {'Mean NCC (best)':>16} {'Mean NCC for other':<20}  {'Status'}",
        "-" * 135,
    ]
    n_match = 0
    for r in all_results:
        status = "MATCH" if r["best_match_camera"] == r["expected_camera"] else "MISMATCH"
        if status == "MATCH":
            n_match += 1
        not_match = r.get("not_match_mean_ncc", float("nan"))
        not_match_str = f"{not_match:>19.7f}" if not_match == not_match else f"{'N/A':>19}"
        lines.append(
            f"{r['path'].name:<40} {r['expected_camera']:<22} "
            f"{str(r['best_match_camera']):<22} {r['best_mean_ncc']:>16.7f} {not_match_str:<20}  {status}"
        )
    lines += [
        "-" * 135,
        f"Matched: {n_match} / {len(all_results)}  "
        f"({100*n_match/max(len(all_results),1):.1f}%)",
        "",
        "NOTE: 'Expected camera' is parsed from the VISION-style filename prefix.",
        "      A MISMATCH may indicate tampering or an unexpected source camera.",
        "      'Mean NCC for other' is the average NCC across all non-best-match cameras.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect source (reference) images ──────────────────────────────
    print(f"[1/6] Scanning source directory: {SRC_DIR}")
    if not SRC_DIR.exists():
        sys.exit(f"ERROR: Source directory '{SRC_DIR}' does not exist.")
    src_all = collect_image_paths(SRC_DIR)
    if not src_all:
        sys.exit(f"ERROR: No supported images found in '{SRC_DIR}'.")
    print(f"      Found {len(src_all)} source image(s).")

    src_accepted, src_rejected = filter_by_size_per_camera(src_all, label="src")
    if src_rejected:
        print(f"      {len(src_rejected)} source image(s) rejected due to size inconsistency.")
    if not src_accepted:
        sys.exit("ERROR: No source images passed the size filter.")

    # ── 2. Collect destination (target) images ────────────────────────────
    print(f"[2/6] Scanning destination directory: {DST_DIR}")
    if not DST_DIR.exists():
        sys.exit(f"ERROR: Destination directory '{DST_DIR}' does not exist.")
    dst_all = collect_image_paths(DST_DIR)
    if not dst_all:
        sys.exit(f"ERROR: No supported images found in '{DST_DIR}'.")
    print(f"      Found {len(dst_all)} destination image(s).")

    dst_accepted, dst_rejected = filter_by_size_per_camera(dst_all, label="dst")
    if dst_rejected:
        print(f"      {len(dst_rejected)} destination image(s) rejected due to size inconsistency.")
    if not dst_accepted:
        sys.exit("ERROR: No destination images passed the size filter.")

    # ── 3. Build per-camera fingerprints from src images ─────────────────
    backend = "Noiseprint CNN" if NOISEPRINT_CNN_AVAILABLE else "Gaussian high-pass (fallback)"
    print(f"[3/6] Extracting noise residuals and building camera fingerprints... (backend: {backend})")
    camera_residuals: dict[str, list[np.ndarray]] = defaultdict(list)

    for p in src_accepted:
        cam = parse_camera_model(p.name)
        try:
            gray, _ = load_gray(p)
            residual = extract_noise_residual(gray, image_path=p)
            camera_residuals[cam].append(residual)
            print(f"      [{cam}] loaded {p.name}  ({gray.shape[1]}x{gray.shape[0]})")
        except Exception as e:
            print(f"  [WARNING] Failed to process {p.name}: {e} -- skipped")

    camera_fingerprints: dict[str, np.ndarray] = {
        cam: build_camera_fingerprint(residuals)
        for cam, residuals in camera_residuals.items()
        if residuals
    }
    print(f"      Built fingerprints for {len(camera_fingerprints)} camera(s): "
          f"{list(camera_fingerprints.keys())}")

    print("[4/6] Saving fingerprint visualizations...")
    fp_figs = save_fingerprint_figures(camera_fingerprints, camera_residuals, OUTPUT_DIR)
    for p in fp_figs:
        print(f"      Fingerprint figure saved: {p.name}")

    # ── 4. Detect on each destination image ───────────────────────────────
    print("[5/6] Running detection on destination images...")
    all_results = []

    for p in dst_accepted:
        print(f"      Processing {p.name} ...")
        try:
            gray, _ = load_gray(p)
        except Exception as e:
            print(f"  [WARNING] Failed to load {p.name}: {e} -- skipped")
            continue

        result = detect(p, camera_fingerprints, gray)
        all_results.append(result)

        fig_path = save_detection_figure(result, OUTPUT_DIR)
        if fig_path:
            print(f"        -> saved {fig_path.name}  "
                  f"(best match: {result['best_match_camera']}, "
                  f"NCC={result['best_mean_ncc']:.4f})")

    # ── 5. Write summary ──────────────────────────────────────────────────
    print("[6/6] Writing summary...")
    if all_results:
        summary_png = save_summary_figure(all_results, OUTPUT_DIR)
        summary_txt = write_summary_txt(all_results, OUTPUT_DIR)
        print(f"      {summary_png}")
        print(f"      {summary_txt}")
    else:
        print("      No results to summarize.")

    print("\nDone.")


if __name__ == "__main__":
    main()