"""
delta_e_2000_image_checker.py

Compare two images using the CIEDE2000 (ΔE00) color difference formula.
Each image can independently specify its color space and white point.
"""

import argparse
import sys
import numpy as np
from PIL import Image


# ── Global color space and white point definitions ───────────────────────────

SRGB_PRIMARIES = {
    "r": (0.64, 0.33),
    "g": (0.30, 0.60),
    "b": (0.15, 0.06),
}

DISPLAY_P3_PRIMARIES = {
    "r": (0.680, 0.320),
    "g": (0.265, 0.690),
    "b": (0.150, 0.060),
}

D65 = (0.95047, 1.00000, 1.08883)
D50 = (0.96422, 1.00000, 0.82521)


# ── Color conversion utilities ───────────────────────────────────────────────

def linearize(channel: np.ndarray, gamma) -> np.ndarray:
    """
    Remove gamma encoding from a channel array.

    Args:
        channel: Float array of pixel values in [0, 255].
        gamma:   'srgb' for the piecewise sRGB transfer function (IEC 61966-2-1),
                 or a numeric exponent for a simple power-law curve.

    Returns:
        Linear-light float array in [0, 1].
    """
    c = channel / 255.0
    if gamma == "srgb":
        return np.where(
            c <= 0.04045,
            c / 12.92,
            ((c + 0.055) / 1.055) ** 2.4,
        )
    return c ** float(gamma)


def build_xyz_matrix(primaries: dict, white_point: tuple) -> np.ndarray:
    """
    Construct the 3x3 linear RGB -> XYZ matrix for the given primaries and
    white point, following the standard CIE derivation.

    Args:
        primaries:   Dict with keys 'r', 'g', 'b', each a (x, y) chromaticity.
        white_point: (X, Y, Z) tristimulus values of the reference white.

    Returns:
        3x3 numpy array M such that XYZ = M @ rgb_linear.
    """
    def xy_to_XYZ(x, y):
        return np.array([x / y, 1.0, (1.0 - x - y) / y])

    Xr, Xg, Xb = [xy_to_XYZ(*primaries[c]) for c in ("r", "g", "b")]
    M = np.column_stack([Xr, Xg, Xb])
    W = np.array(white_point)
    S = np.linalg.solve(M, W)
    return M * S  # scale each column by its corresponding S coefficient


# Bradford chromatic adaptation matrix D65 -> D50
BRADFORD_D65_TO_D50 = np.array([
    [ 1.0478112,  0.0228866, -0.0501270],
    [ 0.0295424,  0.9904844, -0.0170491],
    [-0.0092345,  0.0150436,  0.7521316],
])

# Bradford chromatic adaptation matrix D50 -> D65 (inverse of above)
BRADFORD_D50_TO_D65 = np.linalg.inv(BRADFORD_D65_TO_D50)


def adapt_white_point(xyz: np.ndarray, src_wp: tuple, dst_wp: tuple) -> np.ndarray:
    """
    Apply Bradford chromatic adaptation to convert XYZ values from one white
    point to another. Only D65 <-> D50 conversions are supported.

    Args:
        xyz:    Array of shape (..., 3) in XYZ.
        src_wp: Source white point tuple, one of D65 or D50.
        dst_wp: Destination white point tuple, one of D65 or D50.

    Returns:
        Adapted XYZ array of the same shape.
    """
    if src_wp == dst_wp:
        return xyz

    if src_wp == D65 and dst_wp == D50:
        M = BRADFORD_D65_TO_D50
    elif src_wp == D50 and dst_wp == D65:
        M = BRADFORD_D50_TO_D65
    else:
        raise ValueError(
            f"Unsupported white-point adaptation: {src_wp} -> {dst_wp}. "
            "Only D65 and D50 are supported."
        )
    return xyz @ M.T


def xyz_to_lab(xyz: np.ndarray, white_point: tuple) -> np.ndarray:
    """
    Convert XYZ to CIE L*a*b* (CIELAB, 1976).

    Args:
        xyz:         Array of shape (..., 3).
        white_point: (X, Y, Z) of the reference white used for normalisation.

    Returns:
        L*a*b* array of the same shape.
    """
    xyz_n = xyz / np.array(white_point)

    delta = 6.0 / 29.0

    def f(t):
        return np.where(
            t > delta ** 3,
            t ** (1.0 / 3.0),
            t / (3.0 * delta ** 2) + 4.0 / 29.0,
        )

    fx = f(xyz_n[..., 0])
    fy = f(xyz_n[..., 1])
    fz = f(xyz_n[..., 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


# ── CIEDE2000 formula ────────────────────────────────────────────────────────

def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Compute the CIEDE2000 color difference between two L*a*b* arrays.

    Args:
        lab1: Array of shape (..., 3) — reference colors.
        lab2: Array of shape (..., 3) — sample colors.

    Returns:
        ΔE00 array of shape (...).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Step 1 — a' adjustment (G factor)
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0 ** 7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    # Step 2 — hue angles h'
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    # Step 3 — delta L', delta C', delta H'
    dLp = L2 - L1
    dCp = C2p - C1p

    h_diff = h2p - h1p
    dhp = np.where(
        np.abs(h_diff) <= 180.0,
        h_diff,
        np.where(h_diff > 180.0, h_diff - 360.0, h_diff + 360.0),
    )
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    # Step 4 — CIEDE2000 weighting functions
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    hp_avg = np.where(
        np.abs(h1p - h2p) <= 180.0,
        (h1p + h2p) / 2.0,
        np.where(
            h1p + h2p < 360.0,
            (h1p + h2p + 360.0) / 2.0,
            (h1p + h2p - 360.0) / 2.0,
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_avg - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_avg))
        + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0))
    )

    SL = 1.0 + 0.015 * (Lp_avg - 50.0) ** 2 / np.sqrt(20.0 + (Lp_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    # Step 5 — RT rotation term (blue-purple hue correction)
    Cp_avg7 = Cp_avg ** 7
    d_theta = 30.0 * np.exp(-((hp_avg - 275.0) / 25.0) ** 2)
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0 ** 7))
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    return np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


# ── Image processing pipeline ────────────────────────────────────────────────

COLOR_SPACES = {
    "srgb":      SRGB_PRIMARIES,
    "displayp3": DISPLAY_P3_PRIMARIES,
}

WHITE_POINTS = {
    "d65": D65,
    "d50": D50,
}

# Internal Lab reference white point used for both images.
# Both images are Bradford-adapted to this point before Lab conversion,
# ensuring a consistent comparison regardless of each image's original
# white point declaration.
LAB_REFERENCE_WHITE = D50


def image_to_lab(path: str, color_space: str, white_point: tuple) -> np.ndarray:
    """
    Load an image and convert every pixel to CIE L*a*b*.

    Pipeline:
        R'G'B' (gamma-encoded)
            -> [Linearization]
        RGB_linear
            -> [RGB-to-XYZ matrix built from primaries + white point]
        XYZ  (native white point of this image)
            -> [Bradford adaptation -> LAB_REFERENCE_WHITE]
        XYZ_adapted
            -> [xyz_to_lab normalised by LAB_REFERENCE_WHITE]
        L*a*b*

    Args:
        path:        Path to the image file.
        color_space: One of 'srgb' or 'displayp3'.
        white_point: White point tuple (D65 or D50).

    Returns:
        L*a*b* float32 array of shape (H, W, 3).
    """
    primaries = COLOR_SPACES[color_space]
    gamma = "srgb" if color_space == "srgb" else 2.2  # Display P3 uses gamma 2.2

    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)

    # Linearize each channel independently
    rgb_linear = np.stack(
        [linearize(img[..., c], gamma) for c in range(3)],
        axis=-1,
    )

    # Linear RGB -> XYZ (in the image's declared white point)
    M = build_xyz_matrix(primaries, white_point)
    xyz = rgb_linear @ M.T

    # Chromatic adaptation to the shared Lab reference white (D50)
    xyz_adapted = adapt_white_point(xyz, white_point, LAB_REFERENCE_WHITE)

    return xyz_to_lab(xyz_adapted, LAB_REFERENCE_WHITE)


# ── CLI argument parser ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="delta_e_2000_image_checker.py",
        description=(
            "Compare two images pixel-by-pixel using the CIEDE2000 (ΔE00)\n"
            "color difference formula.\n\n"
            "Each image may independently declare its color space and white point.\n"
            "Both images must have identical pixel dimensions."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    for n in (1, 2):
        parser.add_argument(
            f"--image_{n}",
            required=True,
            metavar="PATH",
            help=f"Path to image {n}.",
        )
        parser.add_argument(
            f"--color_space_{n}",
            default="srgb",
            choices=["srgb", "displayp3"],
            metavar="COLOR_SPACE",
            help=(
                f"Color space of image {n}.\n"
                "  srgb      — sRGB IEC 61966-2-1 (default)\n"
                "  displayp3 — Display P3 (DCI-P3 primaries, D65 white point, gamma 2.2)"
            ),
        )
        parser.add_argument(
            f"--white_point_{n}",
            default="d65",
            choices=["d65", "d50"],
            metavar="WHITE_POINT",
            help=(
                f"Reference white point of image {n}.\n"
                "  d65 — CIE Illuminant D65 (default)\n"
                "  d50 — CIE Illuminant D50"
            ),
        )

    return parser


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths        = (args.image_1,        args.image_2)
    color_spaces = (args.color_space_1,  args.color_space_2)
    wp_keys      = (args.white_point_1,  args.white_point_2)
    white_points = (WHITE_POINTS[wp_keys[0]], WHITE_POINTS[wp_keys[1]])

    # ── Validate image files and check dimension parity ──────────────────────
    sizes = []
    for path in paths:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)  # (width, height)
        except FileNotFoundError:
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: Could not open '{path}': {exc}", file=sys.stderr)
            sys.exit(1)

    if sizes[0] != sizes[1]:
        print(
            "WARNING: Image size mismatch — "
            f"image_1 is {sizes[0][0]}x{sizes[0][1]} px, "
            f"image_2 is {sizes[1][0]}x{sizes[1][1]} px. "
            "Both images must have identical dimensions. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Convert both images to L*a*b* ─────────────────────────────────────────
    for n, (path, cs, wp_key) in enumerate(zip(paths, color_spaces, wp_keys), start=1):
        print(f"Processing image {n}: {path}")
        print(f"  Color space : {cs}")
        print(f"  White point : {wp_key.upper()}")

    print()
    lab1 = image_to_lab(paths[0], color_spaces[0], white_points[0])
    lab2 = image_to_lab(paths[1], color_spaces[1], white_points[1])

    # ── Compute ΔE00 ──────────────────────────────────────────────────────────
    print("Computing ΔE00 for all pixels ...")
    de_map = delta_e_2000(lab1, lab2)

    # ── Report statistics ─────────────────────────────────────────────────────
    print()
    print("── ΔE00 Results " + "─" * 34)
    print(f"  Image size          : {sizes[0][0]} x {sizes[0][1]} px ({de_map.size:,} pixels)")
    print(f"  Mean   ΔE00         : {de_map.mean():.4f}")
    print(f"  Median ΔE00         : {float(np.median(de_map)):.4f}")
    print(f"  Max    ΔE00         : {de_map.max():.4f}")
    print(f"  Pixels with ΔE00 > 1 : {(de_map > 1).mean() * 100:.2f}%")
    print(f"  Pixels with ΔE00 > 2 : {(de_map > 2).mean() * 100:.2f}%")
    print(f"  Pixels with ΔE00 > 5 : {(de_map > 5).mean() * 100:.2f}%")
    print("─" * 50)


if __name__ == "__main__":
    main()