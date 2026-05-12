'''
Usage: python rgb-value-reader-center-30px-avg-with-notation-output.py \
        <image_path> \
        --pixels <pixels_file> \
        --resize <resize_factor>
'''

import sys
import os
import csv
import argparse
from PIL import Image, ImageDraw, ImageFont


WINDOW_SIZE = 100
HALF = WINDOW_SIZE // 2


# Read pixel values from CSV, format in: name,x,y
def load_pixels_from_csv(csv_path):
    pixels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            name = row[0].strip()
            x = int(row[1])
            y = int(row[2])
            pixels[name] = (x, y)
    return pixels


# Check if file is HEIF, otherwise enables pillow-heif
def maybe_enable_heif(image_path: str):

    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".heic", ".heif"):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            print("  [ERROR] HEIC/HEIF detected, but require pillow-heif library.")
            print("  Please install pillow-heif first.")
            return False
    return True


def average_rgb_in_window(img, cx, cy, width, height):

    x0 = max(0, cx - HALF)
    x1 = min(width, cx + HALF)
    y0 = max(0, cy - HALF)
    y1 = min(height, cy + HALF)

    total_r = total_g = total_b = 0
    count = 0

    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = img.getpixel((x, y))
            total_r += r
            total_g += g
            total_b += b
            count += 1

    if count == 0:
        return None

    return (
        total_r / count,
        total_g / count,
        total_b / count,
    )


# With center of (cx, cy), draw a suqare marker of size 100x100, with label.
def draw_marker(draw, c_name, cx, cy, width, height, resize_factor):

    x0 = max(0, cx - HALF)
    x1 = min(width - 1, cx + HALF)      # Minus 1 to prevent out-of-the-boarder.
    y0 = max(0, cy - HALF)
    y1 = min(height - 1, cy + HALF)

    # Draw red-color edge.
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        outline="red",
        width=2
    )

    # Draw cross-mark center point.
    draw.line([(cx - 5, cy), (cx + 5, cy)], fill="red", width=1)
    draw.line([(cx, cy - 5), (cx, cy + 5)], fill="red", width=1)

    # Draw the text remarks
    text = f"{c_name} ({cx},{cy})"

    try:
        font = ImageFont.load_default(size=50 * resize_factor)
    except Exception:
        font = None

    # text_x = min(cx + 5, width - 1)
    # text_y = max(cy - 15, 0)
    text_x = cx - 200 * resize_factor
    text_y = cy - 220 * resize_factor

    draw.text(
        (text_x, text_y),
        text,
        fill="red",
        font=font
    )


def process_image(image_path: str, pixels: dict, resize_factor: float):
    print("=" * 72)
    print(f"Reading file: {image_path}")

    if not os.path.isfile(image_path):
        print("  [ERROR] File not exist.")
        return

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Could not open image: {e}")
        return

    width, height = img.size
    print(f"  Inputted Image Size: {width} x {height}")

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    # Output path of CSV.
    base, _ = os.path.splitext(image_path)
    csv_path = base + "_marked.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Pixel Name", "X", "Y",
            "Single_R", "Single_G", "Single_B",
            "Avg_R", "Avg_G", "Avg_B"
        ])

        for name, (x, y) in pixels.items():

            # Apply resize.
            sx = int(x * resize_factor)
            sy = int(y * resize_factor)

            print(f"\n  Pixel [{name}] ({sx},{sy}):")

            if not (0 <= sx < width and 0 <= sy < height):
                print("    ERROR: Out of the range.")
                continue

            r, g, b = img.getpixel((sx, sy))
            print(f"    Single Pixel RGB: R={r}, G={g}, B={b}")

            avg = average_rgb_in_window(img, sx, sy, width, height)
            if avg is None:
                print("    ERROR: Could not calculate.")
                ar = ag = ab = None
            else:
                ar, ag, ab = avg
                print(
                    f"    {WINDOW_SIZE}x{WINDOW_SIZE} area Average RGB Value: "
                    f"R={ar:.2f}, G={ag:.2f}, B={ab:.2f}"
                )

            writer.writerow([
                name, sx, sy,
                r, g, b,
                f"{ar:.2f}" if avg else "",
                f"{ag:.2f}" if avg else "",
                f"{ab:.2f}" if avg else ""
            ])

            draw_marker(draw, name, sx, sy, width, height, resize_factor)

    output_path = base + "_marked.png"
    draw_img.save(output_path, format="PNG")

    print(f"\n  Marking completed, output to: {output_path}")
    print(f"  CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Images to process")
    parser.add_argument("--pixels", help="CSV file containing pixel list")
    parser.add_argument("--resize", type=float, default=1.0,
                        help="Resize factor (0~1), default=1.0")

    args = parser.parse_args()

    # Load pixels.
    if args.pixels:
        pixels = load_pixels_from_csv(args.pixels)
    else:
        print("Error: no pixels file specified.")
        exit(1)

    # Process every inputted images.
    for image_path in args.images:
        process_image(image_path, pixels, args.resize)


if __name__ == "__main__":
    main()