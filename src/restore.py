"""Classic restoration pipeline on one image: denoise -> damage mask -> inpaint -> deblur.

Usage:  python src/restore.py image.png [--noise gaussian|saltpepper|none] [--blur 9] [--balance 0.2]
Saves every step and a side-by-side overview in results/restore/<image name>/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.restoration import estimate_sigma, wiener

from damage import motion_psf

ROOT = Path(__file__).resolve().parent.parent


def denoise(img, noise="gaussian"):
    if noise == "saltpepper":
        return cv2.medianBlur(img, 3)
    if noise == "gaussian":
        h = max(3.0, float(estimate_sigma(img)))  # filter strength follows the estimated noise level
        return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
    return img


def detect_scratches(img):
    # bright thin lines survive a white top-hat; small opening removes them
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    # keep only detections that lie on a straight segment of at least 40 px (texture is not straight)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=5)
    on_lines = np.zeros_like(mask)
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
        cv2.line(on_lines, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)
    return cv2.dilate(cv2.bitwise_and(mask, on_lines), np.ones((3, 3), np.uint8))


def detect_holes(img, min_area=64):
    # missing regions: large blobs of (almost) pure black
    mask = cv2.morphologyEx((img < 10).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    holes = np.isin(labels, keep).astype(np.uint8) * 255
    return cv2.dilate(holes, np.ones((9, 9), np.uint8))  # denoising softens the edges, so grow the mask a bit


def wiener_restore(img, psf, balance=0.02, pad=32):
    # mirror the borders first, otherwise Wiener makes stripes at the image edges
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    out = wiener(big / 255.0, psf, balance)[pad:-pad, pad:-pad]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def restore(img, noise="gaussian", blur_len=0, balance=0.2):
    steps = {"0_input": img}
    steps["1_denoised"] = denoise(img, noise)
    steps["2_mask"] = cv2.bitwise_or(detect_scratches(steps["1_denoised"]), detect_holes(steps["1_denoised"]))
    steps["3_inpainted"] = cv2.inpaint(steps["1_denoised"], steps["2_mask"], 3, cv2.INPAINT_TELEA)
    steps["4_deblurred"] = wiener_restore(steps["3_inpainted"], motion_psf(blur_len), balance) if blur_len else steps["3_inpainted"]
    return steps


def overview(steps, width=300):
    tiles = []
    for name, im in steps.items():
        h = int(im.shape[0] * width / im.shape[1])
        tile = cv2.resize(im, (width, h))
        label = np.full((28, width), 255, np.uint8)
        cv2.putText(label, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
        tiles.append(np.vstack([label, tile]))
    gap = np.full((tiles[0].shape[0], 6), 255, np.uint8)
    return np.hstack([x for t in tiles for x in (t, gap)][:-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--noise", default="gaussian", choices=["gaussian", "saltpepper", "none"])
    p.add_argument("--blur", type=int, default=0, help="length of horizontal motion blur in px (0 = no deblurring)")
    p.add_argument("--balance", type=float, default=0.2, help="Wiener noise-to-signal ratio (higher = smoother)")
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    steps = restore(img, args.noise, args.blur, args.balance)

    out = ROOT / "results/restore" / Path(args.image).stem
    out.mkdir(parents=True, exist_ok=True)
    for name, im in steps.items():
        cv2.imwrite(str(out / f"{name}.png"), im)
    cv2.imwrite(str(out / "overview.png"), overview(steps))
    print(f"saved {len(steps)} steps + overview.png in {out}")


if __name__ == "__main__":
    main()
