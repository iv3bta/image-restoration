"""Runs the pretrained Restormer on one image and shows input / output side by side.

Usage:  python src/run_restormer.py image.png [--noise 25] [--show 6]
  --noise S  first adds Gaussian noise of sigma S (use on a clean image, so PSNR can be measured)
  --show T   seconds to show the result window (0 = don't show)
"""
import argparse
import ctypes
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from test_restormer import denoise, load_model

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models/gaussian_gray_denoising_blind.pth"


def labeled(img, text, height=420):
    img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
    bar = np.full((34, img.shape[1]), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2)
    return np.vstack([bar, img])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--noise", type=float, default=0)
    p.add_argument("--show", type=float, default=6)
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    print(f"image: {args.image} ({img.shape[1]} x {img.shape[0]} px)")

    start = time.time()
    model = load_model(WEIGHTS)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: Restormer, pretrained weights {WEIGHTS.name}, {params:.1f} M parameters, loaded in {time.time() - start:.1f} s")

    if args.noise:
        np.random.seed(0)
        x = img / 255.0 + np.random.normal(0, args.noise / 255.0, img.shape)
        noisy8 = np.clip(np.round(x * 255), 0, 255).astype(np.uint8)
        print(f"added Gaussian noise, sigma = {args.noise:g}  ->  PSNR {psnr(img, noisy8):.2f} dB")
    else:
        x, noisy8 = img / 255.0, img

    print(f"running Restormer on {next(model.parameters()).device.type.upper()} ...", flush=True)
    start = time.time()
    out = denoise(model, x)
    print(f"done in {time.time() - start:.1f} s")
    if args.noise:
        print(f"PSNR after Restormer: {psnr(img, out):.2f} dB  (noisy: {psnr(img, noisy8):.2f} dB)")

    save = ROOT / "results/restormer" / (Path(args.image).stem + "_restormer.png")
    save.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save), out)
    print(f"saved {save.relative_to(ROOT)}")

    if args.show:
        tiles = [labeled(img, "original"), labeled(noisy8, "noisy input"), labeled(out, "Restormer")] if args.noise \
            else [labeled(img, "input"), labeled(out, "Restormer")]
        gap = np.full((tiles[0].shape[0], 10), 255, np.uint8)
        row = np.hstack([t for tile in tiles for t in (tile, gap)][:-1])
        name = "Restormer - " + Path(args.image).name
        if sys.platform == "win32":
            ctypes.windll.user32.SetProcessDPIAware()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, row)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(name, 1880 - row.shape[1], 330)
        cv2.waitKey(int(args.show * 1000))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
