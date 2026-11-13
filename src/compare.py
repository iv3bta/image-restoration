"""Compares classic pipeline vs pretrained Restormer vs fine-tuned Restormer on the same test images.

Test sets:
  bsd68_mixed   BSD68 + a random mix of our damage (same kinds as in training, fixed seed per image)
  bsd68_sigma25 BSD68 + Gaussian noise sigma=25 (the paper's setting: checks nothing was forgotten)
  film_test     real damaged film scans not used in training, scored against the expert restoration

Usage:  python src/compare.py [--limit 5] [--finetuned models/restormer_finetuned.pth]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from finetune_restormer import load_gray, random_damage
from restore import restore
from test_restormer import denoise, load_model

ROOT = Path(__file__).resolve().parent.parent


def test_sets(limit, film_train):
    bsd = sorted((ROOT / "data/BSD68").glob("*.png"))[:limit]
    for i, f in enumerate(bsd):
        clean = load_gray(f)
        yield "bsd68_mixed", f.name, random_damage(clean, np.random.default_rng(1000 + i)), clean
    for f in bsd:
        clean = load_gray(f)
        np.random.seed(0)
        noisy = clean / 255.0 + np.random.normal(0, 25 / 255.0, clean.shape)
        yield "bsd68_sigma25", f.name, noisy, clean
    film = sorted((ROOT / "data/film_scans/damaged").glob("*.jpg"))[film_train:][:limit]
    for f in film:
        expert = load_gray(f.parent.parent / "restored" / f.name.replace(".jpg", "_restored.jpg"), 0.5)
        yield "film_test", f.name, load_gray(f, 0.5), expert


def damaged_area_psnr(target, damaged, out):
    # PSNR only where the input differs clearly from the target (the damage), grown by a few pixels
    area = cv2.dilate((np.abs(damaged.astype(int) - target.astype(int)) > 20).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if not area.any():
        return np.nan
    mse = np.mean((out[area].astype(float) - target[area]) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def to_uint8(img):
    return img if img.dtype == np.uint8 else np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=1000, help="max images per test set")
    p.add_argument("--finetuned", default=ROOT / "models/restormer_finetuned.pth")
    p.add_argument("--film-train", type=int, default=30, help="same split as in finetune_restormer.py")
    args = p.parse_args()

    models = {"pretrained Restormer": load_model(ROOT / "models/gaussian_gray_denoising_blind.pth")}
    if Path(args.finetuned).exists():
        models["fine-tuned Restormer"] = load_model(args.finetuned)

    scores = {}  # (test set, method) -> list of (psnr, ssim)
    examples = ROOT / "results/compare"
    examples.mkdir(parents=True, exist_ok=True)
    for test_set, name, damaged, target in test_sets(args.limit, args.film_train):
        as_float = damaged if damaged.dtype != np.uint8 else damaged / 255.0
        outputs = {"damaged": to_uint8(damaged), "classic pipeline": restore(to_uint8(damaged))["4_deblurred"]}
        for label, model in models.items():
            outputs[label] = denoise(model, as_float)
        for method, out in outputs.items():
            s = (peak_signal_noise_ratio(target, out, data_range=255), structural_similarity(target, out, data_range=255),
                 damaged_area_psnr(target, outputs["damaged"], out))
            scores.setdefault((test_set, method), []).append(s)
        cv2.imwrite(str(examples / f"{test_set}_{Path(name).stem}.png"), np.hstack([target, *outputs.values()]))
        print(f"{test_set} {name}: " + ", ".join(f"{m} {s[-1][0]:.2f}" for (t, m), s in scores.items() if t == test_set), flush=True)

    lines = ["# Classic vs pretrained vs fine-tuned", "",
             "Each example image in results/compare/ shows: target | " + " | ".join(outputs), "",
             "| Test set | Images | Method | PSNR (dB) | SSIM | PSNR on damaged area (dB) |", "|---|---:|---|---:|---:|---:|"]
    for (test_set, method), s in scores.items():
        s = np.array(s)
        lines.append(f"| {test_set} | {len(s)} | {method} | {s[:, 0].mean():.2f} | {s[:, 1].mean():.3f} | {np.nanmean(s[:, 2]):.2f} |")
    (ROOT / "results/compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
