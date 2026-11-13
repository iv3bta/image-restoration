"""Synthetic damage + classic restoration baselines, scored with PSNR / SSIM.

Usage:  python src/evaluate.py [images_folder]
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from damage import blur, gaussian_noise, holes, motion_psf, salt_pepper, scratches
from restore import detect_scratches, wiener_restore

SEED = 0
IMAGES = ["lena512.bmp", "photo.jpg", "ct.jpg", "Fig5.03.jpg"]
LAB_NOISY = ["Fig5.03_zgomotGaussian.jpg", "Fig5.03_zgomotSarePiper.jpg", "Fig5.03_zgomotUniform.jpg"]
OUT = Path(__file__).resolve().parent.parent / "results"


# damage functions live in damage.py, restoration steps in restore.py


DENOISERS = {
    "median 3x3": lambda d: cv2.medianBlur(d, 3),
    "bilateral (9,75,75)": lambda d: cv2.bilateralFilter(d, 9, 75, 75),
    "NL-means (h=20)": lambda d: cv2.fastNlMeansDenoising(d, None, 20, 7, 21),
}


# ---------- metrics ----------

def score(clean, restored):
    psnr = peak_signal_noise_ratio(clean, restored, data_range=255)
    ssim = structural_similarity(clean, restored, data_range=255)
    return psnr, ssim


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


def run_image(name, clean, rng, rows):
    psf = motion_psf()
    tests = {}

    tests["gaussian noise sigma=25"] = (gaussian_noise(clean, rng), DENOISERS)
    tests["salt & pepper 5%"] = (salt_pepper(clean, rng), DENOISERS)
    tests["motion blur 9px + noise sigma=2"] = (blur(clean, rng, psf), {"Wiener (balance=0.02, mirrored edges)": lambda d: wiener_restore(d, psf)})

    scr, scr_mask = scratches(clean, rng)
    det_mask = detect_scratches(scr)
    tests["scratches (15 lines)"] = (scr, {
        "TELEA, detected mask": lambda d: cv2.inpaint(d, det_mask, 3, cv2.INPAINT_TELEA),
        "NS, detected mask": lambda d: cv2.inpaint(d, det_mask, 3, cv2.INPAINT_NS),
        "TELEA, true mask": lambda d: cv2.inpaint(d, scr_mask, 3, cv2.INPAINT_TELEA),
        "NS, true mask": lambda d: cv2.inpaint(d, scr_mask, 3, cv2.INPAINT_NS),
    })

    hol, hol_mask = holes(clean, rng)
    tests["missing regions (4 boxes)"] = (hol, {
        "TELEA": lambda d: cv2.inpaint(d, hol_mask, 5, cv2.INPAINT_TELEA),
        "NS": lambda d: cv2.inpaint(d, hol_mask, 5, cv2.INPAINT_NS),
    })

    folder = OUT / Path(name).stem
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "scratches__mask_true.png"), scr_mask)
    cv2.imwrite(str(folder / "scratches__mask_detected.png"), det_mask)
    for defect, (damaged, methods) in tests.items():
        cv2.imwrite(str(folder / f"{slug(defect)}__damaged.png"), damaged)
        rows.append((name, defect, "none (damaged)", *score(clean, damaged)))
        for method, fn in methods.items():
            restored = fn(damaged)
            cv2.imwrite(str(folder / f"{slug(defect)}__{slug(method)}.png"), restored)
            rows.append((name, defect, method, *score(clean, restored)))

    return iou(det_mask, scr_mask)


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "PI/Lab2_PIM/Lab2_PIM/Images"
    rng = np.random.default_rng(SEED)
    rows, ious = [], {}

    for name in IMAGES:
        clean = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        ious[name] = run_image(name, clean, rng, rows)

    # lab-provided noisy versions of Fig5.03 (noise level unknown)
    clean = cv2.imread(str(folder / "Fig5.03.jpg"), cv2.IMREAD_GRAYSCALE)
    lab_rows = []
    for name in LAB_NOISY:
        noisy = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        offset = noisy.mean() - clean.mean()
        lab_rows.append((name, offset, "none (damaged)", *score(clean, noisy)))
        for method, fn in DENOISERS.items():
            lab_rows.append((name, offset, method, *score(clean, fn(noisy))))

    lines = ["# Results - classic baselines", "",
             f"Seed: {SEED}. Grayscale. PSNR in dB (higher is better), SSIM in [0, 1] (higher is better).", "",
             "| Image | Defect | Method | PSNR | SSIM |", "|---|---|---|---:|---:|"]
    lines += [f"| {i} | {d} | {m} | {p:.2f} | {s:.3f} |" for i, d, m, p, s in rows]
    lines += ["", "## Scratch mask detection (white top-hat, threshold 40)", "",
              "| Image | IoU with true mask |", "|---|---:|"]
    lines += [f"| {i} | {v:.2f} |" for i, v in ious.items()]
    lines += ["", "## Lab-provided noisy images (Fig5.03, unknown noise level)", "",
              "| Noisy image | Mean offset vs clean | Method | PSNR | SSIM |", "|---|---:|---|---:|---:|"]
    lines += [f"| {i} | {o:+.1f} | {m} | {p:.2f} | {s:.3f} |" for i, o, m, p, s in lab_rows]
    lines += ["", "Note: a large mean offset means the lab image is globally brighter than the clean one",
              "(non-zero-mean noise or rescaling). Local filters cannot remove a global offset, so PSNR",
              "on those rows mostly measures the offset; SSIM is the more informative column there."]

    OUT.mkdir(exist_ok=True)
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
