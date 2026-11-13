"""Synthetic damage + classic restoration baselines, scored with PSNR / SSIM.

Usage:  python src/evaluate.py [images_folder]
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import wiener

SEED = 0
IMAGES = ["lena512.bmp", "photo.jpg", "ct.jpg", "Fig5.03.jpg"]
LAB_NOISY = ["Fig5.03_zgomotGaussian.jpg", "Fig5.03_zgomotSarePiper.jpg", "Fig5.03_zgomotUniform.jpg"]
OUT = Path(__file__).resolve().parent.parent / "results"


# ---------- damage ----------

def gaussian_noise(img, rng, sigma=25):
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_pepper(img, rng, amount=0.05):
    out = img.copy()
    r = rng.random(img.shape)
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out


def motion_psf(length=9):
    psf = np.zeros((length, length))
    psf[length // 2, :] = 1  # horizontal motion
    return psf / psf.sum()


def blur(img, rng, psf, sigma=2):
    blurred = cv2.filter2D(img.astype(np.float64), -1, psf)
    blurred += rng.normal(0, sigma, img.shape)  # a little sensor noise
    return np.clip(blurred, 0, 255).astype(np.uint8)


def scratches(img, rng, count=15):
    out, mask = img.copy(), np.zeros_like(img)
    h, w = img.shape
    for _ in range(count):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        thick = int(rng.integers(1, 3))
        cv2.line(out, p1, p2, int(rng.integers(220, 256)), thick)
        cv2.line(mask, p1, p2, 255, thick)
    return out, mask


def holes(img, rng, count=4):
    out, mask = img.copy(), np.zeros_like(img)
    h, w = img.shape
    for _ in range(count):
        bw, bh = int(rng.integers(w // 16, w // 8)), int(rng.integers(h // 16, h // 8))
        x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        out[y:y + bh, x:x + bw] = 0
        mask[y:y + bh, x:x + bw] = 255
    return out, mask


# ---------- restoration ----------

def detect_scratches(img):
    # bright thin lines survive a white top-hat; small opening removes them
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))


def wiener_restore(img, psf, balance=0.05):
    out = wiener(img / 255.0, psf, balance)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


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
    tests["motion blur 9px + noise sigma=2"] = (blur(clean, rng, psf), {"Wiener (balance=0.05)": lambda d: wiener_restore(d, psf)})

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
