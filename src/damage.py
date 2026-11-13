"""Synthetic damage: noise, blur, scratches, missing regions (all on grayscale uint8 images)."""
import cv2
import numpy as np


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
