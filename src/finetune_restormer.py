"""Fine-tunes the pretrained Restormer (grayscale denoising) on our four kinds of damage.

Training pairs are made on the fly: a random crop of a clean image gets a random mix of
noise, motion blur, scratches and missing regions; the model learns to give back the clean crop.
Real pairs from the film scans (damaged -> expert restoration) are mixed in.

Usage:  python src/finetune_restormer.py [--iters 20000] [--batch 4] [--patch 128]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import damage
from test_restormer import load_model

ROOT = Path(__file__).resolve().parent.parent


def random_damage(clean, rng):
    img = clean
    if rng.random() < 0.5:
        img = damage.blur(img, rng, damage.motion_psf(int(rng.choice([5, 7, 9, 11]))), sigma=0)
    if rng.random() < 0.5:
        img = damage.scratches(img, rng, count=int(rng.integers(1, 5)))[0]
    if rng.random() < 0.3:
        img = damage.holes(img, rng, count=int(rng.integers(1, 3)))[0]
    if rng.random() < 0.2:
        img = damage.salt_pepper(img, rng, amount=float(rng.uniform(0.01, 0.05)))
    if rng.random() < 0.7:
        img = damage.gaussian_noise(img, rng, sigma=float(rng.uniform(5, 50)))
    return img


def random_crop(images, size, rng):
    h, w = images[0].shape
    y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    return [im[y:y + size, x:x + size] for im in images]


def load_gray(path, scale=1.0):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img


def make_batch(clean_images, film_pairs, batch, patch, rng):
    inputs, targets = [], []
    for _ in range(batch):
        if film_pairs and rng.random() < 0.25:  # real damage
            damaged, restored = random_crop(film_pairs[int(rng.integers(len(film_pairs)))], patch, rng)
        else:  # synthetic damage
            restored = random_crop([clean_images[int(rng.integers(len(clean_images)))]], patch, rng)[0]
            damaged = random_damage(restored, rng)
        inputs.append(damaged)
        targets.append(restored)
    to_tensor = lambda x: torch.from_numpy(np.stack(x)).float()[:, None] / 255.0
    return to_tensor(inputs), to_tensor(targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clean-dir", default=ROOT / "data/DIV2K_valid_HR")
    p.add_argument("--film-train", type=int, default=30, help="film scans used for training (rest kept for testing)")
    p.add_argument("--out", default=ROOT / "models/restormer_finetuned.pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    clean_images = [load_gray(f, 0.5) for f in sorted(Path(args.clean_dir).glob("*.png"))]
    damaged_files = sorted((ROOT / "data/film_scans/damaged").glob("*.jpg"))[:args.film_train]
    film_pairs = [[load_gray(f, 0.5), load_gray(f.parent.parent / "restored" / f.name.replace(".jpg", "_restored.jpg"), 0.5)]
                  for f in damaged_files]
    print(f"{len(clean_images)} clean images, {len(film_pairs)} film pairs, device {device}")

    model = load_model(ROOT / "models/gaussian_gray_denoising_blind.pth").to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    start = time.time()
    for it in range(1, args.iters + 1):
        x, y = make_batch(clean_images, film_pairs, args.batch, args.patch, rng)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, enabled=device == "cuda"):
            loss = torch.nn.functional.l1_loss(model(x), y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if it % 100 == 0 or it == args.iters:
            print(f"iter {it}/{args.iters}  loss {loss.item():.4f}  {(time.time() - start) / it:.2f} s/iter", flush=True)
        if it % 1000 == 0 or it == args.iters:
            torch.save({"params": model.state_dict()}, args.out)


if __name__ == "__main__":
    main()
