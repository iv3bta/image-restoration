"""Pretrained Restormer (grayscale blind denoising) on BSD68, same protocol as the official test script.

Usage:  python src/test_restormer.py [weights] [sigma]
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio

from restormer_arch import Restormer

ROOT = Path(__file__).resolve().parent.parent


def load_model(weights):
    model = Restormer(inp_channels=1, out_channels=1, dim=48, num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)
    model.load_state_dict(torch.load(weights, map_location="cpu")["params"])
    return model.eval()


@torch.no_grad()
def denoise(model, noisy):
    # noisy: float image in [0, 1] (may go outside because noise is not clipped)
    x = torch.from_numpy(noisy).float()[None, None]
    h, w = noisy.shape
    x = F.pad(x, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8), "reflect")  # size must be a multiple of 8
    out = model(x)[0, 0, :h, :w].clamp(0, 1).numpy()
    return np.round(out * 255).astype(np.uint8)


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else ROOT / "models/gaussian_gray_denoising_blind.pth"
    sigma = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    model = load_model(weights)
    out_dir = ROOT / f"results/restormer_bsd68_sigma{sigma}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    files = sorted((ROOT / "data/BSD68").glob("*.png"))
    for f in files:
        clean = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        np.random.seed(0)  # same noise as the official script
        noisy = clean / 255.0 + np.random.normal(0, sigma / 255.0, clean.shape)
        start = time.time()
        restored = denoise(model, noisy)
        scores.append(peak_signal_noise_ratio(clean, restored, data_range=255))
        cv2.imwrite(str(out_dir / f.name), restored)
        print(f"{f.name}: {scores[-1]:.2f} dB ({time.time() - start:.1f} s)", flush=True)

    print(f"\nBSD68 sigma={sigma}: mean PSNR {np.mean(scores):.2f} dB over {len(scores)} images")


if __name__ == "__main__":
    main()
