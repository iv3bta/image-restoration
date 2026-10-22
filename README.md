# Image Restoration

Restoration of damaged images (noise, scratches, blur, missing regions) using image processing techniques.

## Project Description

Old and damaged photographs often show several defects at once: noise, scratches, blur and missing regions. This project restores such damage in a single grayscale image: the input is the damaged image and the output is the restored image, while colour images, colorization and video are out of scope. The pipeline applies classical image processing methods in a fixed order (denoising with median / bilateral / NL-means filters, damage-mask detection, OpenCV inpainting with TELEA / Navier-Stokes, Wiener deblurring) and is evaluated on lab images with synthetic damage, where the clean original is known. Success means a higher PSNR than the damaged input; on 4 test images the current baselines raise mean PSNR by +7.28 dB for Gaussian noise (NL-means), +17.51 dB for salt & pepper (median), +2.35 dB for motion blur (Wiener), +12.22 dB for missing regions (TELEA) and +9.04 dB for scratches with the detected mask, versus +22.62 dB with the true mask, which makes scratch detection the main open problem.

## State of the Art

Analysis of similar articles / applications / projects from the last 5 years.

| No. | Author(s) / Year | Title of the article / project | Application / Domain | Technologies used | Methodology / Approach | Results | Limitations | Additional comments |
|-----|------------------|--------------------------------|----------------------|-------------------|------------------------|---------|-------------|---------------------|
| 1 | [Zamir et al., 2022](https://arxiv.org/abs/2111.09881) | Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022) | General image restoration: Gaussian and real denoising, motion and defocus deblurring, deraining | PyTorch 1.8.1, BasicSR (official repo); GPU not stated in the paper | | Gaussian grayscale denoising σ=25, BSD68: 29.51 dB (single model, Table 4)<br>Real denoising, SIDD: 40.02 dB / SSIM 0.960 (Table 6)<br>Motion deblurring, GoPro: 32.92 dB / SSIM 0.961 (Table 2) | | |
| 2 | [Potlapalli et al., 2023](https://arxiv.org/abs/2306.13090) | PromptIR: Prompting for All-in-One Blind Image Restoration (NeurIPS 2023) | All-in-one restoration: one model for denoising, deraining and dehazing | PyTorch 1.8.1, Python 3.8, CUDA 11.6 (official repo); GPU not stated in the paper | | All-in-one model, BSD68 σ=25: 31.31 dB / SSIM 0.888 (Table 1)<br>Single-task model, BSD68 σ=25: 31.71 dB / SSIM 0.897 (Table 4) | | |
| 3 | [Suvorov et al., 2022](https://arxiv.org/abs/2109.07161) | Resolution-robust Large Mask Inpainting with Fourier Convolutions (WACV 2022) | Image inpainting: filling large missing regions given a mask | PyTorch, PyTorch-Lightning, Hydra (Sec. 3); Big LaMa trained on 8× NVIDIA V100 for ~240 h (Sec. 3.4) | | Places 512×512, wide masks: FID 2.21, LPIPS 0.135 (Table 1)<br>CelebA-HQ 256×256, wide masks: FID 6.96, LPIPS 0.098 (Table 1) | | |
| 4 | [Kuo et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9656350/) | Learning-Based Image Damage Area Detection for Old Photo Recovery (Sensors 22(21), 8580) | Old photo restoration: detecting damaged areas (scratches, cracks) before inpainting | NVIDIA 2070 8 GB GPU, Intel i5-2400 CPU; 170 old photos labelled in GIMP; framework not stated | | Damage detection on 18 real test photos: recall 0.911, precision 0.847, F1 0.873 (Table 2)<br>No PSNR/SSIM reported for the restored photos | | |
| 5 | [Ivanova et al., 2023](https://arxiv.org/abs/2302.10004) | Simulating analogue film damage to analyse and improve artefact restoration on high-resolution scans (Computer Graphics Forum, Eurographics 2023) | Film scan restoration: removing dust and scratches from 4K scans of analogue film | Damage simulator in Python with OpenCV, NumPy, SciPy, scikit-image (official repo); Tesla P100 GPU (Sec. 5.3) | | 44 real 4K scans vs. expert restorations, LaMa with the authors' masks: 39.6 dB / SSIM 0.991 / LPIPS 0.042 (Table 4)<br>Damage segmentation: IoU 0.256, F1 0.408 (Table 3) | | |

- **Application / Domain** – domain in which the described solution is applied (e.g., face recognition, augmented reality).
- **Technologies used** – technologies / hardware used in the project (e.g., OpenCV, TensorFlow, ZED stereo cameras).
- **Methodology / Approach** – description of the method used in the project.
- **Results** – main results obtained (metrics used for evaluation).
- **Limitations** – limitations encountered in research or implementation.
- **Additional comments** – observations on relevance to the current project; suggestions for improving the methodology.

## Proposed Architecture

![Architecture](docs/architecture.png)

### Components

- **Damaged image** – input: a real damaged photo, or a clean image damaged by the generator.
- **1. Denoising** – removes noise first (median filter for salt & pepper; bilateral / NL-means for Gaussian noise).
- **2. Damage mask detection** – builds the binary mask of damaged pixels (scratches via white top-hat + threshold; missing regions as missing pixels).
- **3. Inpainting** – fills the masked pixels from their surroundings (OpenCV TELEA or Navier-Stokes).
- **4. Deblurring** – reverses blur with Wiener deconvolution, using a known or estimated blur kernel (PSF).
- **Restored image** – output of the pipeline.
- **5. Evaluation** – PSNR and SSIM against the clean original when one exists; no-reference metrics (NIQE / BRISQUE) and visual inspection for real photos.
- **Synthetic damage generator** – adds noise, blur, scratches and holes to clean images with a fixed seed, so results can be measured and reproduced.