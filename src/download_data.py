"""Downloads the datasets into data/ (skips files that are already there).

Usage:  python src/download_data.py                  # everything
        python src/download_data.py bsd68 div2k      # only some
"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def download(url, dest, size=None):
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    if size is not None and dest.stat().st_size != size:
        raise RuntimeError(f"incomplete download: {dest}")


def github_folder(repo, folder, dest):
    files = get_json(f"https://api.github.com/repos/{repo}/contents/{folder}")
    for f in files:
        if f["type"] == "file":
            download(f["download_url"], dest / f["name"], f["size"])
    print(f"{dest.relative_to(DATA)}: {len(list(dest.iterdir()))} files")


def bsd68():
    # 68 clean grayscale test images
    github_folder("cszn/DnCNN", "testsets/BSD68", DATA / "BSD68")


def old_photos():
    # real old photos, no clean original (MIT license)
    repo = "microsoft/Bringing-Old-Photos-Back-to-Life"
    github_folder(repo, "test_images/old_w_scratch", DATA / "old_photos/old_w_scratch")
    github_folder(repo, "test_images/old", DATA / "old_photos/old")


def film_scans():
    # 44 real damaged film scans + expert restorations (Ivanova et al. 2023, CC BY 4.0)
    files = get_json("https://api.figshare.com/v2/articles/21803304/files?page_size=500")
    for f in files:
        sub = "restored" if f["name"].endswith("_restored.jpg") else "damaged"
        download(f["download_url"], DATA / "film_scans" / sub / f["name"], f["size"])
    print(f"film_scans: {len(files)} files")


def div2k_zip(name, count, size):
    # clean high-resolution images for training (DIV2K, academic use)
    folder = DATA / name
    if folder.exists() and len(list(folder.glob("*.png"))) == count:
        print(f"{name}: {count} files")
        return
    zip_path = DATA / f"{name}.zip"
    download(f"https://data.vision.ee.ethz.ch/cvl/DIV2K/{name}.zip", zip_path, size)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
    zip_path.unlink()
    print(f"{name}: {len(list(folder.glob('*.png')))} files")


def div2k():
    div2k_zip("DIV2K_valid_HR", 100, 448993893)  # validation set, 100 images


def div2k_train():
    div2k_zip("DIV2K_train_HR", 800, 3530603713)  # training set, 800 images (~3.5 GB)


DATASETS = {"bsd68": bsd68, "old_photos": old_photos, "film_scans": film_scans, "div2k": div2k, "div2k_train": div2k_train}

if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        DATASETS[name]()
