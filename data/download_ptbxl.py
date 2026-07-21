"""
Downloads PTB-XL (v1.0.3) from PhysioNet and unzips it.

Run this once. It's ~1.7GB zipped, ~3GB unzipped — will take a while
depending on your connection. Safe to re-run if it fails partway;
it skips the download if the zip is already there.
"""

import os
import urllib.request
import zipfile

PTBXL_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
OUTPUT_DIR = "./ptbxl_data"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    print(f"\rDownloading... {percent:.1f}%", end="")


def download_ptbxl(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "ptbxl.zip")

    if os.path.exists(zip_path):
        print(f"Zip already exists at {zip_path}, skipping download.")
    else:
        print(f"Downloading PTB-XL from {PTBXL_URL} ...")
        urllib.request.urlretrieve(PTBXL_URL, zip_path, reporthook=_progress)
        print("\nDownload complete.")

    print("Extracting (this also takes a few minutes)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"Done. Data extracted to {output_dir}")


if __name__ == "__main__":
    download_ptbxl()