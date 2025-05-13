#!/usr/bin/env python3
"""
download_medsam2_models.py

Download all released MedSAM-2 model checkpoints from Hugging Face
into a local `checkpoints/` folder.

Requires: pip install --upgrade huggingface_hub
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

# ----------------------------------------------------------------------
REPO_ID = "wanglab/MedSAM2"
CHECKPOINT_DIR = Path("checkpoints")
MODEL_FILES = [
    "MedSAM2_2411.pt",
    "MedSAM2_US_Heart.pt",
    "MedSAM2_MRI_LiverLesion.pt",
    "MedSAM2_CTLesion.pt",
    "MedSAM2_latest.pt",
]
# ----------------------------------------------------------------------

def ensure_checkpoint_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)

def download_file(model_file: str) -> Path:
    print(f"➡️  Downloading {model_file} …")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=model_file,
        local_dir=CHECKPOINT_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✅  Saved to {local_path}")
    return Path(local_path)

def main() -> None:
    ensure_checkpoint_dir(CHECKPOINT_DIR)

    api = HfApi()
    # list_repo_files now returns List[str]
    repo_files = set(api.list_repo_files(REPO_ID))

    missing = [f for f in MODEL_FILES if f not in repo_files]
    if missing:
        print("⚠️  The following files were not found in the repo and will be skipped:")
        for f in missing:
            print("   ", f)

    for model_file in MODEL_FILES:
        if model_file in repo_files:
            download_file(model_file)

    print("\n🎉  All available MedSAM-2 checkpoints are now in", CHECKPOINT_DIR.resolve())

if __name__ == "__main__":
    main()

