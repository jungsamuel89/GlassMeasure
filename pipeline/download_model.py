"""Download SAM3 fine-tuned weights from HuggingFace."""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download


MODEL_REPO = "jungsamu89/glass-sam3-finetuned"
CHECKPOINT_NAME = "exp4/best.pt"


def get_weights_path() -> Path:
    """Return local path to the exp4 checkpoint, downloading if needed."""
    cache_dir = Path.home() / ".cache" / "glassmeasure"
    local_path = cache_dir / "exp4_best.pt"

    if local_path.exists():
        print(f"[OK] Weights found at {local_path}")
        return local_path

    print("[*] Downloading fine-tuned weights from HuggingFace ...")
    print("    This only happens once (~5 GB). Please wait.")

    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CHECKPOINT_NAME,
        cache_dir=str(cache_dir / "hf_cache"),
        token=os.environ.get("HF_TOKEN"),
    )

    # Symlink or copy to clean path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        import shutil
        shutil.copy2(downloaded, str(local_path))

    print(f"[OK] Weights saved to {local_path}")
    return local_path
