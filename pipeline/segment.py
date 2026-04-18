"""SAM3 segmentation with fine-tuned weights."""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration

from .download_model import get_weights_path

# Global model cache
_model = None
_processor = None
_device = None

PROMPT = "measurement glass area(s)"
CONFIDENCE_THRESHOLD = 0.5
MIN_MASK_AREA = 2500


def _load_model():
    """Load SAM3 model with fine-tuned exp4 weights (cached)."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading SAM3 model on {_device} ...")

    _processor = AutoProcessor.from_pretrained(
        "facebook/sam2.1-hiera-large",
        trust_remote_code=True,
    )
    _model = AutoModelForMaskGeneration.from_pretrained(
        "facebook/sam2.1-hiera-large",
        trust_remote_code=True,
    )

    # Load fine-tuned weights
    weights_path = get_weights_path()
    checkpoint = torch.load(str(weights_path), map_location=_device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load with strict=False to handle partial fine-tuning
    _model.load_state_dict(state_dict, strict=False)
    _model.to(_device)
    _model.eval()

    print("[OK] Model loaded successfully.")
    return _model, _processor, _device


def segment_glass(image: Image.Image) -> list[dict]:
    """
    Segment glass surfaces in an image.

    Returns list of dicts with keys: mask (np.ndarray H×W bool), score (float)
    """
    model, processor, device = _load_model()

    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process masks
    target_size = [(image.height, image.width)]
    masks_output = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"] if "original_sizes" in inputs else target_size,
        target_size,
    )

    if isinstance(masks_output, list):
        masks = masks_output[0]  # first (only) image
    else:
        masks = masks_output

    scores = outputs.iou_scores[0] if hasattr(outputs, "iou_scores") else None
    if scores is None and hasattr(outputs, "pred_scores"):
        scores = outputs.pred_scores[0]

    results = []
    if masks.dim() == 4:
        masks = masks.squeeze(0)  # remove batch dim

    for i in range(masks.shape[0]):
        mask = masks[i].cpu()
        if mask.dim() == 3:
            mask = mask[0]  # take first channel
        mask_np = mask.numpy() > 0.5

        score = float(scores[i].max()) if scores is not None else 1.0

        if score < CONFIDENCE_THRESHOLD:
            continue
        if mask_np.sum() < MIN_MASK_AREA:
            continue

        results.append({"mask": mask_np, "score": score})

    return results
