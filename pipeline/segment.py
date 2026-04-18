"""SAM3 segmentation with fine-tuned Exp4 weights."""

import sys
import types

# Stub out triton before SAM3 tries to import it (not available on Windows/CPU)
if "triton" not in sys.modules:
    # Create a permissive stub that returns itself for any attribute access
    class _TritonStub(types.ModuleType):
        def __getattr__(self, name):
            return _TritonStub(name)
        def __call__(self, *args, **kwargs):
            def wrapper(fn):
                return fn
            return wrapper

    _triton = _TritonStub("triton")
    for submod in ["triton", "triton.jit", "triton.language", "triton.language.core"]:
        sys.modules[submod] = _TritonStub(submod)

import torch
import numpy as np
from PIL import Image

from .download_model import get_weights_path

# Global model cache
_model = None
_processor = None
_device = None

PROMPT = "measurement glass area(s)"
CONFIDENCE_THRESHOLD = 0.3


def _load_model():
    """Load SAM3 model with fine-tuned exp4 weights (cached)."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading SAM3 model on {_device} ...")

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Load base SAM3 from HuggingFace
    _model = build_sam3_image_model(load_from_HF=True)

    # Load fine-tuned weights (Exp4)
    weights_path = get_weights_path()
    checkpoint = torch.load(str(weights_path), map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint["model"], strict=False)
    _model = _model.to(_device).eval()

    _processor = Sam3Processor(_model, resolution=1008, device=_device)
    _processor.set_confidence_threshold(CONFIDENCE_THRESHOLD)

    print("[OK] SAM3 model loaded successfully.")
    return _model, _processor, _device


def segment_glass(image: Image.Image) -> list[dict]:
    """
    Segment glass surfaces in an image.

    Returns list of dicts with keys: mask (np.ndarray H×W bool), score (float)
    """
    model, processor, device = _load_model()

    # Use autocast for efficiency (works on both CPU and CUDA)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.amp.autocast(autocast_device, dtype=dtype):
        state = processor.set_image(image)
        output = processor.set_text_prompt(PROMPT, state)

    raw_masks = output.get("masks", [])
    raw_scores = output.get("scores", [])
    raw_masks = list(raw_masks) if raw_masks is not None and len(raw_masks) > 0 else []
    raw_scores = list(raw_scores) if raw_scores is not None and len(raw_scores) > 0 else []

    results = []
    for i, m in enumerate(raw_masks):
        # Convert to numpy boolean mask
        if hasattr(m, "cpu"):
            m_np = m.cpu().numpy().squeeze()
        else:
            m_np = np.array(m).squeeze()

        mask_bool = m_np > 0.5
        score = float(raw_scores[i]) if i < len(raw_scores) else 0.0

        results.append({"mask": mask_bool, "score": score})

    return results
