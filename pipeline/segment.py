"""SAM3 segmentation with fine-tuned Exp4 weights."""

import sys
import types

# ---------------------------------------------------------------------------
# Triton stub – must run BEFORE torch is imported.
#
# SAM3 (and recent PyTorch builds) try to import triton at various points.
# Triton is CUDA-only and unavailable on Windows / CPU-only installs.
# We install a full import-hook + attribute-proxy so that ANY access to
# triton.* silently succeeds and returns inert stubs.
# ---------------------------------------------------------------------------

class _TritonStubModule(types.ModuleType):
    """A module that acts as a package and returns stubs for every attribute."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []          # looks like a package
        self.__file__ = f"<triton-stub>"
        self.__package__ = name
        self.__spec__ = None
        self.__loader__ = None

    def __getattr__(self, name):
        # Return a callable stub that works as decorator, value, class, etc.
        return _TritonDummy()

    def __repr__(self):
        return f"<triton-stub:{self.__name__}>"


class _TritonDummy:
    """A universal stand-in: callable, subscriptable, iterable, boolean-false."""

    def __call__(self, *a, **kw):
        # Works as decorator: @triton.jit → returns the function unchanged
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return _TritonDummy()

    def __getattr__(self, name):
        return _TritonDummy()

    def __getitem__(self, key):
        return _TritonDummy()

    def __bool__(self):
        return False

    def __repr__(self):
        return "<triton-dummy>"

    def __iter__(self):
        return iter([])

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0


class _TritonImportHook:
    """Meta-path hook: intercept every `import triton*` and return a stub."""

    def find_module(self, fullname, path=None):
        if fullname == "triton" or fullname.startswith("triton."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _TritonStubModule(fullname)
        sys.modules[fullname] = mod
        # Also register as attribute on parent so `triton.language` works
        parts = fullname.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], parts[-1], mod)
        return mod


# Install the hook immediately (before torch import)
if "triton" not in sys.modules:
    sys.meta_path.insert(0, _TritonImportHook())

# Now safe to import torch and everything else
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

    # CPU: no autocast needed; CUDA: use bfloat16
    if device.type == "cuda":
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = torch.no_grad()

    with ctx:
        state = processor.set_image(image)
        output = processor.set_text_prompt(PROMPT, state)

    raw_masks = output.get("masks", [])
    raw_scores = output.get("scores", [])
    raw_masks = list(raw_masks) if raw_masks is not None and len(raw_masks) > 0 else []
    raw_scores = list(raw_scores) if raw_scores is not None and len(raw_scores) > 0 else []

    results = []
    for i, m in enumerate(raw_masks):
        if hasattr(m, "cpu"):
            m_np = m.cpu().numpy().squeeze()
        else:
            m_np = np.array(m).squeeze()

        mask_bool = m_np > 0.5
        score = float(raw_scores[i]) if i < len(raw_scores) else 0.0

        results.append({"mask": mask_bool, "score": score})

    return results
