"""Central device selection for GlassMeasure.

Import `get_device()` wherever SAM3 is loaded or inference runs.
CUDA -> MPS -> CPU fallback. Autocast is only enabled on CUDA;
CPU and MPS use float32 and plain inference_mode/no_grad.
"""

from contextlib import contextmanager

import torch

_device_cache: torch.device | None = None


def get_device() -> torch.device:
    """Return the best available torch device: CUDA > MPS > CPU."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        _device_cache = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        _device_cache = torch.device("mps")
    else:
        _device_cache = torch.device("cpu")
    return _device_cache


def get_map_location() -> str:
    """Safe `map_location` for `torch.load` — always load onto CPU first,
    then the caller moves the model to the active device. Avoids
    'Torch not compiled with CUDA enabled' on CPU-only builds when the
    checkpoint was pickled with CUDA tensor metadata.
    """
    return "cpu"


@contextmanager
def inference_context():
    """Inference context manager.

    - Always wraps in `torch.no_grad()`.
    - On CUDA: additionally enables bfloat16 autocast.
    - On CPU / MPS: plain float32, no autocast.
    """
    device = get_device()
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield
