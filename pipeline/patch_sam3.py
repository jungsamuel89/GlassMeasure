"""Patch SAM3 package for CPU compatibility.

SAM3 has hardcoded `device="cuda"` strings and module-level CUDA calls in
several files that crash on CPU-only PyTorch builds with
'Torch not compiled with CUDA enabled'. This module applies idempotent
string replacements at install/startup time so the image pipeline can run
on CPU (and MPS).
"""

import importlib.util
from pathlib import Path


def _patch_file(filepath: Path, replacements: list[tuple[str, str]]) -> int:
    """Apply (old, new) replacements to a file, skipping ones already applied.

    Returns the number of edits actually made.
    """
    if not filepath.exists():
        return 0
    content = filepath.read_text(encoding="utf-8")
    edits = 0
    for old, new in replacements:
        if old in content and new not in content:
            content = content.replace(old, new)
            edits += 1
    if edits:
        filepath.write_text(content, encoding="utf-8")
    return edits


def patch_sam3() -> None:
    """Patch hardcoded CUDA device references in SAM3 source files.

    Locates the sam3 package via importlib's spec machinery so we don't
    trigger sam3's top-level imports (which can fail on CPU-only machines
    without triton installed) just to find the install directory.
    """
    spec = importlib.util.find_spec("sam3")
    if spec is None or not spec.submodule_search_locations:
        return

    sam3_path = Path(next(iter(spec.submodule_search_locations)))

    cpu_fallback_device = 'device="cuda" if torch.cuda.is_available() else "cpu"'

    # (file, [(old, new), ...]) — each tuple applied once.
    files_to_patch: list[tuple[Path, list[tuple[str, str]]]] = [
        # Hardcoded device= kwargs in positional encoding / decoder
        (
            sam3_path / "model" / "position_encoding.py",
            [('device="cuda")', cpu_fallback_device + ")")],
        ),
        (
            sam3_path / "model" / "decoder.py",
            [('device="cuda"', cpu_fallback_device)],
        ),
        # Default arg in Sam3Processor — harmless if we override, but make
        # the library usable without an explicit device kwarg too.
        (
            sam3_path / "model" / "sam3_image_processor.py",
            [
                (
                    'def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):',
                    'def __init__(self, model, resolution=1008, device=None, confidence_threshold=0.5):\n'
                    '        if device is None:\n'
                    '            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"',
                ),
            ],
        ),
        # VL combiner default device kwargs
        (
            sam3_path / "model" / "vl_combiner.py",
            [
                ('additional_text=None, device="cuda"',
                 'additional_text=None, device=("cuda" if torch.cuda.is_available() else "cpu")'),
            ],
        ),
        # Module-level CUDA probe in multiplex base — guard with is_available.
        (
            sam3_path / "model" / "sam3_multiplex_base.py",
            [
                (
                    "if torch.cuda.get_device_properties(0).major >= 8:",
                    "if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:",
                ),
            ],
        ),
        # perflib fused addmm_act hardcodes bfloat16 casts — on CPU-only builds
        # that produces dtype-mismatch errors against the float32 model weights.
        # Keep the original input dtype on CPU; only cast to bf16 when CUDA is
        # available (where downstream autocast assumes bf16).
        (
            sam3_path / "perflib" / "fused.py",
            [
                (
                    "    self = self.to(torch.bfloat16)\n"
                    "    mat1 = mat1.to(torch.bfloat16)\n"
                    "    mat2 = mat2.to(torch.bfloat16)",
                    "    _cast_dtype = torch.bfloat16 if torch.cuda.is_available() else mat1.dtype\n"
                    "    self = self.to(_cast_dtype)\n"
                    "    mat1 = mat1.to(_cast_dtype)\n"
                    "    mat2 = mat2.to(_cast_dtype)",
                ),
            ],
        ),
        # pin_memory() is CUDA-only. On CPU-only torch builds it raises
        # "Cannot access accelerator device when none is available."
        # The box-scaling tensor is tiny — just skip the pin/non_blocking hop.
        (
            sam3_path / "model" / "geometry_encoders.py",
            [
                (
                    "scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)",
                    "scale = scale.to(device=boxes_xyxy.device) if not torch.cuda.is_available() "
                    "else scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)",
                ),
            ],
        ),
    ]

    total = 0
    for filepath, repls in files_to_patch:
        total += _patch_file(filepath, repls)

    if total:
        print(f"[OK] Patched {total} hardcoded CUDA reference(s) in SAM3 for CPU compatibility.")


if __name__ == "__main__":
    patch_sam3()
