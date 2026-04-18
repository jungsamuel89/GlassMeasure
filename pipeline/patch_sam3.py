"""Patch SAM3 package for CPU compatibility.

SAM3 has hardcoded device="cuda" in two files that must be changed
to support CPU-only inference on Windows/Mac laptops.
"""

import importlib
import site


def patch_sam3():
    """Patch hardcoded CUDA device references in SAM3 source files."""
    try:
        import sam3
    except ImportError:
        return

    sam3_dir = sam3.__path__[0] if hasattr(sam3, "__path__") else None
    if not sam3_dir:
        return

    from pathlib import Path
    sam3_path = Path(sam3_dir)

    patches = [
        (
            sam3_path / "model" / "position_encoding.py",
            'device="cuda")',
            'device="cuda" if torch.cuda.is_available() else "cpu")',
        ),
        (
            sam3_path / "model" / "decoder.py",
            'device="cuda"',
            'device="cuda" if torch.cuda.is_available() else "cpu"',
        ),
    ]

    patched = 0
    for filepath, old, new in patches:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        if old in content and new not in content:
            content = content.replace(old, new)
            filepath.write_text(content, encoding="utf-8")
            patched += 1

    if patched:
        print(f"[OK] Patched {patched} hardcoded CUDA references in SAM3 for CPU compatibility.")


if __name__ == "__main__":
    patch_sam3()
