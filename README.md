# GlassMeasure

Local web tool for measuring glass surfaces using fine-tuned SAM3 + LiDAR depth maps.

## Quick Start

> **Python 3.10–3.12 recommended** (tested with 3.10.11). Python 3.14 is **not supported** — several pinned packages do not yet ship wheels for it and will fail to build.

```bash
# 1. Clone
git clone https://github.com/jungsamuel89/GlassMeasure.git
cd GlassMeasure

# 2. Create environment (Python 3.10–3.12)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3a. CPU-only laptops: install the pinned PyTorch CPU wheels FIRST
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu

# 3b. Install pinned dependencies (exact versions from requirements.txt)
pip install -r requirements.txt

# 3c. Install the GlassMeasure package itself (without re-resolving deps)
pip install -e . --no-deps

# 4. Set HuggingFace token (for model download)
set HF_TOKEN=your_token_here          # Windows
# export HF_TOKEN=your_token_here     # macOS/Linux

# 5. Run
samu
```

The web interface opens at `http://127.0.0.1:5000`.

## Usage

1. Upload 3 files from a 3D Scanner App LiDAR scan:
   - **RGB Image** (.jpg) – the photo
   - **Depth Map** (.png) – 16-bit depth in mm
   - **Intrinsics** (.json) – camera parameters (`{"intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1]}`)

2. Click **Messen** – the pipeline runs:
   - SAM3 segments glass surfaces (prompt: "measurement glass area(s)")
   - Suzuki-Abe contour detection + Douglas-Peucker → 4-corner polygons
   - Depth sampling at frame edges (20×20 px patches)
   - 3D backprojection → real-world dimensions

3. Results appear as annotated image + measurements table.

4. **CSV Export** downloads all session measurements.

## Model

Uses [SAM3](https://github.com/facebookresearch/sam3) fine-tuned on glass surfaces (Experiment 4). Weights (~5 GB) are downloaded automatically on first run from [HuggingFace](https://huggingface.co/jungsamu89/glass-sam3-finetuned).

## Requirements

- Python 3.10–3.12 (tested with 3.10.11; Python 3.14 is not supported)
- ~8 GB RAM (CPU inference)
- ~6 GB disk for model weights (cached in `~/.cache/glassmeasure/`)
