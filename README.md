# GlassMeasure

Local web tool for measuring glass surfaces using fine-tuned SAM3 + LiDAR depth maps.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jungsamuel89/GlassMeasure.git
cd GlassMeasure

# 2. Create environment (Python 3.10+)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install
pip install -e .

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

- Python 3.10+
- ~8 GB RAM (CPU inference)
- ~6 GB disk for model weights (cached in `~/.cache/glassmeasure/`)
