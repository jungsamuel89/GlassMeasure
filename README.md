# GlassMeasure

Local web tool for measuring glass surfaces from RGB images and LiDAR depth maps using a fine-tuned SAM-based segmentation pipeline.

## What It Shows

- Applied computer vision workflow for real-world measurements
- Flask-based local web interface for scan upload and CSV export
- RGB + 16-bit depth map processing
- Contour extraction, polygon fitting, depth sampling, and 3D backprojection
- Model download and local inference setup for reproducible experiments

## Quick Start

Python 3.10-3.12 is recommended. Python 3.14 is not supported because several pinned packages do not yet ship compatible wheels.

```powershell
git clone https://github.com/jungsamuel89/GlassMeasure.git
cd GlassMeasure

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps

$env:HF_TOKEN="your_token_here"
samu
```

The web interface opens at `http://127.0.0.1:5000`.

## Usage

Upload three files from a LiDAR scan:

- RGB image (`.jpg`)
- 16-bit depth map in millimeters (`.png`)
- Camera intrinsics (`.json`)

The pipeline segments glass areas, extracts contours, fits 4-corner polygons, samples depth around frame edges, backprojects points into 3D space, and exports the resulting measurements.

## Model

The project uses SAM3-style segmentation weights fine-tuned for glass surfaces. Model weights are downloaded on first run from HuggingFace when `HF_TOKEN` is configured.

## Requirements

- Python 3.10-3.12
- About 8 GB RAM for CPU inference
- About 6 GB disk space for cached model weights

## Security

Secrets are not committed. HuggingFace access is read from the `HF_TOKEN` environment variable, and model artifacts are ignored by Git.
