"""GlassMeasure – local web app for glass surface measurement."""

import os
import sys
import uuid
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, session

from pipeline.segment import segment_glass
from pipeline.mask_utils import mask_to_polygon, order_corners
from pipeline.measure import measure_glass
from pipeline.visualize import draw_measurements

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory session storage (per-process, resets on restart)
_sessions: dict[str, list[dict]] = {}
_result_images: dict[str, str] = {}  # result_id -> file path

UPLOAD_DIR = Path(tempfile.gettempdir()) / "glassmeasure_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


@app.route("/")
def index():
    sid = get_session_id()
    rows = _sessions.get(sid, [])
    return render_template("index.html", rows=rows)


@app.route("/measure", methods=["POST"])
def measure():
    """Run the full pipeline on uploaded files."""
    sid = get_session_id()

    # Save uploaded files
    image_file = request.files.get("image")
    depth_file = request.files.get("depth")
    intrinsics_file = request.files.get("intrinsics")

    if not all([image_file, depth_file, intrinsics_file]):
        return jsonify({"error": "Bitte alle 3 Dateien hochladen."}), 400

    run_id = str(uuid.uuid4())[:8]
    run_dir = UPLOAD_DIR / run_id
    run_dir.mkdir(exist_ok=True)

    img_path = str(run_dir / "image.jpg")
    depth_path = str(run_dir / "depth.png")
    intrinsics_path = str(run_dir / "intrinsics.json")

    image_file.save(img_path)
    depth_file.save(depth_path)
    intrinsics_file.save(intrinsics_path)

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Step 1: Segment
    print(f"[{run_id}] Segmenting ...")
    masks = segment_glass(image)

    if not masks:
        return jsonify({"error": "Keine Glasflächen erkannt."}), 200

    # Step 2-4: For each mask → polygon → measure → visualize
    results = []
    img_vis = np.array(image)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    for i, mask_data in enumerate(masks):
        mask = mask_data["mask"]
        score = mask_data["score"]

        # Polygon
        polygon = mask_to_polygon(mask)
        if polygon is None:
            continue

        corners = order_corners(polygon)

        # Measure
        try:
            meas = measure_glass(corners, mask, depth_path, intrinsics_path)
        except Exception as e:
            print(f"  [!] Measurement failed for glass #{i+1}: {e}")
            continue

        # Visualize
        img_vis_pil = Image.fromarray(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        img_vis = draw_measurements(img_vis_pil, corners, meas, mask, glass_index=i + 1)

        result = {
            "glass_nr": i + 1,
            "width_cm": meas["width_cm"],
            "height_cm": meas["height_cm"],
            "area_m2": meas["area_m2"],
            "score": round(score, 3),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        results.append(result)

    if not results:
        return jsonify({"error": "Polygone konnten nicht extrahiert werden."}), 200

    # Save result image
    result_path = str(run_dir / "result.jpg")
    cv2.imwrite(result_path, img_vis)
    result_id = run_id
    _result_images[result_id] = result_path

    # Append to session table
    if sid not in _sessions:
        _sessions[sid] = []

    scan_name = image_file.filename or "scan"
    for r in results:
        r["scan"] = scan_name
        r["result_id"] = result_id
        _sessions[sid].append(r)

    return jsonify({
        "results": results,
        "result_id": result_id,
        "total_glasses": len(results),
    })


@app.route("/result/<result_id>")
def get_result_image(result_id):
    """Serve a result image."""
    path = _result_images.get(result_id)
    if not path or not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/history")
def history():
    """Return session measurement history."""
    sid = get_session_id()
    rows = _sessions.get(sid, [])
    return jsonify(rows)


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear session history."""
    sid = get_session_id()
    _sessions[sid] = []
    return jsonify({"ok": True})


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


def main():
    print()
    print("=" * 55)
    print("  GlassMeasure – Glass Surface Measurement Tool")
    print("=" * 55)
    print()
    print("  Starting web interface at http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop.")
    print()

    # Pre-load model on startup
    print("[*] Pre-loading SAM3 model (first time may download ~5 GB) ...")
    try:
        from pipeline.segment import _load_model
        _load_model()
    except Exception as e:
        print(f"[!] Model pre-load failed: {e}")
        print("    The model will be loaded on first measurement request.")

    Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
