"""GlassMeasure – local web app for glass surface measurement."""

import os
import sys
import uuid
import traceback
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Timer

import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify, send_file, session

from pipeline.segment import segment_glass
from pipeline.mask_utils import masks_to_polygon_dicts, polygon_to_binary_mask
from pipeline.measure import load_depth, load_intrinsics, calculate_area
from pipeline.visualize import draw_measurements

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory session storage
_sessions: dict[str, list[dict]] = {}
_result_images: dict[str, str] = {}

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
    try:
        sid = get_session_id()

        image_file = request.files.get("image")
        depth_file = request.files.get("depth")
        intrinsics_file = request.files.get("intrinsics")

        if not all([image_file, depth_file, intrinsics_file]):
            return jsonify({"error": "Please upload all 3 files."}), 400

        run_id = str(uuid.uuid4())[:8]
        run_dir = UPLOAD_DIR / run_id
        run_dir.mkdir(exist_ok=True)

        img_path = str(run_dir / "image.jpg")
        depth_path = str(run_dir / "depth.png")
        intrinsics_path = str(run_dir / "intrinsics.json")

        image_file.save(img_path)
        depth_file.save(depth_path)
        intrinsics_file.save(intrinsics_path)

        # Load image with EXIF rotation
        pil_img = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
        image_np = np.array(pil_img)
        img_h, img_w = image_np.shape[:2]

        # Step 1: Segment
        print(f"[{run_id}] Step 1: Segmenting ...")
        mask_results = segment_glass(pil_img)

        if not mask_results:
            return jsonify({"error": "No glass surfaces detected."})

        masks = [r["mask"] for r in mask_results]
        scores = [r["score"] for r in mask_results]

        # Step 2: Masks -> Polygons
        print(f"[{run_id}] Step 2: Mask -> Polygon ...")
        polygon_dicts = masks_to_polygon_dicts(masks)

        # Step 3: Load depth + intrinsics
        print(f"[{run_id}] Step 3: Loading depth + intrinsics ...")
        depth = load_depth(depth_path, img_w, img_h)
        fx, fy, cx, cy = load_intrinsics(intrinsics_path)

        # Step 4: Area calculation + visualization
        print(f"[{run_id}] Step 4: Measuring ...")
        results = []
        img_vis = image_np.copy()

        for i, poly in enumerate(polygon_dicts):
            if not poly["points"]:
                continue

            # Convert polygon back to binary mask for depth sampling
            binary_mask = polygon_to_binary_mask(poly, img_w, img_h)

            try:
                result = calculate_area(poly["points"], binary_mask, depth,
                                        fx, fy, cx, cy)
            except Exception as e:
                print(f"  [!] Measurement failed for glass #{i+1}: {e}")
                continue

            sides = result.get("sides", {})
            width_cm = result.get("width_cm")
            height_cm = result.get("height_cm")
            area_m2 = result.get("area_m2", 0)

            if area_m2 <= 0:
                continue

            # Visualize on image
            img_vis = draw_measurements(
                img_vis, sides, result,
                binary_mask, glass_index=i + 1,
            )

            row = {
                "glass_nr": i + 1,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "area_m2": area_m2,
                "score": round(scores[i] if i < len(scores) else 0, 3),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            results.append(row)

        if not results:
            return jsonify({"error": "Measurement failed – no valid polygons extracted."})

        # Save result image
        result_path = str(run_dir / "result.jpg")
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_path, img_bgr)
        _result_images[run_id] = result_path

        # Append to session table
        if sid not in _sessions:
            _sessions[sid] = []
        scan_name = image_file.filename or "scan"
        for r in results:
            r["scan"] = scan_name
            r["result_id"] = run_id
            _sessions[sid].append(r)

        return jsonify({
            "results": results,
            "result_id": run_id,
            "total_glasses": len(results),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500


@app.route("/result/<result_id>")
def get_result_image(result_id):
    path = _result_images.get(result_id)
    if not path or not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/history")
def history():
    sid = get_session_id()
    rows = _sessions.get(sid, [])
    return jsonify(rows)


@app.route("/clear", methods=["POST"])
def clear_history():
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
    print("  Web interface: http://127.0.0.1:5000")
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
