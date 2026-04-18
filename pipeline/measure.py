"""3D measurement of glass surfaces using depth map + camera intrinsics.

Adapted from 04_pipeline/calculate_glass_area.py in BachelorThesis_GlassAssessment.
"""

import json
import numpy as np
from PIL import Image


# ── Load depth map ───────────────────────────────────────────────────────────

def load_depth(depth_path: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Load 16-bit depth map PNG (values in mm) and convert to meters.
    Rotates 90° CW to match EXIF-corrected RGB orientation.
    Scales to match image dimensions if needed.
    """
    depth_img = Image.open(depth_path)
    depth_arr = np.array(depth_img, dtype=np.float32)

    # Rotate 90° clockwise to match RGB orientation
    depth_arr = np.rot90(depth_arr, k=-1)

    # 16-bit values are in millimeters -> meters
    depth_meters = depth_arr / 1000.0

    # Scale to image size if needed
    rotated_w, rotated_h = depth_arr.shape[1], depth_arr.shape[0]
    if (rotated_w, rotated_h) != (img_w, img_h):
        depth_pil = Image.fromarray(depth_arr.astype(np.uint16))
        depth_pil = depth_pil.resize((img_w, img_h), Image.BILINEAR)
        depth_meters = np.array(depth_pil, dtype=np.float32) / 1000.0

    return depth_meters


# ── Load camera intrinsics ───────────────────────────────────────────────────

def load_intrinsics(frame_json_path: str):
    """
    Load fx, fy, cx, cy from the 3D Scanner App JSON.
    Format: {"intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1]}
    """
    with open(frame_json_path) as f:
        meta = json.load(f)
    intr = meta["intrinsics"]
    fx, fy = intr[0], intr[4]
    cx, cy = intr[2], intr[5]
    return fx, fy, cx, cy


# ── Backprojection ───────────────────────────────────────────────────────────

def backproject(u: float, v: float, z: float,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Backproject a pixel (u, v) at depth z into 3D camera coordinates."""
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.array([X, Y, z])


# ── Corner ordering ─────────────────────────────────────────────────────────

def order_corners(polygon_pts: list) -> dict:
    """
    Order polygon corners as TL, TR, BR, BL using sum/difference heuristic.
    """
    pts = np.array(polygon_pts, dtype=np.float64)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return {
        "TL": pts[np.argmin(s)].copy(),
        "TR": pts[np.argmax(d)].copy(),
        "BR": pts[np.argmax(s)].copy(),
        "BL": pts[np.argmin(d)].copy(),
    }


# ── Depth sampling ──────────────────────────────────────────────────────────

def _corner_sign(corner_name):
    """Direction signs (du, dv) for sampling away from the glass interior."""
    return {
        "TL": (-1, -1),
        "TR": (+1, -1),
        "BR": (+1, +1),
        "BL": (-1, +1),
    }[corner_name]


def _sample_depth_near(u, v, depth, radius=8):
    """Fallback: median depth in a neighborhood."""
    h, w = depth.shape
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[patch > 0.1]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def sample_frame_depth(corners, depth, mask, patch_size=20):
    """
    Sample frame depth at each corner: 20×20 px square placed away from glass center.
    Only non-glass pixels are used. Returns minimum (nearest/frame surface) depth.
    """
    img_h, img_w = depth.shape
    result = {}

    for corner_name in ["TL", "TR", "BR", "BL"]:
        cu, cv = corners[corner_name].astype(float)
        cu_i, cv_i = int(round(cu)), int(round(cv))
        du_sign, dv_sign = _corner_sign(corner_name)

        # Square position: away from the glass center
        u0 = cu_i if du_sign > 0 else cu_i - patch_size
        v0 = cv_i if dv_sign > 0 else cv_i - patch_size

        # Clip to image boundaries
        u0 = max(u0, 0)
        v0 = max(v0, 0)
        u1 = min(u0 + patch_size, img_w)
        v1 = min(v0 + patch_size, img_h)

        patch_depth = depth[v0:v1, u0:u1].copy()
        patch_mask = mask[v0:v1, u0:u1]
        patch_depth[patch_mask] = 0  # exclude glass pixels

        z_valid = patch_depth[patch_depth > 0.1]

        if len(z_valid) == 0:
            result[corner_name] = _sample_depth_near(cu_i, cv_i, depth, radius=15)
        else:
            result[corner_name] = float(np.min(z_valid))

    return result


# ── Side length calculation ─────────────────────────────────────────────────

def calculate_side_lengths(polygon_pts, mask, depth, fx, fy, cx, cy):
    """
    Calculate 4 side lengths of the window in meters via 3D backprojection.
    """
    corners = order_corners(polygon_pts)
    depth_at = sample_frame_depth(corners, depth, mask, patch_size=20)

    pts_3d = {}
    for name, corner_val in corners.items():
        u, v = corner_val
        z = depth_at.get(name, 0.0)
        if z > 0.1:
            pts_3d[name] = backproject(u, v, z, fx, fy, cx, cy)
        else:
            pts_3d[name] = None

    def side_len(A, B):
        if A is None or B is None:
            return None
        return float(np.linalg.norm(A - B))

    w_top = side_len(pts_3d.get("TL"), pts_3d.get("TR"))
    w_bottom = side_len(pts_3d.get("BL"), pts_3d.get("BR"))
    h_left = side_len(pts_3d.get("TL"), pts_3d.get("BL"))
    h_right = side_len(pts_3d.get("TR"), pts_3d.get("BR"))

    def safe_mean(*vals):
        v = [x for x in vals if x is not None]
        return round(float(np.mean(v)), 4) if v else None

    width_m = safe_mean(w_top, w_bottom)
    height_m = safe_mean(h_left, h_right)

    return {
        "corners_px": {k: v.tolist() for k, v in corners.items()},
        "corners_3d_m": {k: v.tolist() if v is not None else None
                         for k, v in pts_3d.items()},
        "depth_at_corners_m": {k: round(z, 4) for k, z in depth_at.items()},
        "width_top_m": round(w_top, 4) if w_top else None,
        "width_bottom_m": round(w_bottom, 4) if w_bottom else None,
        "height_left_m": round(h_left, 4) if h_left else None,
        "height_right_m": round(h_right, 4) if h_right else None,
        "width_m": width_m,
        "height_m": height_m,
    }


# ── Main area calculation ───────────────────────────────────────────────────

def calculate_area(polygon_pts, mask, depth, fx, fy, cx, cy):
    """
    Calculate glass area via 3D side lengths.
    Returns dict with area_m2, width/height, and sides sub-dict.
    """
    # Depth statistics within mask
    v_coords, u_coords = np.where(mask)
    z_values = depth[v_coords, u_coords]
    valid = z_values > 0.1
    z_valid = z_values[valid]

    if len(z_valid) == 0:
        return {
            "area_m2": 0, "mean_depth_m": 0,
            "pixel_count": 0, "valid_pixels": 0,
            "sides": {},
        }

    mean_depth = float(z_valid.mean())

    sides = calculate_side_lengths(polygon_pts, mask, depth, fx, fy, cx, cy)
    width_m = sides.get("width_m")
    height_m = sides.get("height_m")

    total_area = (width_m * height_m) if (width_m and height_m) else 0.0

    return {
        "area_m2": round(total_area, 4),
        "area_cm2": round(total_area * 10000, 1),
        "mean_depth_m": round(mean_depth, 3),
        "width_m": width_m,
        "height_m": height_m,
        "width_cm": round(width_m * 100, 1) if width_m else None,
        "height_cm": round(height_m * 100, 1) if height_m else None,
        "pixel_count": len(z_values),
        "valid_pixels": len(z_valid),
        "sides": sides,
    }
