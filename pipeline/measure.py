"""3D measurement of glass surfaces using depth map + camera intrinsics."""

import json
import numpy as np
from pathlib import Path


PATCH_SIZE = 20  # px for depth sampling


def load_intrinsics(intrinsics_path: str) -> dict:
    """Load camera intrinsics from 3D Scanner App JSON export."""
    with open(intrinsics_path, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if "intrinsicMatrix" in data:
        m = data["intrinsicMatrix"]
        # Flat [fx, 0, cx, 0, fy, cy, 0, 0, 1] or nested
        if isinstance(m, list) and len(m) == 9:
            fx, _, cx, _, fy, cy = m[0], m[1], m[2], m[3], m[4], m[5]
        elif isinstance(m, list) and len(m) == 3:
            fx, _, cx = m[0]
            _, fy, cy = m[1]
        else:
            fx, fy, cx, cy = m.get("fx", m[0]), m.get("fy", m[4]), m.get("cx", m[2]), m.get("cy", m[5])
    elif "fx" in data:
        fx = data["fx"]
        fy = data["fy"]
        cx = data["cx"]
        cy = data["cy"]
    else:
        raise ValueError(f"Unknown intrinsics format. Keys: {list(data.keys())}")

    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def load_depth_map(depth_path: str) -> np.ndarray:
    """
    Load 16-bit depth map PNG (values in mm) and convert to meters.
    Rotates 90deg CW to match EXIF-corrected RGB orientation.
    """
    import cv2
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth map: {depth_path}")

    # Rotate 90° clockwise to match RGB orientation
    depth = np.rot90(depth, k=-1)

    # Convert mm to meters
    depth_m = depth.astype(np.float64) / 1000.0
    return depth_m


def sample_frame_depth(
    corner: np.ndarray,
    mask: np.ndarray,
    depth_map: np.ndarray,
    center: np.ndarray,
) -> float:
    """
    Sample depth at a corner point from the window frame (not the glass).

    Places a patch away from the glass center, uses only non-glass pixels,
    and returns the minimum (nearest/frame surface) depth.
    """
    h, w = depth_map.shape
    cx, cy = int(corner[0]), int(corner[1])
    mcx, mcy = int(center[0]), int(center[1])

    # Direction away from glass center
    dx = cx - mcx
    dy = cy - mcy
    norm = max(np.sqrt(dx**2 + dy**2), 1e-6)
    dx, dy = dx / norm, dy / norm

    # Place patch center outside the glass
    offset = PATCH_SIZE
    px = int(cx + dx * offset)
    py = int(cy + dy * offset)

    # Clamp to image bounds
    x1 = max(0, px - PATCH_SIZE // 2)
    x2 = min(w, px + PATCH_SIZE // 2)
    y1 = max(0, py - PATCH_SIZE // 2)
    y2 = min(h, py + PATCH_SIZE // 2)

    patch_depth = depth_map[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]

    # Only use non-glass pixels
    valid = (~patch_mask) & (patch_depth > 0.01)
    if valid.sum() == 0:
        # Fallback: use all non-zero depth
        valid = patch_depth > 0.01
    if valid.sum() == 0:
        return float(depth_map[max(0, min(cy, h-1)), max(0, min(cx, w-1))]) or 1.0

    return float(np.min(patch_depth[valid]))


def backproject_to_3d(u: float, v: float, z: float, intrinsics: dict) -> np.ndarray:
    """Backproject pixel (u,v) at depth z to 3D point using pinhole model."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


def measure_glass(
    corners: np.ndarray,
    mask: np.ndarray,
    depth_path: str,
    intrinsics_path: str,
) -> dict:
    """
    Measure a glass surface given its 4 ordered corners (TL, TR, BR, BL),
    binary mask, depth map path, and intrinsics path.

    Returns dict with width_m, height_m, area_m2, width_cm, height_cm, corners_3d.
    """
    intrinsics = load_intrinsics(intrinsics_path)
    depth_map = load_depth_map(depth_path)

    # Resize depth to match mask if needed
    if depth_map.shape != mask.shape:
        import cv2
        depth_map = cv2.resize(depth_map, (mask.shape[1], mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # Glass center
    center = corners.mean(axis=0)

    # Sample frame depth at each corner and backproject to 3D
    points_3d = []
    for corner in corners:
        z = sample_frame_depth(corner, mask, depth_map, center)
        p3d = backproject_to_3d(float(corner[0]), float(corner[1]), z, intrinsics)
        points_3d.append(p3d)

    p3d = np.array(points_3d)  # shape (4, 3), order: TL, TR, BR, BL

    # Compute side lengths
    width_top = np.linalg.norm(p3d[1] - p3d[0])
    width_bot = np.linalg.norm(p3d[2] - p3d[3])
    height_left = np.linalg.norm(p3d[3] - p3d[0])
    height_right = np.linalg.norm(p3d[2] - p3d[1])

    width_m = (width_top + width_bot) / 2
    height_m = (height_left + height_right) / 2
    area_m2 = width_m * height_m

    return {
        "width_m": round(float(width_m), 4),
        "height_m": round(float(height_m), 4),
        "area_m2": round(float(area_m2), 4),
        "width_cm": round(float(width_m * 100), 1),
        "height_cm": round(float(height_m * 100), 1),
        "width_top_m": round(float(width_top), 4),
        "width_bottom_m": round(float(width_bot), 4),
        "height_left_m": round(float(height_left), 4),
        "height_right_m": round(float(height_right), 4),
        "corners_3d": p3d.tolist(),
    }
