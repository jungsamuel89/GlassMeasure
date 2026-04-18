"""Mask-to-polygon conversion using contour detection."""

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.02) -> np.ndarray | None:
    """
    Convert binary mask to a simplified polygon (ideally 4 corners).

    Args:
        mask: H×W boolean array
        epsilon_ratio: Douglas-Peucker simplification ratio

    Returns:
        Nx2 array of (x, y) corner coordinates, or None if no valid contour
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)

    points = approx.reshape(-1, 2)

    if len(points) < 4:
        # Fall back to bounding rect corners
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        points = box.astype(int)

    return points


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Order points as: top-left, top-right, bottom-right, bottom-left.

    Uses sum (x+y) for TL/BR and difference (y-x) for TR/BL.
    """
    if len(points) > 4:
        # Use convex hull and pick 4 extreme points
        hull = cv2.convexHull(points)
        hull = hull.reshape(-1, 2)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        points = box.astype(int)

    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left

    return ordered
