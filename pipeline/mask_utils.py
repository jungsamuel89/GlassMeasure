"""Mask-to-polygon conversion (Suzuki-Abe + Douglas-Peucker)."""

import cv2
import numpy as np
from PIL import Image, ImageDraw


def mask_to_polygon(mask: np.ndarray,
                    epsilon_factors=(0.01, 0.02, 0.03, 0.05, 0.08)) -> dict:
    """
    Convert a SAM mask (H, W) into a polygon dict.

    Algorithm:
      1. Suzuki-Abe contour detection (cv2.findContours)
      2. Douglas-Peucker simplification (cv2.approxPolyDP)
         -> epsilon is gradually increased until >= 4 points are found
      3. Fallback: bounding rectangle if DP does not yield 4 points

    Returns {"type": "polygon", "points": [[x, y], ...]}
    """
    binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"type": "polygon", "points": []}

    contour = max(contours, key=cv2.contourArea)
    arc_len = cv2.arcLength(contour, True)

    approx = None
    for factor in epsilon_factors:
        approx = cv2.approxPolyDP(contour, factor * arc_len, True)
        if len(approx) >= 4:
            break

    if approx is None or len(approx) < 4:
        x, y, w, h = cv2.boundingRect(contour)
        points = [[int(x), int(y)], [int(x + w), int(y)],
                  [int(x + w), int(y + h)], [int(x), int(y + h)]]
    else:
        squeezed = approx.squeeze()
        if squeezed.ndim == 1:
            squeezed = squeezed.reshape(1, 2)
        points = squeezed.tolist()

    return {"type": "polygon", "points": points}


def masks_to_polygon_dicts(masks: list) -> list[dict]:
    """Convert all SAM masks into a list of polygon dicts."""
    results = []
    for i, mask in enumerate(masks):
        poly = mask_to_polygon(mask)
        results.append(poly)
    return results


def polygon_to_binary_mask(polygon_dict: dict, img_w: int, img_h: int) -> np.ndarray:
    """Convert a polygon dict into a binary NumPy mask (bool, H x W)."""
    points = polygon_dict["points"]
    poly_img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(poly_img)
    flat_pts = [(p[0], p[1]) for p in points]
    draw.polygon(flat_pts, fill=255)
    return np.array(poly_img) > 0
