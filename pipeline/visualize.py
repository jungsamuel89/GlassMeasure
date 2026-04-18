"""Visualization: draw measurements on the image."""

import cv2
import numpy as np
from PIL import Image


def draw_measurements(
    image: Image.Image,
    corners: np.ndarray,
    measurements: dict,
    mask: np.ndarray,
    glass_index: int = 1,
) -> np.ndarray:
    """
    Draw measurement overlay on the image.

    Returns BGR numpy array with annotations.
    """
    img = np.array(image)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Semi-transparent mask overlay
    overlay = img.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 200, 0], dtype=np.uint8) * 0.5
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # Draw polygon
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    # Draw corners
    for i, (x, y) in enumerate(corners.astype(int)):
        labels = ["TL", "TR", "BR", "BL"]
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img, labels[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    w_cm = measurements["width_cm"]
    h_cm = measurements["height_cm"]
    area = measurements["area_m2"]

    # Width label (top edge)
    mid_top = ((corners[0] + corners[1]) / 2).astype(int)
    cv2.putText(img, f"{w_cm:.1f} cm", (mid_top[0] - 40, mid_top[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Height label (left edge)
    mid_left = ((corners[0] + corners[3]) / 2).astype(int)
    cv2.putText(img, f"{h_cm:.1f} cm", (mid_left[0] - 90, mid_left[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Area label (center)
    center = corners.mean(axis=0).astype(int)
    label = f"#{glass_index}: {area:.4f} m2"
    cv2.putText(img, label, (center[0] - 60, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img
