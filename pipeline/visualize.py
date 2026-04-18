"""Visualization: draw measurements on the image."""

import cv2
import numpy as np


CORNER_COLORS = {
    "TL": (0, 215, 255),  # gold BGR
    "TR": (209, 206, 0),  # dark cyan BGR
    "BR": (50, 205, 50),  # lime BGR
    "BL": (0, 140, 255),  # dark orange BGR
}


def draw_measurements(
    image_np: np.ndarray,
    sides: dict,
    result: dict,
    mask: np.ndarray,
    glass_index: int = 1,
) -> np.ndarray:
    """
    Draw measurement overlay on the image (RGB numpy array).
    Returns modified RGB numpy array.
    """
    img = image_np.copy()

    # Semi-transparent mask overlay
    overlay = img.copy()
    green = np.array([0, 200, 100], dtype=np.uint8)
    overlay[mask] = (overlay[mask] * 0.5 + green * 0.5).astype(np.uint8)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    corners_px = sides.get("corners_px", {})
    if not corners_px:
        return img

    # Draw polygon edges with side lengths
    edge_pairs = [
        ("TL", "TR", sides.get("width_top_m")),
        ("TR", "BR", sides.get("height_right_m")),
        ("BR", "BL", sides.get("width_bottom_m")),
        ("BL", "TL", sides.get("height_left_m")),
    ]
    for p1, p2, length in edge_pairs:
        if p1 not in corners_px or p2 not in corners_px:
            continue
        x1, y1 = int(corners_px[p1][0]), int(corners_px[p1][1])
        x2, y2 = int(corners_px[p2][0]), int(corners_px[p2][1])
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if length is not None:
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            label = f"{length:.3f} m"
            cv2.putText(img, label, (mx - 40, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw corner points
    for name, (u, v) in corners_px.items():
        u_i, v_i = int(u), int(v)
        z = sides.get("depth_at_corners_m", {}).get(name)
        cv2.circle(img, (u_i, v_i), 8, (0, 0, 255), -1)
        corner_label = f"{name}"
        if z:
            corner_label += f" Z={z:.2f}m"
        cv2.putText(img, corner_label, (u_i + 12, v_i - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Area annotation in center
    all_u = [corners_px[k][0] for k in corners_px]
    all_v = [corners_px[k][1] for k in corners_px]
    cx = int(np.mean(all_u))
    cy = int(np.mean(all_v))

    area = result.get("area_m2", 0)
    w_m = result.get("width_m")
    h_m = result.get("height_m")

    line1 = f"#{glass_index}: {area:.4f} m2"
    cv2.putText(img, line1, (cx - 80, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if w_m and h_m:
        line2 = f"({w_m:.3f} x {h_m:.3f} m)"
        cv2.putText(img, line2, (cx - 80, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return img
