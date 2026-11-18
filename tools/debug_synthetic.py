"""합성 실험 그림: 가상 카메라로 본 3D 차 + 접점 방식별 맵 위치.

    python tools/debug_synthetic.py            # -> outputs/debug/synthetic_contact.png

왼쪽: 카메라 영상 (칸 선, 차, 접점 3종), 오른쪽: 탑뷰 (칸, 실제 차 바닥, 접점 3종). 위쪽이 카메라에서 먼 쪽.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import box

from parking.contact import contact_map, contact_point_img
from parking.homography import transform_points
from parking.occupancy import nearest_slot
from parking.synthetic import IMG_H, IMG_W, make_camera, project

# 점 색 (BGR): 박스 중심 빨강, 지면 접점 초록, 지면 접점+오프셋 파랑
COLORS = {"bbox_center": (0, 0, 255), "mask_bottom": (0, 200, 0), "mask_bottom+off": (255, 120, 0)}
LABELS = {"bbox_center": "bbox center (paper 2)", "mask_bottom": "ground contact",
          "mask_bottom+off": "ground contact + offset"}


def scene():
    """앞줄: 정면이 카메라를 향한 칸 (2.5x5.0), 뒷줄: 옆면이 카메라를 향한 칸 (5.0x2.5)."""
    slots = {f"A{i + 1}": box(-5 + 2.5 * i, 10, -2.5 + 2.5 * i, 15) for i in range(4)}
    for r, y0 in (("B", 15.0), ("C", 17.5)):
        for i in range(2):
            slots[f"{r}{i + 1}"] = box(-5 + 5 * i, y0, 5 * i, y0 + 2.5)
    cars = [((-3.75, 12.5), (1.8, 4.7)), ((1.25, 12.5), (1.8, 4.7)), ((2.5, 16.25), (4.7, 1.8))]
    return slots, cars


def car_box_3d(center, size, h=1.5):
    (cx, cy), (w, l) = center, size
    return np.array([(cx + sx * w / 2, cy + sy * l / 2, z) for z in (0, h) for sx, sy in
                     ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/debug/synthetic_contact.png")
    ap.add_argument("--offset", type=float, default=0.9)
    a = ap.parse_args()

    cam = make_camera()
    H, Hinv = cam["H"], np.linalg.inv(cam["H"])
    slots, cars = scene()
    img = np.full((IMG_H, IMG_W, 3), 70, np.uint8)
    px, (ox, oy), bev_w, bev_h = 60, (-6.0, 9.0), 12 * 60, 12 * 60
    bev = np.full((bev_h, bev_w, 3), 70, np.uint8)

    def to_bev(xy):  # 맵(m) -> 탑뷰 픽셀 (위쪽 = 먼 쪽)
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.int32(np.c_[(xy[:, 0] - ox) * px, bev_h - (xy[:, 1] - oy) * px])

    for sid, poly in slots.items():
        corners = np.array(poly.exterior.coords[:-1])
        cv2.polylines(img, [np.int32(transform_points(corners.astype(np.float32), Hinv))], True, (230, 230, 230), 2)
        cv2.polylines(bev, [to_bev(corners)], True, (230, 230, 230), 2)
        cv2.putText(bev, sid, tuple(to_bev(poly.centroid.coords[0])[0] - [12, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    lines = []
    for center, size in sorted(cars, key=lambda c: -c[0][1]):  # 먼 차부터 그려야 가까운 차가 위에 덮임
        b3 = car_box_3d(center, size)
        uv = project(cam, b3).astype(np.float32)
        hull = cv2.convexHull(uv).reshape(-1, 2)
        cv2.fillPoly(img, [np.int32(hull)], (150, 150, 150))
        cv2.polylines(img, [np.int32(uv[:4]), np.int32(uv[4:])], True, (40, 40, 40), 1)
        for i in range(4):
            cv2.line(img, tuple(np.int32(uv[i])), tuple(np.int32(uv[i + 4])), (40, 40, 40), 1)
        cv2.fillPoly(bev, [to_bev(b3[:4, :2])], (150, 150, 150))  # 실제 차 바닥

        bb = [*uv.min(0), *uv.max(0)]
        p_bc = contact_point_img(None, bb, "bbox_center", (IMG_W, IMG_H), 2)[0]
        p_mb = contact_point_img([hull.tolist()], bb, "mask_bottom_center", (IMG_W, IMG_H), 2)[0]
        pts = {"bbox_center": (p_bc, contact_map(p_bc, H, 0)),
               "mask_bottom": (p_mb, contact_map(p_mb, H, 0)),
               "mask_bottom+off": (p_mb, contact_map(p_mb, H, a.offset))}
        true_slot = nearest_slot(np.array(center), slots, 0)
        for k, (pi, pm) in pts.items():
            cv2.circle(img, tuple(np.int32(pi)), 9 if k != "mask_bottom+off" else 5, COLORS[k], -1)
            cv2.circle(bev, tuple(to_bev(pm)[0]), 9, COLORS[k], -1)
            got = nearest_slot(pm, slots, 0.3)
            lines.append(f"car@{true_slot}: {k:16s} -> {got} {'OK' if got == true_slot else 'WRONG'}")

    for i, (k, c) in enumerate(COLORS.items()):
        cv2.circle(img, (30, 40 + 40 * i), 10, c, -1)
        label = LABELS[k] + (f" {a.offset}m" if k == "mask_bottom+off" else "")
        cv2.putText(img, label, (50, 48 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "(in camera view, green and blue are the same pixel: offset is applied on the map)", (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(img, "camera view (h=8m, pitch 35deg)", (30, IMG_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(bev, "top view (far = up)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    bev = cv2.resize(bev, (IMG_H, IMG_H))
    out = np.hstack([img, bev])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, out)
    print("\n".join(lines))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
