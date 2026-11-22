"""Split-View 영상, 2D 맵 그리기 (CLAUDE.md 5.6).

색: 빈칸 초록, 점유 빨강, 침범 노랑(2차), 판단불가 회색. 가시 비율이 낮은(불확실) 칸은 이름 뒤에 '?'.
"""
from __future__ import annotations

import cv2
import numpy as np

from parking.occupancy import ENCROACH, FREE, OCCUPIED, UNKNOWN

STATE_COLOR = {FREE: (0, 200, 0), OCCUPIED: (0, 0, 230), ENCROACH: (0, 220, 255), UNKNOWN: (150, 150, 150)}


def _fill(img: np.ndarray, polys: dict[str, list[np.ndarray]], states: dict[str, int], uncertain: set[str],
          alpha: float, thick: int, font: float) -> np.ndarray:
    """칸 다각형(픽셀 좌표 조각 목록)을 상태 색으로 반투명 채우고 테두리·이름을 그린다."""
    over = img.copy()
    for sid, pieces in polys.items():
        col = STATE_COLOR[states.get(sid, UNKNOWN)]
        for p in pieces:
            if len(p) >= 3:
                cv2.fillPoly(over, [np.int32(p)], col)
    out = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for sid, pieces in polys.items():
        col = STATE_COLOR[states.get(sid, UNKNOWN)]
        for p in pieces:
            cv2.polylines(out, [np.int32(p)], True, col, thick, cv2.LINE_AA)
        if pieces:
            c = np.concatenate(pieces).mean(0)
            label = sid + ("?" if sid in uncertain else "")
            cv2.putText(out, label, (int(c[0]) - 14, int(c[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255),
                        1, cv2.LINE_AA)
    return out


def draw_frame_view(frame_img: np.ndarray, dets: list[dict], slots_img: dict, states: dict[str, int],
                    uncertain: set[str] = frozenset()) -> np.ndarray:
    """왼쪽: 원본 + 칸 폴리곤(H^-1로 투영) + 검출 껍질·접점. dets에는 'contact_img', 'slot', 'truncated'가 있어야 함."""
    out = _fill(frame_img, slots_img, states, uncertain, 0.3, 2, 0.5)
    for d in dets:
        if d["mask_polys_img"]:
            hull = cv2.convexHull(np.concatenate([np.int32(p) for p in d["mask_polys_img"]]))
            cv2.polylines(out, [hull], True, (255, 220, 0), 1, cv2.LINE_AA)
        col = (0, 255, 255) if d["slot"] else (255, 0, 255)  # 칸에 배정 안 된 접점(통로 등)은 자홍
        cv2.circle(out, tuple(np.int32(d["contact_img"])), 6, col, -1)
        if d["truncated"]:
            cv2.circle(out, tuple(np.int32(d["contact_img"])), 10, col, 1)
    return out


def draw_map_view(bev_img: np.ndarray, slots_bev: dict, states: dict[str, int], contacts_bev: np.ndarray,
                  uncertain: set[str] = frozenset(), footprints_bev: list | None = None) -> np.ndarray:
    """오른쪽: map_bev 위 칸 상태 색칠 + (2차) 차 바닥 사각형 + 접점."""
    out = _fill(bev_img, slots_bev, states, uncertain, 0.45, 1, 0.38)
    for fp in footprints_bev or []:
        cv2.polylines(out, [np.int32(fp).reshape(-1, 1, 2)], True, (255, 255, 255), 1, cv2.LINE_AA)
    for p in np.asarray(contacts_bev).reshape(-1, 2):
        cv2.circle(out, tuple(np.int32(p)), 4, (0, 255, 255), -1)
    return out


def split_view(left: np.ndarray, right: np.ndarray, header: str, height: int = 720) -> np.ndarray:
    """두 그림을 같은 높이로 맞춰 나란히 + 위쪽 정보 줄."""
    scale = lambda im: cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)
    out = np.hstack([scale(left), scale(right)])
    bar = np.zeros((44, out.shape[1], 3), np.uint8)
    cv2.putText(bar, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, out])
