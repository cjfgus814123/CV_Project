"""검출·지면 접점 시각 디버깅 (실제 영상). detect 캐시(detections.jsonl)가 있어야 한다.

    python tools/view_dets.py --clip data/clips/lotA_cam1_full              # OpenCV 창
    python tools/view_dets.py --clip data/clips/lotA_cam1_full --save 5     # 창 없이 5장 저장

표시: 노랑 = 마스크, 하늘색 = 볼록 껍질, 점 = 접점 3종 (빨강 박스 중심 / 파랑 박스 아래 / 초록 지면 접점),
자홍 박스 = 화면 경계에 잘린 검출(truncated).
키: d/→ 다음, a/← 이전, w 10장 앞으로, m 마스크 켜기/끄기, s 현재 그림 저장, q/ESC 종료.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from parking.contact import contact_point_img
from parking.detect import cache_name, load_cache
from parking.io import load_config

PT_COLORS = {"bbox_center": (0, 0, 255), "bbox_bottom_center": (255, 120, 0), "mask_bottom_center": (0, 220, 0)}


class FrameReader:
    """프레임을 처음부터 순서대로 읽는다 (seek 금지 규칙). 지나간 프레임은 JPEG로 기억해 뒤로 가기 지원."""

    def __init__(self, video: Path):
        self.cap = cv2.VideoCapture(str(video))
        self.i = -1
        self.seen: dict[int, bytes] = {}

    def get(self, idx: int) -> np.ndarray:
        if idx in self.seen:
            return cv2.imdecode(np.frombuffer(self.seen[idx], np.uint8), cv2.IMREAD_COLOR)
        while self.i < idx:
            if not self.cap.grab():
                raise ValueError(f"프레임 {idx}까지 읽지 못함")
            self.i += 1
        frame = self.cap.retrieve()[1]
        self.seen[idx] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        return frame


def draw(frame: np.ndarray, rec: dict, cfg: dict, show_mask: bool) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]
    margin = cfg["contact"]["edge_margin_px"]
    for d in rec["dets"]:
        polys = d["mask_polys_img"]
        if show_mask and polys:
            cv2.polylines(out, [np.int32(p) for p in polys], True, (0, 230, 255), 1)
            hull = cv2.convexHull(np.concatenate([np.int32(p) for p in polys]))
            cv2.polylines(out, [hull], True, (255, 220, 0), 1)
        truncated = False
        for mode, color in PT_COLORS.items():
            pt, truncated = contact_point_img(polys, d["bbox_img"], mode, (w, h), margin)
            cv2.circle(out, tuple(np.int32(pt)), 5, color, -1)
        if truncated:
            x1, y1, x2, y2 = map(int, d["bbox_img"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(out, f"{d['conf']:.2f}", tuple(np.int32(d["bbox_img"][:2])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    head = f"frame {rec['frame_idx']}  t={rec['time_sec']:.1f}s  cars={len(rec['dets'])}"
    legend = "red=bbox center  blue=bbox bottom  green=ground contact(hull)  magenta=truncated"
    keys = "[d/->] next  [a/<-] prev  [w] +10  [m] mask  [s] save  [q] quit"
    cv2.rectangle(out, (0, 0), (900, 92), (0, 0, 0), -1)
    for i, t in enumerate((head, legend, keys)):
        cv2.putText(out, t, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save", type=int, default=0, help="창 없이 고르게 N장만 저장")
    a = ap.parse_args()

    clip = Path(a.clip)
    out_dir = Path(a.out) / clip.name
    cfg = load_config(a.config)
    recs = load_cache(out_dir / cache_name(cfg["sampling"]["fps"]))
    reader = FrameReader(clip / "video.mp4")
    save_dir = out_dir / "debug_contact"
    save_dir.mkdir(parents=True, exist_ok=True)

    if a.save:
        for k in np.linspace(0, len(recs) - 1, a.save).round().astype(int):
            p = save_dir / f"f{recs[k]['frame_idx']:06d}.jpg"
            cv2.imwrite(str(p), draw(reader.get(recs[k]["frame_idx"]), recs[k], cfg, True))
            print(p)
        return

    win = "view_dets"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)
    k, show_mask = 0, True
    while True:
        img = draw(reader.get(recs[k]["frame_idx"]), recs[k], cfg, show_mask)
        cv2.imshow(win, img)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("d"), 65363):
            k = min(k + 1, len(recs) - 1)
        elif key in (ord("a"), 65361):
            k = max(k - 1, 0)
        elif key == ord("w"):
            k = min(k + 10, len(recs) - 1)
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("s"):
            p = save_dir / f"f{recs[k]['frame_idx']:06d}.jpg"
            cv2.imwrite(str(p), img)
            print(f"저장: {p}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
