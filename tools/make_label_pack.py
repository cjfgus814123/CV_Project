"""GT 라벨링 꾸러미: 프레임 고르기 -> gt_frames/ -> zip (팀원용, team_guide.md).

    python tools/make_label_pack.py --clips data/clips/lotA_cam7_t* --every-sec 120
    python tools/make_label_pack.py --clips data/clips/lotA_cam4_a --every-sec 5

프레임은 검출 결과와 상관없이 영상에서 바로 고른다 (예측을 보지 않는 공정한 GT, 검출 전에 라벨링 시작 가능).
- 후보: valid_sec 안, 프레임 번호가 15의 배수 (2fps·10fps 검출 캐시 모두에 있는 프레임)
- 고름: every_sec 간격 + "마지막으로 고른 프레임과 비교해 주차장 영역 픽셀이 change_frac 이상 바뀐" 프레임
  (차가 들어오고 나간 순간. min_gap_sec 간격 이상). 픽셀 차이만 보므로 시스템 예측과 무관.
- 결과: data/clips/<clip>/gt_frames/f<번호>.jpg, outputs/label_packs/<clip>_label.zip
  zip 안: data/clips/<clip>/{meta.json, gt_frames/}, data/cams/<cam>/H.json, data/lots/<lot>/map.json
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from parking.autoh import lot_polygon
from parking.homography import map_polyline_to_img
from parking.io import load_cam, load_clip_meta, load_map

STEP = 15  # 2fps 캐시 간격 (원본 약 30fps)
SMALL = (480, 270)


def lot_mask_small(cam: dict, m: dict, buffer_m: float) -> np.ndarray:
    """주차장 영역(칸·통로 + buffer)을 영상에 투영한 마스크 (SMALL 크기, bool)."""
    w, h = cam["image_size"]
    poly = list(lot_polygon(m, buffer_m).exterior.coords)
    mask = np.zeros((h, w), np.uint8)
    for p in map_polyline_to_img(poly, cam["H"], (w / 2, h - 1), closed=True):
        cv2.fillPoly(mask, [np.int32(p)], 1)
    return cv2.resize(mask, SMALL, interpolation=cv2.INTER_NEAREST).astype(bool)


def pick_frames(video: Path, valid: tuple[float, float], mask: np.ndarray, every_sec: float, change_frac: float,
                min_gap_sec: float, out_dir: Path) -> list[tuple[int, str]]:
    """영상을 처음부터 순서대로 읽으며 (seek 금지) 프레임을 골라 저장. [(번호, 이유)]."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked, last_small, last_t, i = [], None, -1e9, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / fps
        if i % STEP == 0 and valid[0] <= t <= valid[1]:
            frame = cap.retrieve()[1]
            small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, SMALL), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            reason = None
            if last_small is None or t - last_t >= every_sec:
                reason = "every"
            elif t - last_t >= min_gap_sec:
                frac = float((cv2.absdiff(small, last_small)[mask] > 25).mean())
                if frac >= change_frac:
                    reason = f"change {frac:.3f}"
            if reason:
                cv2.imwrite(str(out_dir / f"f{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                picked.append((i, reason))
                last_small, last_t = small, t
        i += 1
    return picked


def make_zip(clip: Path, meta: dict, data: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{clip.name}_label.zip"
    files = [clip / "meta.json", *sorted((clip / "gt_frames").glob("*.jpg")),
             data / "cams" / meta["cam"] / "H.json", data / "lots" / meta["lot"] / "map.json"]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # jpg는 이미 압축됨
        for f in files:
            z.write(f, Path("data") / f.relative_to(data))
    return zpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--every-sec", type=float, default=120, help="일정 간격 (초)")
    ap.add_argument("--change-frac", type=float, default=0.01, help="주차장 영역 픽셀 중 바뀐 비율이 이 이상이면 고름")
    ap.add_argument("--min-gap-sec", type=float, default=3.0, help="변화로 고를 때 최소 간격 (초)")
    ap.add_argument("--buffer-m", type=float, default=1.0, help="주차장 영역 = 칸·통로 + 이 값 (m)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/label_packs")
    ap.add_argument("--force", action="store_true", help="gt_frames가 있어도 다시 고름 (GT가 이미 있으면 거부)")
    a = ap.parse_args()

    data = Path(a.data)
    for clip in map(Path, a.clips):
        meta = load_clip_meta(clip)
        gt_dir = clip / "gt_frames"
        if (clip / "gt_occupancy.csv").exists():
            print(f"건너뜀 {clip.name}: gt_occupancy.csv가 이미 있음 (프레임을 바꾸면 GT가 어긋남)")
            continue
        if gt_dir.exists() and any(gt_dir.iterdir()):
            if not a.force:
                print(f"{clip.name}: gt_frames 이미 있음 -> zip만 다시 만듦 (--force로 다시 고르기)")
                print(f"  -> {make_zip(clip, meta, data, Path(a.out))}")
                continue
            shutil.rmtree(gt_dir)
        cam, m = load_cam(data / "cams" / meta["cam"]), load_map(data / "lots" / meta["lot"])
        mask = lot_mask_small(cam, m, a.buffer_m)
        valid = tuple(meta.get("valid_sec") or (0, float("inf")))
        picked = pick_frames(clip / "video.mp4", valid, mask, a.every_sec, a.change_frac, a.min_gap_sec, gt_dir)
        n_change = sum(r != "every" for _, r in picked)
        z = make_zip(clip, meta, data, Path(a.out))
        print(f"{clip.name}: {len(picked)}장 (간격 {len(picked) - n_change} + 변화 {n_change}) -> {z} "
              f"({z.stat().st_size / 1e6:.0f}MB)")


if __name__ == "__main__":
    main()
