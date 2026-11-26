"""GT 라벨링용 프레임 샘플링 (CLAUDE.md 7).

    python tools/sample_frames.py --clip data/clips/lotA_cam3_mid --every-sec 10
    python tools/sample_frames.py --clip data/clips/lotA_cam3_mid --every-sec 2 --start 455 --end 705 --max 200  # 연출 구간만 촘촘히

검출 캐시(outputs/<clip>/detections.jsonl)에 있는 프레임 중에서 every-sec 간격으로 골라
data/clips/<clip>/gt_frames/f<frame_idx>.jpg로 저장한다 (예측과 같은 프레임이어야 비교 가능).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from parking.detect import cache_name, load_cache


def pick(times: list[float], every_sec: float, max_n: int) -> list[int]:
    """시각 목록에서 every_sec 간격에 가장 가까운 인덱스들 (최대 max_n개, 넘으면 간격을 넓힘)."""
    t0, t1 = times[0], times[-1]
    step = max(every_sec, (t1 - t0) / max(max_n - 1, 1))
    out, k = [], 0
    t = t0
    while t <= t1 + 1e-9:
        while k + 1 < len(times) and abs(times[k + 1] - t) <= abs(times[k] - t):
            k += 1
        if not out or out[-1] != k:
            out.append(k)
        t += step
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--every-sec", type=float, default=5.0)
    ap.add_argument("--max", type=int, default=100, help="최대 프레임 수 (넘으면 간격을 자동으로 넓힘)")
    ap.add_argument("--start", type=float, default=None, help="이 시각(초) 이후만")
    ap.add_argument("--end", type=float, default=None, help="이 시각(초) 이전만")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--fps", type=float, default=2, help="고를 검출 캐시의 fps. 기본 2 (15의 배수 프레임 -> 2·10·30fps 캐시 모두에 있음)")
    a = ap.parse_args()

    clip = Path(a.clip)
    recs = [r for r in load_cache(Path(a.out) / clip.name / cache_name(a.fps))
            if (a.start is None or r["time_sec"] >= a.start) and (a.end is None or r["time_sec"] <= a.end)]
    idx = [recs[k]["frame_idx"] for k in pick([r["time_sec"] for r in recs], a.every_sec, a.max)]
    out = clip / "gt_frames"
    out.mkdir(exist_ok=True)
    cap, i, want = cv2.VideoCapture(str(clip / "video.mp4")), -1, set(idx)
    while want and cap.grab():  # 순서대로 세며 읽기 (seek 금지)
        i += 1
        if i in want:
            cv2.imwrite(str(out / f"f{i:06d}.jpg"), cap.retrieve()[1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            want.discard(i)
    print(f"{len(idx)}프레임 -> {out}/ (간격 약 {(recs[-1]['time_sec'] - recs[0]['time_sec']) / max(len(idx) - 1, 1):.1f}s)")


if __name__ == "__main__":
    main()
