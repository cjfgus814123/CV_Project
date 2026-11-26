"""불법주차 GT 확인용 영상: 통로·금지구역 테두리 + 경과 시간(초)만 그린 720p 영상 (시스템 판정은 넣지 않음).

    python tools/make_event_video.py --clip data/clips/lotA_cam7_t00

결과: outputs/label_packs/<clip>_events_check.mp4 (10fps, H.264). 화면의 t = clip 시작부터의 초 = gt_events.csv의 시각.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from parking.homography import map_polyline_to_img
from parking.io import load_cam, load_clip_meta, load_map


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/label_packs")
    ap.add_argument("--fps", type=float, default=10, help="확인용 영상 fps (시각 0.1초 단위면 충분)")
    a = ap.parse_args()

    clip, data = Path(a.clip), Path(a.data)
    meta = load_clip_meta(clip)
    cam, m = load_cam(data / "cams" / meta["cam"]), load_map(data / "lots" / meta["lot"])
    w, h = cam["image_size"]
    ref = (w / 2, h - 1)
    zones = [(z["id"], map_polyline_to_img(z["polygon"], cam["H"], ref, closed=True))
             for z in m.get("aisles", []) + m.get("no_parking", [])]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp, dst = out / f"{clip.name}_events_tmp.mp4", out / f"{clip.name}_events_check.mp4"
    cap = cv2.VideoCapture(str(clip / "video.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(src_fps / a.fps))
    t0, t1 = meta.get("valid_sec") or (0, float("inf"))
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / step, (1280, 720))
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / src_fps
        if i % step == 0 and t0 <= t <= t1:
            f = cap.retrieve()[1]
            for zid, pieces in zones:
                for p in pieces:
                    cv2.polylines(f, [np.int32(p)], True, (0, 165, 255), 3, cv2.LINE_AA)
                if pieces:
                    c = np.concatenate(pieces).mean(0).astype(int)
                    cv2.putText(f, zid, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(f, zid, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2, cv2.LINE_AA)
            txt = f"{clip.name}  t = {t:7.1f} s"
            cv2.rectangle(f, (0, 0), (760, 70), (0, 0, 0), -1)
            cv2.putText(f, txt, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            writer.write(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA))
        i += 1
    writer.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "26",
                        "-preset", "fast", "-pix_fmt", "yuv420p", str(dst)], check=True)
        tmp.unlink()
    else:
        tmp.replace(dst)
    template = out / "gt_events_template.csv"
    template.write_text("event_id,clip_id,type,zone_id,start_sec,end_sec\n", encoding="utf-8")
    print(f"-> {dst} ({dst.stat().st_size / 1e6:.0f}MB), 입력 양식 {template}")


if __name__ == "__main__":
    main()
