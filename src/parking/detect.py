"""YOLO-seg 추론, 결과 JSONL 캐시 (CLAUDE.md 5.2, 5.3).

    python -m parking.detect --clip data/clips/lotA_cam1_full

- 원본(비스듬한) 프레임에 그대로 돌린다 (탑뷰로 펴지 않음).
- 프레임은 처음부터 순서대로 읽어 번호를 센다 (seek 금지: 폰 영상에서 프레임 번호가 어긋날 수 있음).
- 마스크는 영상 픽셀 좌표 다각형 목록(mask_polys_img)으로 저장한다 (RLE보다 작고 의존성 불필요).
  가로등·나무에 가려 마스크가 여러 조각이면 조각마다 다각형 1개.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def load_model(cfg: dict):
    """configs의 model.name을 weights_dir에서 읽는다. 없으면 Ultralytics가 자동 다운로드."""
    from ultralytics import YOLO

    m = cfg["model"]
    path = Path(m.get("weights_dir", ".")) / m["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(path))


def mask_to_polys(mask: np.ndarray, min_area_px: float = 20.0) -> list[list[list[int]]]:
    """이진 마스크 -> 외곽선 다각형 목록 (조각마다 1개, 1px 허용 오차로 단순화).

    Ultralytics의 masks.xy는 여러 조각을 한 다각형으로 이어 붙여 모양이 깨지므로 직접 구한다.
    """
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).tolist()
            for c in cs if cv2.contourArea(c) >= min_area_px]


def polys_to_mask(polys: list, image_size: tuple[int, int]) -> np.ndarray:
    """mask_to_polys의 역변환. image_size = (w, h) -> bool (h, w)."""
    m = np.zeros(image_size[::-1], np.uint8)
    cv2.fillPoly(m, [np.int32(p) for p in polys], 1)
    return m.astype(bool)


def detect_frame(model, frame_img: np.ndarray, cfg: dict) -> list[dict]:
    """프레임 1장 검출 -> [{cls, conf, bbox_img, mask_polys_img}]"""
    m = cfg["model"]
    r = model.predict(frame_img, imgsz=m["imgsz"], conf=m["conf"], classes=m["classes"],
                      retina_masks=True, verbose=False)[0]  # retina: 마스크를 원본 해상도로
    masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else [None] * len(r.boxes)
    dets = []
    for box, cls, conf, mask in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), masks):
        dets.append({
            "cls": int(cls),
            "conf": round(float(conf), 4),
            "bbox_img": [round(v, 1) for v in box],
            "mask_polys_img": [] if mask is None else mask_to_polys(mask),
        })
    return dets


def run_detection(video_path: str | Path, cfg: dict, cache_path: str | Path,
                  valid_sec: tuple[float, float] | None = None, vis_dir: str | Path | None = None,
                  vis_every: int = 0) -> Path:
    """sampling.fps 간격으로 검출해 프레임별 JSONL로 저장한다. valid_sec 밖의 프레임은 건너뜀."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / cfg["sampling"]["fps"]))
    t0, t1 = valid_sec if valid_sec else (0.0, float("inf"))
    model = load_model(cfg)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    n, n_det, t_start, i = 0, 0, time.time(), 0
    with cache_path.open("w", encoding="utf-8") as f:
        while True:
            t = i / fps
            take = i % step == 0 and t0 <= t <= t1
            ok = cap.grab()  # 건너뛸 프레임은 grab만 (색 변환 생략) -> 순서대로 세면서도 빠름
            if not ok:
                break
            if take:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                dets = detect_frame(model, frame, cfg)
                f.write(json.dumps({"frame_idx": i, "time_sec": round(t, 3), "dets": dets}) + "\n")
                if vis_dir and vis_every and n % vis_every == 0:
                    cv2.imwrite(str(Path(vis_dir) / f"f{i:06d}.jpg"), _draw(frame, dets))
                n += 1
                n_det += len(dets)
            i += 1
    cap.release()
    dt = time.time() - t_start
    print(f"{n}프레임 검출 (step {step}, 원본 {fps:.2f}fps), 평균 {n_det / max(n, 1):.1f}대/프레임, "
          f"{dt:.0f}s ({dt / max(n, 1) * 1000:.0f} ms/프레임, 영상 디코딩 포함) -> {cache_path}")
    return cache_path


def _draw(frame: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox_img"])
        cv2.polylines(out, [np.int32(p) for p in d["mask_polys_img"]], True, (0, 255, 255), 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(out, f"{d['cls']}:{d['conf']:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return out


def dedup(dets: list[dict], ios_thr: float) -> list[dict]:
    """한 차가 두 번 검출된 경우 제거: 두 박스의 겹침 / 작은 박스 면적 >= ios_thr 이면 신뢰도 낮은 쪽을 버린다.

    IoU 대신 '작은 박스 기준 겹침'을 쓰는 이유: 문 열린 차·사람 때문에 큰 박스 안에 작은 박스가 또 잡히면 IoU는 낮다.
    """
    keep: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        x1, y1, x2, y2 = d["bbox_img"]
        a = (x2 - x1) * (y2 - y1)
        for k in keep:
            X1, Y1, X2, Y2 = k["bbox_img"]
            inter = max(0, min(x2, X2) - max(x1, X1)) * max(0, min(y2, Y2) - max(y1, Y1))
            if inter / max(min(a, (X2 - X1) * (Y2 - Y1)), 1e-9) >= ios_thr:
                break
        else:
            keep.append(d)
    return keep


def load_cache(cache_path: str | Path) -> list[dict]:
    with open(cache_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    from parking.io import load_clip_meta, load_config

    ap = argparse.ArgumentParser(description="clip 검출 -> outputs/<clip_id>/detections.jsonl")
    ap.add_argument("--clip", required=True, help="data/clips/<clip_id>")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--vis-every", type=int, default=0, help="N번째 검출 프레임마다 확인용 그림 저장 (0=안 함)")
    a = ap.parse_args()

    clip = Path(a.clip)
    meta = load_clip_meta(clip)
    out_dir = Path(a.out) / clip.name
    run_detection(clip / "video.mp4", load_config(a.config), out_dir / "detections.jsonl",
                  meta.get("valid_sec"), out_dir / "det_vis", a.vis_every)


if __name__ == "__main__":
    main()
