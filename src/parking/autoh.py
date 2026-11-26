"""새 촬영 위치의 H 자동 추정 (같은 lot의 다른 cam을 기준으로). setup_cam.py --auto 에서 쓴다.

1. 기준 cam마다: 기준 프레임에서 바닥(주차장 안) SIFT 특징점만 골라 새 프레임과 매칭 -> 영상 간 바닥 Homography G.
   RANSAC 일치점이 가장 많은 기준 cam의 G로 대략 H를 정한다 (시점이 많이 다르면 일치점이 적어 실패로 판단).
2. 국소 정밀화: 기준 cam들의 손 대응점(맵 좌표 앎)을 G^-1로 새 프레임에 예측하고, 기준 프레임을 G^-1로
   새 시점에 맞게 편 뒤 점 주변 패치를 새 프레임에서 ±search_px 안에서 찾는다 (NCC). 바닥 평면이라 편 패치가 잘 맞는다.
3. 찾은 점 + 맵 좌표로 RANSAC Homography -> 일치한 점만 대응점으로 돌려준다 (사람은 setup_cam 화면에서 확인).
시험 (2026-11-26, lotA 세 cam을 하나씩 새 위치로 가정): 대략 H 평균 0.14~0.19m -> 정밀화 후 0.06~0.15m (손 H 0.03~0.05m).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from parking.homography import transform_points


def lot_polygon(map_data: dict, buffer_m: float) -> Polygon:
    """맵의 칸·통로를 감싸는 볼록 다각형 + buffer_m (바닥 특징점을 고르는 범위)."""
    pts = [p for z in map_data["slots"] + map_data.get("aisles", []) for p in z["polygon"]]
    return MultiPoint(pts).convex_hull.buffer(buffer_m)


def load_refs(data: Path, lot: str, exclude_cam: str | None = None) -> list[dict]:
    """같은 lot의 다른 cam들: {cam, img(회색), H, points([{img, map}])}."""
    refs = []
    for f in sorted((Path(data) / "cams").glob("*/H.json")):
        h = json.loads(f.read_text(encoding="utf-8"))
        img_path = f.parent / "ref_frame.png"
        if h.get("lot_id") != lot or h.get("cam_id") == exclude_cam or not img_path.exists():
            continue
        refs.append({"cam": h["cam_id"], "img": cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE),
                     "H": np.array(h["H"], np.float64), "points": h["correspondences"] + h["validation"]})
    return refs


def _features(img: np.ndarray, n: int):
    k, d = cv2.SIFT_create(n).detectAndCompute(img, None)
    return np.float32([x.pt for x in k]).reshape(-1, 2), d


def match_ground(new: tuple, ref: dict, lot: Polygon, cfg: dict) -> tuple[np.ndarray | None, int]:
    """새 프레임 특징점 new=(pts, desc) -> 기준 프레임 바닥 특징점. (G: 새 영상 -> 기준 영상, RANSAC 일치 수)."""
    pr, dr = ref.setdefault("_feat", _features(ref["img"], cfg["sift_features"]))
    ground = np.array([lot.contains(Point(*m)) for m in transform_points(pr, ref["H"])], bool)
    if ground.sum() < 4 or new[1] is None:
        return None, 0
    pr, dr = pr[ground], dr[ground]
    good = [m[0] for m in cv2.BFMatcher().knnMatch(new[1], dr, k=2)
            if len(m) == 2 and m[0].distance < cfg["ratio"] * m[1].distance]
    if len(good) < 4:
        return None, 0
    src = new[0][[g.queryIdx for g in good]]
    dst = pr[[g.trainIdx for g in good]]
    G, inl = cv2.findHomography(src, dst, cv2.RANSAC, cfg["ransac_px"])
    return (G, int(inl.sum())) if G is not None else (None, 0)


def refine_points(img: np.ndarray, ref: dict, G: np.ndarray, cfg: dict) -> list[dict]:
    """기준 cam의 대응점을 새 프레임에서 국소 패치 매칭으로 정밀하게 찾는다. [{img, map, score}]."""
    h, w = img.shape
    Ginv = np.linalg.inv(G)
    warped = cv2.warpPerspective(ref["img"], Ginv, (w, h))  # 기준 프레임을 새 시점으로 (바닥 기준)
    R, T = cfg["patch_px"], cfg["search_px"]
    out = []
    for p in ref["points"]:
        x, y = np.round(transform_points(np.float32([p["img"]]), Ginv)[0]).astype(int)
        if not (R + T <= x < w - R - T and R + T <= y < h - R - T):
            continue
        tpl = warped[y - R:y + R + 1, x - R:x + R + 1]
        if tpl.std() < cfg["min_texture"]:  # 밋밋한 곳 (펴서 생긴 빈 곳 포함)
            continue
        s = cv2.matchTemplate(img[y - R - T:y + R + T + 1, x - R - T:x + R + T + 1], tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(s)
        if score >= cfg["min_score"]:
            out.append({"img": [float(x - T + loc[0]), float(y - T + loc[1])], "map": list(p["map"]),
                        "score": float(score)})
    return out


def auto_correspondences(img: np.ndarray, refs: list[dict], lot: Polygon, cfg: dict) -> tuple[list[dict], dict]:
    """새 프레임(회색) -> (대응점 [{img, map}], 정보). 실패하면 대응점 []."""
    new = _features(img, cfg["sift_features"])
    matches = {r["cam"]: match_ground(new, r, lot, cfg) for r in refs}
    info = {"inliers": {k: v[1] for k, v in matches.items()}}
    best = max(matches, key=lambda k: matches[k][1], default=None)
    if best is None or matches[best][1] < cfg["min_inliers"]:
        info["reason"] = f"특징점 일치 부족 ({info['inliers']}, 기준 {cfg['min_inliers']})"
        return [], info
    info["ref"] = best
    cand = {}
    for r in refs:  # 일치점이 충분한 기준 cam의 점은 모두 쓴다. 같은 맵 좌표는 점수가 높은 것만
        G, n = matches[r["cam"]]
        if n < cfg["min_inliers"]:
            continue
        for p in refine_points(img, r, G, cfg):
            k = tuple(round(v, 3) for v in p["map"])
            if k not in cand or p["score"] > cand[k]["score"]:
                cand[k] = p
    pts = list(cand.values())
    if len(pts) < 4:
        info["reason"] = f"정밀화된 점 부족 ({len(pts)})"
        return [], info
    _, inl = cv2.findHomography(np.float32([p["img"] for p in pts]), np.float32([p["map"] for p in pts]),
                                cv2.RANSAC, cfg["ransac_m"])
    keep = [{"img": p["img"], "map": p["map"]} for p, ok in zip(pts, inl.ravel()) if ok]
    info.update(found=len(pts), kept=len(keep))
    return keep, info
