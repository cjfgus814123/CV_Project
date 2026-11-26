"""논문 ② 방식 베이스라인 (CLAUDE.md 6): 영상 위 칸 폴리곤 + 박스 중심점 판정.

영상 칸 폴리곤 = 같은 map 칸을 H^-1로 투영한 것이므로, "박스 중심이 영상 칸 안" ⇔ "H로 옮긴 박스 중심이 맵 칸 안".
그래서 별도 코드 없이 파이프라인 설정만 바꿔 같은 코드 경로로 돌린다 (공정한 비교).
"""
from __future__ import annotations

import copy

BASELINE = {"contact": {"mode": "bbox_center", "depth_offset_m": 0.0},
            "occupancy": {"boundary_tol_m": 0.0, "smooth_sec": 0}, "footprint": {"enabled": False}}


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """cfg의 깊은 복사본에 {섹션: {키: 값}} 덮어쓰기."""
    out = copy.deepcopy(cfg)
    for sec, kv in overrides.items():
        out.setdefault(sec, {}).update(kv)
    return out
