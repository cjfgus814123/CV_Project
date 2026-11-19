"""칸 줄 생성 (CLAUDE.md 5.1-4a). (확장) 탑뷰 자동 칸 검출은 아직 없음."""
from __future__ import annotations

import numpy as np


def slot_row(p0, p1, p2, count: int, prefix: str, start: int = 1) -> list[dict]:
    """한 줄의 칸들 (맵 좌표, m).

    p0: 줄 첫 칸의 앞 모서리, p1: 줄 마지막 칸의 앞 모서리(줄 끝), p2: 첫 칸의 뒤 모서리 (p0와 같은 선 위).
    칸 폭 = |p1-p0|/count (표준 폭이 아니라 실제 선 간격에 맞춤), 칸 옆변 = p2-p0.
    옆변이 줄과 수직이 아니면 평행사변형 칸 (사선 주차).
    """
    p0, p1, p2 = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    step, side = (p1 - p0) / count, p2 - p0
    slots = []
    for i in range(count):
        a = p0 + step * i
        poly = [a, a + step, a + step + side, a + side]
        slots.append({"id": f"{prefix}{start + i:02d}", "row": prefix,
                      "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in poly]})
    return slots
