"""(3차) 통로·금지구역 정지 시간 기반 불법주차 이벤트 (CLAUDE.md 5.5).

트랙별 맵 좌표 궤적에서, 구역 폴리곤 안에서 속도 < v_stop인 상태가 T_stop 이상 이어지면 이벤트.
속도는 최근 window_sec 동안의 이동 거리 / 시간 (접점이 프레임마다 0.1~0.3m 흔들려서 한 프레임 차이로는 못 씀).
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon


class StopDetector:
    def __init__(self, zones: dict[str, Polygon], v_stop: float, t_stop: float, window_sec: float):
        self.zones, self.v_stop, self.t_stop, self.window = zones, v_stop, t_stop, window_sec
        self.hist: dict[int, list[tuple[float, np.ndarray]]] = {}
        self.run: dict[int, tuple[str, float]] = {}  # track -> (구역, 정지 시작 시각)
        self.active: dict[int, dict] = {}            # track -> 진행 중 이벤트
        self.events: list[dict] = []

    def _zone(self, p: np.ndarray) -> str | None:
        pt = Point(float(p[0]), float(p[1]))
        return next((k for k, z in self.zones.items() if z.contains(pt)), None)

    def _speed(self, h: list[tuple[float, np.ndarray]]) -> float | None:
        """최근 window 동안의 평균 속도. 기록이 window만큼 쌓이기 전이면 None."""
        t1, p1 = h[-1]
        old = [(t, p) for t, p in h if t <= t1 - self.window]
        if not old:
            return None
        t0, p0 = old[-1]
        return float(np.linalg.norm(p1 - p0)) / (t1 - t0)

    def update(self, t: float, tracks: list[tuple[int, np.ndarray]]) -> set[int]:
        """한 프레임: (track_id, 맵 좌표) 목록. 지금 불법주차 중인 track_id 집합을 돌려준다."""
        seen = set()
        for tid, p in tracks:
            seen.add(tid)
            h = self.hist.setdefault(tid, [])
            h.append((t, np.asarray(p, np.float64)))
            del h[:-int(self.window * 10 + 50)]  # ponytail: 최근 기록만 유지 (window 확인용)
            zone, v = self._zone(p), self._speed(h)
            stopped = zone is not None and v is not None and v < self.v_stop
            if stopped:
                z0, t0 = self.run.get(tid, (zone, t - self.window))  # 정지 시작 = 속도 창의 시작
                if z0 != zone:
                    z0, t0 = zone, t
                self.run[tid] = (z0, t0)
                if tid not in self.active and t - t0 >= self.t_stop:
                    self.active[tid] = {"track_id": tid, "zone_id": zone, "start_sec": round(t0, 2),
                                        "detect_sec": round(t, 2), "end_sec": round(t, 2)}
                if tid in self.active:
                    self.active[tid]["end_sec"] = round(t, 2)
            else:
                self.run.pop(tid, None)
                self._close(tid)
        return {tid for tid in self.active if tid in seen}

    def _close(self, tid: int) -> None:
        if tid in self.active:
            self.events.append(self.active.pop(tid))

    def finish(self) -> list[dict]:
        """영상 끝: 진행 중 이벤트를 닫고 전체 이벤트 목록 (event_id 붙임)."""
        for tid in list(self.active):
            self._close(tid)
        self.events.sort(key=lambda e: e["start_sec"])
        return [{"event_id": f"P{i + 1:02d}", **e} for i, e in enumerate(self.events)]
