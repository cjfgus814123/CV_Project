"""(3차) ByteTrack 래퍼 (Ultralytics BYTETracker).

검출 캐시(기본 2fps)에 그대로 붙인다. 불법주차는 T_stop(60s) 이상 서 있는 차를 찾는 일이라
2fps면 충분하고, 서 있는 차는 프레임 사이 박스가 거의 같아 IoU 매칭이 쉽다.
지나가는 차는 2fps에서 ID가 바뀔 수 있지만 정지 판정에는 영향이 적다.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class Tracker:
    """프레임마다 update(dets) -> 같은 순서의 track_id 목록 (매칭 안 된 검출은 None)."""

    def __init__(self, cfg: dict, image_size: tuple[int, int]):
        t = cfg["track"]
        self.tracker = BYTETracker(SimpleNamespace(
            tracker_type="bytetrack", track_high_thresh=t["high_thresh"], track_low_thresh=t["low_thresh"],
            new_track_thresh=t["new_track_thresh"], track_buffer=t["buffer_frames"], match_thresh=t["match_thresh"],
            fuse_score=True))
        self.shape = (image_size[1], image_size[0])

    def update(self, dets: list[dict]) -> list[int | None]:
        if not dets:
            self.tracker.update(Boxes(np.zeros((0, 6), np.float32), self.shape))
            return []
        data = np.float32([[*d["bbox_img"], d["conf"], d["cls"]] for d in dets])
        ids: list[int | None] = [None] * len(dets)
        for row in self.tracker.update(Boxes(data, self.shape)):
            ids[int(row[-1])] = int(row[4])
        return ids
