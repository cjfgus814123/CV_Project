import numpy as np

from parking.io import load_config
from parking.pipeline import process_frame
from parking.synthetic import IMG_H, IMG_W, car_mask_poly, make_camera
from parking.map_builder import slot_row
from parking.occupancy import slots_from_map


def test_process_frame_assigns_cars_and_drops_duplicate():
    cfg = load_config("configs/default.yaml")
    cam = make_camera()
    slots = slots_from_map({"slots": slot_row((-5, 10), (5, 10), (-5, 15), 4, "A")})
    dets = []
    for cx in (-3.75, 1.25):  # A01, A03에 차
        polys, bbox = car_mask_poly(cam, (cx, 12.5))
        dets.append({"cls": 2, "conf": 0.9, "bbox_img": list(bbox), "mask_polys_img": polys})
    x1, y1, x2, y2 = dets[0]["bbox_img"]  # A01 차 안쪽에 작은 중복 검출
    dets.append({"cls": 2, "conf": 0.5, "bbox_img": [x1 + 10, y1 + 5, (x1 + x2) / 2, (y1 + y2) / 2],
                 "mask_polys_img": [[[x1 + 10, y1 + 5], [(x1 + x2) / 2, y1 + 5], [(x1 + x2) / 2, (y1 + y2) / 2]]]})
    out, states = process_frame(dets, cam["H"], slots, cfg, (IMG_W, IMG_H))
    assert len(out) == 2
    assert states == {"A01": 1, "A02": 0, "A03": 1, "A04": 0}
