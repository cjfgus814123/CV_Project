import numpy as np
from shapely.geometry import Polygon

from parking.map_builder import slot_row


def test_rect_row():
    s = slot_row((0, 0), (10, 0), (0, 5), 4, "A")
    assert [x["id"] for x in s] == ["A01", "A02", "A03", "A04"]
    assert np.allclose(s[1]["polygon"], [[2.5, 0], [5, 0], [5, 5], [2.5, 5]])
    assert all(abs(Polygon(x["polygon"]).area - 12.5) < 1e-9 for x in s)


def test_rotated_slanted_row_tiles_without_overlap():
    s = slot_row((3, 1), (3 + 8 * np.cos(0.5), 1 + 8 * np.sin(0.5)), (1, 5), 3, "B", start=7)
    assert s[0]["id"] == "B07"
    polys = [Polygon(x["polygon"]) for x in s]
    assert all(p.is_valid for p in polys)
    assert polys[0].intersection(polys[1]).area < 1e-9 and polys[0].touches(polys[1])
