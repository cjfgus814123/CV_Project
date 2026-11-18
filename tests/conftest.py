"""pytest 공용 fixture. 합성 카메라·칸·차는 parking.synthetic에 있다."""
import pytest

from parking.synthetic import IMG_H, IMG_W, car_mask_poly, make_camera, project, slot_rows  # noqa: F401


@pytest.fixture
def cam():
    return make_camera()
