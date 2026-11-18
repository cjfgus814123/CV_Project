"""기준 프레임에서 대응점 클릭 → H.json, ref_frame.png, map_bev.png. 키: u 되돌리기, s 저장."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="빈 상태 clip 디렉터리")
    ap.add_argument("--lot", required=True)
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
