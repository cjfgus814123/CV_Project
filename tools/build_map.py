"""탑뷰 위 칸 생성·클릭·검수 → map.json. 키: u 되돌리기, s 저장."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lot", required=True)
    ap.add_argument("--cam", required=True)
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
