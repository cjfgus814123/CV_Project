"""칸 클릭으로 0→1→2→-1 순환, gt_occupancy.csv 저장. 키: u 되돌리기, s 저장."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
