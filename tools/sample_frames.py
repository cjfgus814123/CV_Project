"""GT 라벨링용 프레임 샘플링."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--every-sec", type=float, default=5.0)
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
