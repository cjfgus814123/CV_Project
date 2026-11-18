"""clip 단위 전체 실행: python -m parking.pipeline --clip ... --config ..."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="data/clips/<clip_id>")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs")
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
