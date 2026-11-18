"""베이스라인 vs 제안 방식, ablation 표."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--split", choices=["tune", "test"], required=True)
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
