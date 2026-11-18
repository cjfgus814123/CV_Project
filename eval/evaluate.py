"""지표 계산: 혼동행렬, 빈칸 Precision, Recall, F1, 빈자리 수 MAE, 조건·거리별 표."""
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
