# 주차장 여석 판별 2D 맵 (컴퓨터비전 3조)

설계 기준은 [CLAUDE.md](CLAUDE.md) 참고.

## 설치
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

데이터(`data/`)와 결과(`outputs/`)는 저장소에 넣지 않는다. 영상은 공유 드라이브에서 받는다.
