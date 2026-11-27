# GT 라벨링 가이드 (팀원용)

git clone을 받은 상태에서 시작합니다. 공유 드라이브 `CV_GT/`에서 파일을 받아 라벨링하고, 결과를 다시 `CV_GT/`에 올립니다.

```
CV_GT/
  0_안내/       이 문서, 진행표
  1_꾸러미/     라벨링할 사진 묶음 (zip 23개)      → 받기
  2_완료/       라벨링 결과                        ← 올리기
  3_불법주차/   확인용 영상, 입력 양식              → 받기
  4_결과물/     (열지 않기: 채점 후 공지)
```

**주의**: `4_결과물/` 폴더와 결과 영상은 채점 완료 공지 전에는 보지 않습니다. git push는 하지 않습니다.

---

## 1. 설치 (처음 한 번)

```bash
cd CV_Project
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy shapely pyyaml
pip install -e . --no-deps
```

터미널을 새로 열 때마다 `cd CV_Project && source .venv/bin/activate`를 먼저 실행합니다.

---

## 2. 칸 라벨링 (조각 23개, 각각 반복)

### 2.1 Drive에서 받기

`1_꾸러미/`의 zip 23개를 받습니다. 아래 순서대로 하면 됩니다.

| 순서 | zip 파일 | 사진 수 |
|---|---|---|
| 1 | `lotA_cam6_a_label.zip` | 32 |
| 2 | `lotA_cam5_a_label.zip` | 42 |
| 3 | `lotA_cam4_a_label.zip` | 97 |
| 4 | `lotA_cam7_t00_label.zip` | 121 |
| 5~11 | `lotA_cam7_t01_label.zip` ~ `lotA_cam7_t07_label.zip` | 각 16~25 |
| 12~23 | `lotA_cam8_t00_label.zip` ~ `lotA_cam8_t11_label.zip` (11/27 추가) | 각 15~65 (총 410) |

### 2.2 압축 풀기

**CV_Project 폴더 안에서** 풉니다.

```bash
unzip ~/Downloads/lotA_cam6_a_label.zip -d ~/CV_Project
```

### 2.3 라벨링

```bash
python tools/label_gt.py --clip data/clips/lotA_cam6_a
```

사진 위에 칸이 색으로 칠해진 창이 열립니다.

| 조작 | 동작 |
|---|---|
| 칸 클릭 | 상태 바꾸기: 초록 0 → 빨강 1 → 노랑 2 → 회색 -1 → 초록 0 |
| `d` / → | 다음 사진 |
| `a` / ← | 이전 사진 |
| `u` | 되돌리기 |
| `m` | 칸 색 숨기기/보이기 |
| `s` | 저장 |
| `q` | 저장하고 종료 |

1. **첫 사진**은 모든 칸을 맞는 상태로 클릭합니다.
2. `d`로 넘기면 **직전 사진의 라벨이 복사**되어 있으니 **바뀐 칸만** 클릭합니다.
3. **마지막 사진까지 `d`로 넘겨야** 전부 저장됩니다.
4. 중간에 `q`로 끄고 같은 명령으로 다시 열면 이어서 할 수 있습니다.
5. 칸 안쪽 바닥(칸 이름 근처)을 클릭합니다. 차 지붕을 클릭하면 뒤 칸이 눌릴 수 있습니다.

**상태 기준**

| 상태 | 색 | 언제 |
|---|---|---|
| 0 빈칸 | 초록 | 칸에 차가 없다 |
| 1 점유 | 빨강 | 차가 칸 안에 있다 (선을 밟지 않음, 조금 비뚤어도 1) |
| 2 침범 | 노랑 | 차가 선을 밟거나 넘어 두 칸에 걸쳤다 → **걸친 두 칸 모두 2** |
| -1 판단불가 | 회색 | 가려져서 차가 있는지 없는지 알 수 없다 |

- 바닥은 가려졌지만 지붕이 보여 차가 확실하면 1
- 들어오는/나가는 중인 차: 그 사진에서 칸 안이면 1, 선에 걸치면 2, 아직 통로면 칸은 0
- 통로를 지나가는 차, 사람, 오토바이는 무시

### 2.4 Drive에 올리기

저장된 `data/clips/<조각>/gt_occupancy.csv`의 이름을 `<조각>__<내이름>.csv`로 바꿔 **`2_완료/`**에 올립니다 (밑줄 2개).

```bash
cp data/clips/lotA_cam6_a/gt_occupancy.csv ~/lotA_cam6_a__홍길동.csv
```

`0_안내/진행표`에서 그 조각을 **완료**로 바꿉니다. 다음 zip으로 2.2부터 반복합니다.

---

## 3. 불법주차 시각 (`lotA_cam7_t00` 하나만)

### 3.1 Drive에서 받기

`3_불법주차/`의 두 파일:
- `lotA_cam7_t00_events_check.mp4` (통로 주황 테두리, 왼쪽 위에 경과 시간 `t = …s`)
- `gt_events_template.csv` (입력 양식)

### 3.2 입력

영상을 보며 통로(AISLE1·AISLE2)에서 차가 멈춘 일을 양식에 한 줄씩 적습니다 (Google 시트로 열어서 적어도 됨).

```
event_id,clip_id,type,zone_id,start_sec,end_sec
E01,lotA_cam7_t00,aisle_parking,AISLE2,42.0,135.5
E02,lotA_cam7_t00,pass_through,AISLE2,150.0,158.0
```

| 열 | 적는 값 |
|---|---|
| type | `aisle_parking` = 통로에 차를 세움 (연출 장면) / `pass_through` = 지나가거나 잠깐 멈췄다 감 |
| zone_id | 영상에 적힌 통로 이름 `AISLE1` 또는 `AISLE2` |
| start_sec | 바퀴가 완전히 멈춘 순간의 영상 `t` (0.5초 단위면 충분) |
| end_sec | 다시 움직이기 시작한 순간의 영상 `t` |

### 3.3 Drive에 올리기

파일 이름을 **`gt_events.csv`**로 해서 **`2_완료/`**에 올립니다 (Google 시트로 적었다면 파일 → 다운로드 → CSV).

---

## 4. 다 끝나면 `2_완료/`에 있어야 할 파일 (24개)

```
lotA_cam4_a__<내이름>.csv
lotA_cam5_a__<내이름>.csv
lotA_cam6_a__<내이름>.csv
lotA_cam7_t00__<내이름>.csv ~ lotA_cam7_t07__<내이름>.csv   (8개)
lotA_cam8_t00__<내이름>.csv ~ lotA_cam8_t11__<내이름>.csv   (12개)
gt_events.csv
```

---

## 문제 해결

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: parking` | `source .venv/bin/activate` 후 `pip install -e . --no-deps` |
| `gt_frames가 없음` | zip을 CV_Project 폴더 안에서 풀었는지 확인 (`ls data/clips/`) |
| 칸이 사진의 주차선과 크게 어긋남 | 라벨링하지 말고 관리자에게 연락 |
| 실수로 많이 바꿈 | `u`를 여러 번 누른 뒤 `s` (창을 닫으면 그 상태로 저장됨) |

모르는 장면은 화면 위쪽의 사진 번호(`idx`)와 칸 이름을 캡처해서 물어봐 주세요.
