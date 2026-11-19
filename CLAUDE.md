# CLAUDE.md — 주차장 여석 판별 2D 맵 구축 (컴퓨터비전 팀 프로젝트 3조)

이 파일은 Claude Code가 이 저장소에서 작업할 때 따르는 설계 기준서다.
설계를 바꾸면 이 파일도 함께 수정한다.

---

## 1. 프로젝트 한 줄 요약

삼각대로 고정해 **미리 촬영한** 교내 주차장 영상에서 차량을 검출하고, 차가 바닥에 닿는 점(지면 접점)을
Homography로 **위에서 본 2D 맵**에 옮겨 **칸별 점유 상태를 판별**하고, 원본 영상과 2D 맵을 나란히 보여준다.

- 핵심 아이디어: 논문 ③(Carrera García et al., 2022)의 "위에서 본 평면에서 칸과 차의 겹침으로 판정"을
  **수직 카메라 없이**, 비스듬히 찍은 실제 주차장 영상에서 동작하게 만든다.
  - 바닥(주차선)은 Homography로 탑뷰로 편다.
  - 차는 높이가 있으므로 마스크 전체가 아니라 **지면 접점만** 맵으로 옮긴다.
- 비교 기준(베이스라인): 논문 ②(Alshareef et al., 2025) 방식 = 영상 위 칸 폴리곤 + 박스 중심점 판정.

## 2. 단계별 목표

| 단계 | 목표 | 판정 방법 | 맵 색상 |
|---|---|---|---|
| **1차 (핵심)** | 주차칸 점유 판별 → 2D 맵 구축 | 지면 접점(점 1개)을 맵에 투영 → 어느 칸 폴리곤 안에 있는지 | 빈칸 초록 / 점유 빨강 |
| 2차 (확장) | 주차칸 침범 판별 | 차 바닥 영역(footprint) 추정 → 칸별 겹침 면적 비율 | 침범 노랑 |
| 3차 (확장) | 불법주차 탐지 | 추적(ByteTrack) + 통로·금지구역 안 정지 시간 | 불법주차 주황 |
| 공통 | 안 보이는 칸 처리 | 화면 밖 · 가려진 칸은 추측하지 않음 | 판단불가 회색 |

**작업 순서: 1차를 끝까지(평가 포함) 완성한 뒤에 2차, 3차로 넘어간다.**

일정: 코드 완성(동결) 2026-12-05, 최종 발표 2026-12-16.

## 3. 입력 데이터 조건 (고정된 사실)

- 영상은 이미 촬영되어 있거나 이 조건으로 촬영된다. **실시간 스트림이 아니다.**
- 스마트폰을 **삼각대로 고정**해서 촬영 → 한 영상 안에서는 카메라가 움직이지 않는다.
- 영상 단위: **30fps, 약 5분**.
- 주차장마다 기본 촬영: **빈 상태(이른 아침) 1개 + 만차 1개**. 가능하면 중간 상태 1개, 2·3차용 연출 장면 1개를 추가한다.
- 같은 주차장이라도 촬영 날짜마다 카메라 위치가 다를 수 있다.
  같은 날 같은 삼각대 위치에서 찍은 영상들은 **하나의 카메라 설치(camera setup)**로 묶는다.
- 노트북에서 오프라인으로 처리한다 → 실시간 제약이 없다. 정확도를 우선한다.

## 4. 데이터 구조와 용어

| 용어 | 뜻 | 개수 |
|---|---|---|
| lot | 주차장 | 2~3곳 |
| map | 주차장의 2D 맵 (칸·통로·랜드마크 좌표) | **주차장당 1개** (고정) |
| cam | 카메라 설치 = 삼각대 한 위치 | 촬영 위치마다 1개 |
| H | cam의 영상 좌표 → map 좌표 Homography (3×3) | **cam당 1개** |
| clip | 영상 파일 1개 (약 5분) | cam당 여러 개 (빈 상태, 만차 등) |

- 같은 cam의 모든 clip은 같은 H를 공유한다. 빈 상태 clip에서 H를 구하고, 만차 clip에 재사용한다.
  (만차 상태에서는 주차선이 가려져 대응점을 찍기 어렵기 때문)
- 카메라 위치가 다른 영상이 들어오면 map은 그대로 두고 **새 cam을 만들어 H만 새로 구한다.**

```
data/                                   # .gitignore 대상 (영상은 공유 드라이브)
  lots/<lot>/map.json                   # 칸·통로·금지구역·랜드마크 (맵 좌표, m)
  lots/<lot>/map_bev.png                # 탑뷰 배경 이미지 (lot 첫 cam의 기준 프레임을 편 것. 그 cam이 맵 좌표계·탑뷰 범위를 정함)
  lots/<lot>/map_check.png              # 탑뷰 위에 칸을 그린 검수 그림
  cams/<lot>_<cam>/H.json               # 대응점, H, 재투영 오차, (선택) 바닥 가시 영역 ROI
  cams/<lot>_<cam>/points.json          # setup_cam 클릭 진행 상황 (자동 저장, 이어서 하기용)
  cams/<lot>_<cam>/ref_frame.png        # H 계산에 쓴 기준 프레임 (빈 상태)
  cams/<lot>_<cam>/bev.png              # 이 cam의 기준 프레임을 lot 탑뷰 범위로 편 것 (cam 간 정렬 확인용)
  cams/<lot>_<cam>/map_overlay.png      # 맵 칸을 H^-1로 기준 프레임에 되돌려 그린 검수 그림
  clips/<clip_id>/video.mp4             # clip_id 예: lotA_cam1_empty, lotA_cam1_full
  clips/<clip_id>/meta.json             # {"lot": "lotA", "cam": "lotA_cam1", "condition": "empty|mid|full|staged", "valid_sec": [시작, 끝]}
                                        # valid_sec: 카메라가 고정된 구간 (촬영 시작·끝의 흔들림 제외)
  clips/<clip_id>/gt_occupancy.csv      # 평가용 정답
  clips/<clip_id>/gt_events.csv         # 3차용 이벤트 정답
outputs/<clip_id>/                      # 실행 결과 (영상, CSV, 캐시)
```

## 5. 처리 파이프라인

### 5.1 설정 단계 (주차장·카메라 위치마다 1회, 사람이 클릭하는 도구 사용)

1. **(선택) 카메라 캘리브레이션**: 체커보드로 내부 파라미터·왜곡 계수를 구해 프레임을 undistort.
   광각(0.5배) 렌즈를 썼다면 필수. 1배 렌즈면 생략 가능 (효과를 비교해 결정).
   - **결과 (2026-11-19, 촬영 폰 0.5배, 1920x1080)**: 화면 커버리지 86% 체커보드로 측정한 왜곡이
     영상 가장자리에서도 최대 약 4.5px (폰이 내부에서 이미 보정함) → **undistort 생략 (`H.json`의 `undistort: false`)**.
     결과는 `data/calib/phone_0.5x/calib.json` (fx≈936, 가로 화각≈91°). 폰·렌즈·해상도를 바꾸면 다시 측정한다.
   - 체커보드는 화면 **네 모서리·가장자리까지** 덮어야 한다. 가운데만 덮으면 가장자리 보정이 발산한다
     (`calibrate.py`가 가장자리 커버리지를 출력·경고).
2. **H 계산** (`tools/setup_cam.py`):
   - 빈 상태 clip에서 기준 프레임 1장을 뽑는다.
   - 선이 잘 보이는 주차칸 모서리를 **6개 이상** 클릭하고, 각 점의 맵 좌표를 터미널에 입력한다.
     - 먼저 가까운 한 줄(x = 선마다 2.5m, y = 0/5)로 첫 H를 구하고, 1m 격자를 겹쳐 확인한다.
       다른 줄의 점은 화면에 나오는 커서의 맵 좌표(현재 H 기준)를 표준값(칸 5.0m, 직각주차 통로 6.0m)으로 반올림해 추가한다.
     - 지붕·나무에 가린 바닥은 (선택) ROI 다각형으로 빼면 가시 영역 계산에 쓰인다 (`ground_roi_img`, 없으면 영상 네 모서리).
     - 맵 좌표는 **표준 주차칸 크기(기본 폭 2.5m × 길이 5.0m, configs에서 변경 가능)**를 가정해 정한다.
       줄자 실측은 필수가 아니며, 2~3칸을 재서 표준과 다르면 configs 값을 고친다.
     - 점은 화면 양 끝에 넓게 분포시킨다.
   - `cv2.findHomography(..., cv2.RANSAC)`로 H 계산.
   - H 계산에 쓰지 않은 검증점 2~4개로 **재투영 오차(m)**를 계산해 H.json에 저장.
3. **탑뷰 배경 생성**: 기준 프레임을 `cv2.warpPerspective`로 펴서 `map_bev.png` 저장 (주차선이 직사각형 격자로 보여야 정상).
4. **2D 맵 칸 정의** (`tools/build_map.py`), 다음 중 하나 또는 조합:
   - (a) 규칙적인 줄은 **코드로 생성**: 시작 모서리 + 칸 폭·길이 + 개수 + 방향.
   - (b) 탑뷰 위에서 **직접 클릭**으로 폴리곤 입력 (불규칙한 칸, 통로, 금지구역).
   - (a)는 3번 클릭(첫 칸 앞 모서리 → 줄 끝 → 첫 칸 뒤 모서리) + 칸 수. 칸 폭은 표준값이 아니라 클릭한 줄 길이 / 칸 수
     (실제 선에 맞춤), 옆변이 줄과 수직이 아니면 평행사변형(사선 주차).
   - (c) (확장) 논문 ③ 방식 자동 검출: 탑뷰에 미디언 필터 → Otsu → Canny → 윤곽 → RDP 다각형 근사 → 사람이 검수.
   - 결과를 탑뷰 위에 겹쳐 그려 실제 주차선과 맞는지 눈으로 확인한다.

### 5.2 영상 처리 단계 (clip마다)

```
clip → 프레임 샘플링(기본 2fps, 3차는 10fps 이상)
     → [원본 프레임] undistort(선택) → YOLO-seg 검출 (car, bus, truck)
     → (3차) ByteTrack 추적
     → 지면 접점 추출 → H로 맵 좌표 변환
     → 맵 위에서 칸별 판정 → 가시 영역 처리 → 시간 보정
     → 렌더링(Split-View) + 칸 상태 CSV
```

- **YOLO는 원본(비스듬한) 프레임에 돌린다.** 탑뷰로 편 이미지에서 검출하지 않는다 (차가 늘어나고 흐려져 검출 성능이 떨어짐).
- 이미지 전체를 펴지 않고 **점 좌표만** `cv2.perspectiveTransform`으로 변환한다.
- 무거운 검출 결과는 JSONL로 캐시하고, 판정·평가는 캐시를 읽어 반복 실행한다.

### 5.3 검출 모델

- Ultralytics **최신 안정 버전의 segmentation 모델**(예: YOLO11-seg 계열)을 **큰 크기(l 또는 x)**로 사용. 모델 이름은 configs에서 지정.
- COCO 사전학습 가중치를 그대로 사용 → **학습용 데이터셋 구축 불필요.**
  - 사용 클래스: car(2), bus(5), truck(7). SUV·승합차가 truck으로 잡히는 경우가 많다.
- 먼 차 검출을 위해 입력 해상도(`imgsz`)를 크게 (예: 1280) 설정할 수 있게 한다.
- **현재 선택 (2026-11-19, lotA_cam1_full 프레임 3장 비교)**: `yolo26x-seg`, `imgsz=1920`.
  yolo11x-seg는 1920에서 지붕 구조물을 truck으로 오검출. 1280 대비 1920에서 먼 차 신뢰도 0.5~0.6 → 0.8~0.9.
  처리 속도 약 0.33초/검출 프레임 (RTX 4060, 영상 디코딩 포함; 추론만 약 0.12초).
- 우리 영상에서 미검출이 많을 때만 파인튜닝을 검토한다 (기본 계획에는 없음).

### 5.4 지면 접점 (1차의 핵심)

- 이유: H는 **바닥 평면 위의 점만** 정확히 옮긴다. 지붕·박스 중심처럼 높이가 있는 점은 카메라에서 먼 쪽으로 밀려나 뒤 칸에 찍힌다.
- 방법: 마스크의 **아래쪽 경계(바닥에 닿는 쪽)**에서 접점을 구한다.
  - 기본: 마스크 아래 경계의 가운데 점 (`contact_mode: mask_bottom_center`)
  - 비교용: 박스 아래변 중앙 (`bbox_bottom_center`), 박스 중심 (`bbox_center`)
- 화면 경계에 걸려 잘린 검출은 `truncated=true`로 표시하고 판정 신뢰도를 낮춘다.
- 구현: 마스크의 **볼록 껍질** 아래 경계에서 가로 중앙 점을 쓴다 (가로등·나무가 마스크를 파고들어도 접점이 위로 튀지 않음).
- 합성 실험 결과 (`tests/test_contact.py`):
  - 옆면이 카메라를 향한 차: 박스 중심은 뒤 칸으로 밀려 틀리고, 지면 접점은 맞는 칸 → 제안 방식의 근거.
  - 정면이 카메라를 향한 차: 박스 중심도 같은 칸 안. 지면 접점은 차의 **카메라 쪽 끝**이라 칸 앞 경계에서 0.15m 안쪽에 붙음 → H 오차에 취약.
  - 보완 옵션 `contact.depth_offset_m`: 접점을 맵에서 카메라 반대쪽으로 민다. 0.9m(차 폭 절반)면 차 방향과 무관하게
    차 바닥 영역 안쪽에 들어간다. **기본값 0 (원래 설계), 0.9 채택 여부는 실제 영상 비교 후 결정.**

### 5.5 판정 규칙

**1차 점유**
- 지면 접점(맵 좌표)이 들어간 칸 → 점유(1). 차가 없는 칸 → 빈칸(0).
- 접점이 칸 경계 근처면 가장 가까운 칸에 배정하되, 경계 허용 거리는 configs 값.

**2차 침범 (1차 완료 후)**
- 차 바닥 영역(footprint) 추정:
  - 보이는 부분(마스크 아래 경계를 맵에 투영한 선분)은 측정값을 쓴다.
  - 안 보이는 부분(카메라 반대쪽)은 **칸 방향으로 정렬된 표준 차 크기(기본 1.8m × 4.7m)** 사각형으로 채운다.
  - 차 방향은 가장 가까운 칸의 방향으로 가정한다.
- 칸별 겹침 면적 비율 계산 (shapely). 한 차가 두 칸과 각각 τ_straddle 이상 겹치면 두 칸 모두 침범(2).
- 좌우(옆 칸 방향) 위치는 비교적 정확하고 앞뒤(깊이) 방향은 오차가 크다는 점을 고려한다.

**3차 불법주차 (2차 이후)**
- ByteTrack ID별 맵 좌표 궤적을 유지.
- 통로·금지구역 폴리곤 안에서 이동 속도가 v_stop(m/s) 미만인 상태가 T_stop(초) 이상 지속되면 불법주차 이벤트.
- 지나가는 차, 주차하려고 잠깐 멈춘 차는 이벤트가 아니어야 한다.

**판단불가 (-1)**
- 영상 네 모서리를 H로 투영한 다각형 = 맵 위 **가시 영역**.
  영상 윗부분이 지평선 너머(하늘)면 투영이 무한대로 발산하므로, 지평선 아래만 남기고 맵 범위와 교집합을 취한다.
- 칸의 가시 비율 ≥ 0.7 → 판정, 0.3~0.7 → 불확실(판정하되 표시), < 0.3 → 판단불가.
- (확장) 앞차에 가려진 칸 처리.
- 결과에 **커버리지(관측 가능한 칸 / 전체 칸)**를 함께 기록.

**시간 보정**
- 칸 상태는 최근 N프레임(기본 5) 다수결로 결정 → 깜빡임 제거.

### 5.6 출력

- **Split-View 영상**: 왼쪽 원본 프레임 + 칸 폴리곤(맵 좌표를 **H의 역행렬**로 영상에 되돌려 그림) + 검출 결과,
  오른쪽 `map_bev.png` 위에 칸 상태 색칠 + 빈자리 수.
- **칸 상태 CSV**: `frame_idx,time_sec,slot_id,state` (평가에 사용).

## 6. 베이스라인 (논문 ② 방식)

- 칸 폴리곤을 **영상 좌표**로 사용한다. 공정한 비교를 위해 같은 map의 칸을 H^-1로 영상에 투영해 얻는다.
- 판정: **박스 중심점**이 영상 위 칸 폴리곤 안에 있으면 점유.
- 같은 clip, 같은 GT로 제안 방식과 지표를 비교한다.

## 7. 우리가 만들어야 하는 데이터 (학습 데이터는 없음)

| 데이터 | 용도 | 필수 |
|---|---|---|
| map.json | 칸·통로 위치 | 필수 |
| H.json | 영상 ↔ 맵 변환 | 필수 |
| gt_occupancy.csv | 1차·2차 평가 정답 | 필수 |
| gt_events.csv | 3차 평가 정답 | 3차 할 때 |
| 파인튜닝 라벨 | 검출 보강 | 선택 |

**gt_occupancy.csv**
```
clip_id,frame_idx,slot_id,state
lotA_cam1_full,120,A01,1
```
- state: 0 빈칸, 1 점유, 2 침범(2차), -1 판단불가
- clip마다 5~10초 간격으로 50~100프레임 샘플링해 라벨링 (`tools/label_gt.py`).
- Test용 clip은 시스템 예측을 보지 않은 상태로 라벨링한다.

**gt_events.csv**
```
event_id,clip_id,type,zone_id,start_sec,end_sec
E01,lotA_cam1_staged,aisle_parking,AISLE1,42.0,135.5
E02,lotA_cam1_staged,pass_through,AISLE1,150.0,158.0
```
- type: `aisle_parking`(불법주차) | `pass_through`(정상 통과, 오경보 평가용)

## 8. 평가 지표

| 대상 | 지표 | 이유 |
|---|---|---|
| 1차 점유 (핵심) | 혼동행렬, **빈칸 Precision**, 빈칸·점유 Recall, F1 | "점유된 칸을 빈칸으로 안내"하는 실수가 가장 치명적. 빈 상태·만차 영상은 클래스가 치우쳐 Accuracy만으로는 부족 |
| 1차 점유 | Accuracy (보조), **빈자리 수 MAE** | 사용자가 보는 숫자의 정확도 |
| 1차 점유 | 조건별(빈/중간/만차), 거리별(앞줄/뒷줄) 분리 보고 | 어디서 약한지 파악 |
| 가시성 | 커버리지, 판단불가 처리의 정확성 | GT가 -1인 칸은 정확도 계산에서 제외 |
| 기하 | **재투영 오차 (m, 칸 폭 대비 %)** | 모든 판정의 기반인 맵 변환의 정확도 |
| 비교 | 베이스라인(논문 ②) vs 제안 방식, 접점 방식별, 시간 보정 유무, 모델 크기별 | 제안 방식의 효과를 숫자로 증명 |
| 2차 침범 | 침범 칸 Precision, Recall | 드문 사례라 두 지표를 분리 |
| 3차 불법주차 | 이벤트 Precision, Recall, 오경보 수, 탐지 지연(초) | 시간에 걸친 사건 단위 평가 |
| 실용성 | 처리 속도 (FPS, 프레임당 ms) | 논문 ③(장당 약 1.5초)과 비교 |

- 임계값(τ, N, T_stop 등)은 **튜닝용 clip**에서만 조정하고, **테스트용 clip**은 최종 보고에만 사용한다 (clip 단위 분리).

## 9. 디렉터리 구조

```
CV_Project/                   # 저장소 루트
  CLAUDE.md
  README.md                   # 설치, 실행 방법, 결과 예시
  requirements.txt            # 버전 고정
  pyproject.toml              # src 레이아웃 패키지(pip install -e .), pytest 설정
  configs/default.yaml        # 모델, imgsz, conf, 클래스, 샘플링 fps, 칸·차 표준 크기, 임계값
  src/parking/
    io.py                     # map/H/meta/config 로드·저장, 좌표계 변환 유틸
    calibrate.py              # (선택) 체커보드 캘리브레이션, undistort
    homography.py             # H 계산, 재투영 오차, 점·폴리곤 변환, 가시 영역, warp
    map_builder.py            # 칸 줄 생성, (확장) 탑뷰 자동 칸 검출
    detect.py                 # YOLO-seg 추론, 결과 JSONL 캐시
    contact.py                # 지면 접점 추출 (mask_bottom_center 등)
    track.py                  # (3차) ByteTrack 래퍼
    occupancy.py              # 1차 판정, 가시성, 시간 보정
    footprint.py              # (2차) 바닥 영역 추정, 겹침 비율
    illegal.py                # (3차) 정지 시간 기반 이벤트
    baseline.py               # 논문 ② 방식
    render.py                 # Split-View 영상, 2D 맵 그리기
    pipeline.py               # clip 단위 전체 실행
  tools/
    setup_cam.py              # 기준 프레임에서 대응점 클릭 → H.json, ref_frame.png, bev.png (lot 첫 cam이면 map.json 뼈대·map_bev.png)
    build_map.py              # 탑뷰 위 칸 생성·클릭·검수 → map.json
    sample_frames.py          # GT용 프레임 샘플링
    label_gt.py               # 칸 클릭으로 0→1→2→-1 순환, CSV 저장
  eval/
    evaluate.py               # 지표 계산, 혼동행렬, 조건·거리별 표
    compare.py                # 베이스라인 vs 제안, ablation 표
  tests/                      # pytest: homography, contact, occupancy, footprint
  notebooks/                  # 실험용
```

## 10. 코딩 규칙

- Python 3.10+, 타입 힌트, docstring (한국어 가능).
- 좌표계 혼동 방지: 변수명 접미사 `_img`(영상 픽셀, undistort 후), `_map`(맵 좌표, m), `_bev`(탑뷰 이미지 픽셀).
- OpenCV 점 배열은 `np.float32`, shape `(N,1,2)`. 변환 유틸은 `homography.py`에 모은다.
- 폴리곤 연산은 `shapely`.
- 임계값·크기·모델명은 하드코딩하지 않고 `configs/default.yaml`에서 읽는다.
- 모든 진입점은 `argparse` CLI.
- 로직 함수(homography, contact, occupancy, footprint, illegal)는 **합성 데이터로 pytest 단위 테스트**를 작성한다.
  실제 영상이 없어도 개발할 수 있게 한다.
- 데이터, 영상, 모델 가중치, 결과물은 커밋하지 않는다.
- GUI 도구(setup_cam, build_map, label_gt)는 로컬 Ubuntu 데스크톱의 OpenCV 창에서 실행된다고 가정한다.
  키 조작법을 화면에 표시하고, 되돌리기(u)와 저장(s)을 지원한다.

## 11. 데이터 포맷 예시

**map.json**
```json
{
  "lot_id": "lotA",
  "units": "m",
  "slot_size": [2.5, 5.0],
  "bev_image": "map_bev.png",
  "bev_px_per_m": 20,
  "bev_origin_m": [-2.0, -2.0],
  "bev_size_px": [1200, 800],
  "bev_cam": "lotA_cam3",
  "landmarks": [],
  "slots": [{"id": "A01", "row": "A", "polygon": [[0,0],[2.5,0],[2.5,5],[0,5]]}],
  "aisles": [{"id": "AISLE1", "polygon": [[0,5],[30,5],[30,11],[0,11]]}],
  "no_parking": []
}
```

**H.json**
```json
{
  "cam_id": "lotA_cam1",
  "lot_id": "lotA",
  "ref_clip": "lotA_cam1_empty",
  "ref_frame_idx": 0,
  "undistort": false,
  "image_size": [1920, 1080],
  "correspondences": [{"img": [812.0, 433.0], "map": [0.0, 0.0]}],
  "validation":      [{"img": [950.0, 610.0], "map": [7.5, 5.0]}],
  "ground_roi_img": null,
  "H": [[0,0,0],[0,0,0],[0,0,1]],
  "fit_error_m": {"mean": 0.0, "max": 0.0},
  "reproj_error_m": {"mean": 0.0, "max": 0.0}
}
```

**검출 캐시 (프레임별 JSONL)**
```json
{"frame_idx": 120, "time_sec": 4.0, "dets": [
  {"track_id": 7, "cls": 2, "conf": 0.88, "bbox_img": [x1,y1,x2,y2],
   "mask_polys_img": [[[x,y], ...], ...], "contact_img": [x,y], "contact_map": [X,Y], "truncated": false}
]}
```
- 마스크는 RLE 대신 **영상 픽셀 다각형 목록**으로 저장한다 (가로등·나무에 가려 조각나면 조각마다 1개).
  Ultralytics `masks.xy`는 조각들을 한 다각형으로 이어 붙여 모양이 깨지므로 쓰지 않는다 (`detect.mask_to_polys`).
- 프레임은 항상 처음부터 순서대로 읽어 번호를 센다 (seek 금지: 폰 영상에서 번호가 어긋날 수 있음).

## 12. 작업 순서 (Claude Code에 단계별로 요청)

1. 저장소 뼈대: 디렉터리, requirements.txt, .gitignore, configs/default.yaml, 빈 함수 시그니처
2a. `homography.py` + 합성 데이터 pytest
3. `tools/setup_cam.py`, `tools/build_map.py` (대응점 클릭 → H → 탑뷰 → 칸 정의).
   0.5배 렌즈 영상이면 그 전에 `calibrate.py`로 체커보드 캘리브레이션
2b. `contact.py`, `occupancy.py` + 합성 데이터 pytest
   (실제 영상으로 2D 맵을 먼저 확인하려고 2단계를 둘로 나눔)
4. `detect.py` (YOLO-seg, 캐시) → `pipeline.py` + `render.py`로 **1차 Split-View 결과 영상**
5. `tools/sample_frames.py`, `tools/label_gt.py`, `eval/evaluate.py`, `baseline.py`, `eval/compare.py` → **1차 정량 평가**
6. (2차) `footprint.py` + 침범 평가
7. (3차) `track.py`, `illegal.py` + 이벤트 평가
8. README에 실행 방법과 결과 정리

각 단계가 끝나면 실제로 실행해 확인한 뒤 커밋한다.

## 13. 자주 쓰는 명령 (목표 형태)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python tools/setup_cam.py --clip data/clips/lotA_cam3_mid        # lot·cam은 meta.json에서 읽음
python tools/build_map.py --lot lotA --cam lotA_cam3
python -m parking.pipeline --clip data/clips/lotA_cam1_full --config configs/default.yaml
python tools/sample_frames.py --clip data/clips/lotA_cam1_full --every-sec 5
python tools/label_gt.py --clip data/clips/lotA_cam1_full
python eval/evaluate.py --clips data/clips/lotA_* --split test
python eval/compare.py --clips data/clips/lotA_* --split test
pytest -q
```

## 14. 작업할 때 주의

- 큰 변경 전에는 계획을 먼저 제시하고 확인받는다.
- 이 문서의 설계(맵 고정 + cam별 H, 원본 프레임 검출, 지면 접점 투영, 맵 위 판정)를 임의로 바꾸지 않는다.
  바꿀 필요가 있으면 이유를 설명하고 이 파일을 함께 수정한다.
- 실제 데이터가 없으면 합성 데이터(가짜 map, 알려진 H, 가짜 검출)로 먼저 검증한다.
- 결과 수치를 보고할 때는 어떤 clip(튜닝/테스트)과 어떤 설정으로 얻었는지 함께 적는다.
