# 드론 비전 시스템(AeroSentinel) 고도화 기획서 & 로드맵

> **목적** — 광산/발파 현장 안전을 위한 드론 영상 객체 인식 시스템의 현재 상태(As-Is)를 정리하고, 기획·UI/UX·비즈니스 로직·모델 전략·향후 투입 계획을 **반기 단위 3단계 로드맵**으로 구조화한 KPI 관리 문서.

| 항목 | 내용 |
|------|------|
| 문서 버전 | v1.0 (초안) |
| 작성일 | 2026-05-28 |
| 제품/코드명 | AeroSentinel · Drone Vision Operations Console |
| 현재 시스템 버전 | v0.9.0 |
| 배포 목표 환경 | **온프레미스 로컬 GPU 데스크톱** (Electron + Python, 단독 구동) |
| 1순위 KPI | **오탐 최소화(FAR)** · **누락 최소화(Recall/mAP)** |
| 로드맵 범위 | Phase 1 (2026 H2) → Phase 2 (2027 H1) → Phase 3 (2027 H2~) |

---

## 1. 개요 & 배경

### 1.1 프로젝트 목적
드론으로 촬영한 광산/발파 현장 영상에서 **사람과 차량**을 자동 감지하여, **발파 직전 안전 스윕**과 **상시 모니터링**을 수행한다. 운영자는 데스크톱 콘솔에서 실시간 감지 결과·위험도·이벤트 아카이브를 확인한다.

### 1.2 도메인 특수성 — 왜 KPI가 중요한가
발파 안전 도메인은 **두 종류의 오류 비용이 모두 크다.**

- **누락(False Negative)** — 사람/차량을 놓치면 곧바로 **인명 안전사고**로 직결된다. → Recall 중요.
- **오탐(False Positive)** — 바위·그림자·장비 가장자리를 사람으로 오인하면 **불필요한 발파 중단·알람 피로·운영 신뢰도 저하**를 유발한다. → FAR(시간당 오경보) 중요.

따라서 단일 검출기 강화가 아니라 **검출–검증–시간일치 레이어드 구조**로 두 지표를 동시에 개선하는 것이 본 시스템의 핵심 설계 철학이다.

### 1.3 핵심 도메인 클래스
- `person`
- 차량군: `dump_truck`, `tanker_truck`, `truck` (+ `excavator`, `loader`, `bus`, `car`, `suv`, `van` → 내부적으로 `vehicle`로 정규화)

---

## 2. 현재 상태 (As-Is) — v0.9.0

### 2.1 시스템 아키텍처

```
┌─────────────────────────────┐         ZeroMQ REQ/REP          ┌──────────────────────────────┐
│   Electron UI (main.js)     │  ◄────  tcp://127.0.0.1:5555 ──►  │   Python AI Engine (engine.py) │
│   - index.html (콘솔)        │                                  │   - DetectionPipeline          │
│   - 폴더 감시 / 영상 선택      │   start_video / status /        │   - 4-thread VideoSession      │
│   - 실시간 상태 폴링          │   video_status / stop_video      │   - 5-Layer Safety Stack       │
└─────────────────────────────┘                                  └──────────────────────────────┘
```

- **프론트엔드**: Electron 데스크톱 앱. `main.js`가 Python 엔진을 자식 프로세스로 기동하고 ZeroMQ로 통신. 폴더 감시(신규 영상 자동 투입), 영상 파일 선택, 결과 폴더 열기 담당.
- **백엔드(AI 엔진)**: `engine.py`는 ZMQ 서버 + 세션 스레드만 담당. 모든 검출 로직은 `detection/` 패키지의 `DetectionPipeline`에 위임.
- **세션 4스레드 구조** (`VideoAnalysisSession`):
  1. `video_stream_worker` — 소스 영상에서 프레임 읽기
  2. `snapshot_scheduler` — N초마다 1프레임 샘플링
  3. `inference_worker` — 5-Layer 파이프라인 실행
  4. `result_handler` — 박스 렌더링·JPEG 인코딩·이벤트 저장

### 2.2 UI/UX 현황

다크 테마 "Vision Ops Console" (한국어 UI). 주요 구성:

| 영역 | 구성 요소 |
|------|-----------|
| **상단바** | 브랜드, 현장 선택기(Site Alpha · Sector 03), 엔진 상태 LED, Device/Profile/Layers/AL 배지, 위험도 배지, 운영자 칩 |
| **사이드바** | 네비게이션(라이브 분석 / 감지 피드 / 이벤트 아카이브), **5-Layer Safety Stack 칩**, 빌드 버전 |
| **미션 바** | 수동/자동 모드 토글(폴더 감시), 분석 간격 슬라이더(1~10초), 프로필 선택(fast/balanced/accurate), 분석 시작/중지 버튼, 감시 폴더 행 |
| **KPI 스트립** | 현재 사람 / 현재 차량 / 이벤트 / 스냅샷 / 분석 완료 / 원본 FPS / 추론 FPS (7개 카드) |
| **라이브 뷰어** | 영상 + HUD 오버레이(위험 배너·시간·카운트·FPS), 진행률 바, 위험 요약 |
| **우측 레일** | 캡처 스냅샷(원본/분석/분할 탭), 감지 객체 칩, **5-Layer 처리 시간 바**(L1+L2 / L3 VLM / L4 Track) |
| **이벤트 아카이브** | 썸네일 그리드, 저장 폴더 열기 |
| **공통** | 이미지 확대 모달, 토스트 알림 |

### 2.3 비즈니스 로직 (운영 플로우)

1. **영상 투입** — 수동(파일 선택) 또는 자동(폴더 감시: 파일 안정화 2초 확인 후 자동 분석, localStorage로 중복 방지).
2. **샘플링** — 기본 3초 간격(1~10초 조정)으로 스냅샷 캡처.
3. **추론** — 스냅샷마다 5-Layer 파이프라인 실행. 백로그가 쌓이면 **가장 최신 스냅샷만** 처리(실시간성 우선).
4. **위험도 판정** (`infer_risk`):
   - 사람 + 차량 동시 → **critical**
   - 사람만 → **warning**
   - 차량만 → **notice**
   - 없음 → **clear**
5. **이벤트 저장** — 사람/차량이 감지되면 즉시 `results/events/`에 주석 이미지 저장. (현재 정책: `require_confirmation_for_alarm = false` → N-of-M 확정 여부와 무관하게 감지 즉시 저장하고, 확정 트랙은 UI에 ✓ 배지로만 표시)
6. **운영 모드**:
   - **고정밀 모드**(간격 3~10초): 전 레이어 활성 → 발파 직전 안전 스윕
   - **고속 미리보기**(간격 ≤2초): L1만 사용 → 비행 중 실시간 모니터링

### 2.4 모델 스택 — 5-Layer Safety Stack (현재 사용 모델)

```
snapshot ─► L1 Detector ─► L2 SAHI 타일링 ─► L3 SigLIP 검증 ─► L4 ByteTrack(N-of-M) ─► 알람/이벤트
                                                                       │
                                                                       └► L5 Active Learning (재학습 시드)
```

| 레이어 | 역할 | **현재 투입 모델/기법** | 비고 |
|--------|------|------------------------|------|
| **L1 Detector** | 1차 객체 검출 | **YOLOv8l**(primary) + **RT-DETR-L**(secondary, 앙상블 NMS), **YOLOv8m**(fast 모드) | 클래스 별칭 정규화, 기하학(면적·종횡비) 필터, 클래스별 최소 점수 |
| **L2 SAHI** | 고해상도 타일링 추론 | SAHI (tile 768, overlap 0.25, max 36타일) | 작은 객체 Recall의 핵심. 모델 아님(타일링 + NMS) |
| **L3 VLM 검증** | 박스 crop zero-shot 검증 | **google/siglip-base-patch16-224** (FP16) | 광산 하드네거티브 프롬프트(바위/그림자/장비 가장자리)와 비교해 오탐 reject |
| **L4 Tracker** | 시간 일치(N-of-M) | **ByteTrack** (`supervision`) | confirm_window=5, confirm_min_hits=3. 1프레임 깜빡임 제거 |
| **L5 Active Learning** | 불확실 검출 수집 | (모델 아님) | confidence 0.30~0.60 + VLM reject + 미확정 트랙 → manifest로 저장 |

> **A/B 측정 가능 설계**: 각 레이어는 `configs/runtime_detector.json`에서 독립적으로 on/off 가능 → 레이어별 기여도(mAP·FAR)를 신뢰성 있게 측정할 수 있다. CPU 폴백 시 SAHI/VLM/secondary 자동 비활성.

### 2.5 데이터 / 학습 파이프라인 현황

| 단계 | 도구/스크립트 | 상태 |
|------|--------------|------|
| 자동 라벨링 | Grounding DINO + SAM (`scripts/auto_label_bootstrap.py`, `configs/auto_labeling.json`) | 가동 |
| 수동 검수 | CVAT / Label Studio | 외부 도구 |
| 검출기 파인튜닝 | `training/finetune_detector.py` (YOLO11m 기본 / RT-DETR 옵션) | 스크립트 준비 |
| 합성 데이터 | `training/synthetic_gen.py` (SD + ControlNet) | 기반 스크립트 |
| PPE 서브태스크 | `training/ppe_train.py` (YOLO11s/n, 안전모·조끼) | 스크립트 준비 |
| 자기지도 사전학습 | `training/pretrain_dinov2.py` (DINOv2) | 래퍼 스크립트(클러스터 필요) |
| TensorRT 변환 | `training/export_tensorrt.py` (INT8/FP16) | 스크립트 준비 |
| 하드네거티브 루프 | `scripts/generate_false_positive_review_set.py` | 가동 |

### 2.6 현재 식별된 개선 포인트 (Gap)

- **프로필 미연동**: UI의 `fast/balanced/accurate` 프로필이 엔진 파이프라인 전환에 실제로 연결되어 있지 않음(현재 분석 간격만 반영). → Phase 1.
- **자체 학습 모델 미투입**: L1이 아직 범용 COCO 가중치(YOLOv8l/RT-DETR-L) 기반. 현장 도메인 파인튜닝 모델 미적용. → Phase 1~2.
- **정량 베이스라인 부재**: mAP·FAR 측정용 검증셋과 수치가 없음. → Phase 1 최우선.
- **버전 표기 불일치**: `package.json`(1.0.0) / 사이드바(v0.9.0) / 커밋(0.9.0). → 정리 필요.
- **BoT-SORT/오픈보캐뷸러리 미적용**: 로드맵 항목으로 존재하나 미구현. → Phase 2~3.

---

## 3. 고도화 목표 & KPI 정의

### 3.1 1순위 KPI (이번 고도화 핵심)

| KPI | 정의 | 측정 방법 |
|-----|------|-----------|
| **Recall (누락 최소화)** | 실제 객체 중 검출한 비율 (특히 person, 소형 객체) | 검증셋 기준 클래스별 Recall, mAP@0.5 |
| **FAR (오탐 최소화)** | 시간당 오경보 건수 (False Alarm Rate per hour) | 라벨링된 footage에서 단위 시간당 FP 알람 수 |

### 3.2 보조 KPI

| KPI | 정의 |
|-----|------|
| Precision / mAP@0.5:0.95 | 종합 검출 품질 |
| 추론 지연(latency) / 추론 FPS | 스냅샷당 처리시간, 처리량 (실시간성) |
| 레이어별 처리시간 | L1+L2 / L3 / L4 ms (병목 추적) |
| 운영 자동화율 | 무인 폴더감시·자동 재학습 비중 |

### 3.3 KPI 목표 (제안값 — Phase 1 베이스라인 측정 후 확정)

> ⚠️ 현재 정량 베이스라인이 없으므로, 아래 목표치는 **제안(가설)** 이며 Phase 1에서 검증셋으로 베이스라인을 측정한 뒤 수치를 확정·합의한다.

| KPI | 현재(Baseline) | Phase 1 (2026 H2) | Phase 2 (2027 H1) | Phase 3 (2027 H2~) |
|-----|----------------|--------------------|--------------------|---------------------|
| Person Recall | 측정 필요 (TBD) | 베이스라인 +10%p | ≥ 0.90 | ≥ 0.95 |
| mAP@0.5 (전체) | 측정 필요 (TBD) | 베이스라인 확보 | +0.10 | +0.15 |
| FAR (건/시) | 측정 필요 (TBD) | 베이스라인 −30% | −60% | −75% |
| 추론 지연(고정밀) | 측정 필요 (TBD) | 측정·가시화 | −20% (TensorRT) | −40% |
| 무인 운영 가능 시간 | N/A | 반자동 | 8h 무인 | 24h 무인 |

### 3.4 측정 프로토콜
1. 광산 footage **1,000+ 프레임 검증셋** 구축(비행/위치별 분리로 누수 방지).
2. 각 레이어 토글 on/off로 **mAP·FAR 기여도** 측정(ablation).
3. 현장 **shadow run**(신·구 시스템 병행 운영)으로 누락·오탐 차이 분석.

---

## 4. 단계별 로드맵 (반기 3단계)

### 🟢 Phase 1 — 2026 H2 (단기): 안정화 · 정량 기반 구축 · 자체모델 1차 투입

**테마: "측정할 수 없으면 개선할 수 없다" — 베이스라인 확보와 현장 도메인 모델 1차 적용.**

| 영역 | 과제 |
|------|------|
| **기획** | 검증셋 정의·KPI 베이스라인 수치 확정. 발파 SOP와 알람 정책(확정 트랙 게이팅 여부) 합의. |
| **UI/UX** | ① 프로필(fast/balanced/accurate)을 엔진 파이프라인에 실제 연동 ② KPI/지표 대시보드 패널(레이어 ablation 결과 가시화) ③ 오탐 리뷰 워크플로 UI(이벤트에 "오탐 신고" 버튼 → Active Learning 큐 연동) ④ 버전·상태 표기 정합화 |
| **비즈니스 로직** | A/B 토글 측정 자동 리포트 생성. 위험도 판정에 "확정 트랙 기반 알람" 옵션 정식화. |
| **모델 투입** | ▶ **자체 파인튜닝 검출기 1차** — `finetune_detector.py`로 **YOLO11m** 학습 후 L1 primary 교체 ▶ **하드네거티브 라운드 1회** 반영(오탐 감소) ▶ SigLIP 프롬프트/마진 튜닝(FAR 직접 개선) |
| **데이터** | Grounding DINO+SAM 자동라벨 → 수동검수 파이프라인 정착. 클래스별 최소 데이터 확보(person 2,000+ boxes 등). |
| **산출물** | 베이스라인 리포트, v1.0 자체모델, 검증셋, ablation 결과표 |
| **KPI 목표** | FAR −30%, Person Recall +10%p, **지표 측정 체계 가동** |

---

### 🟡 Phase 2 — 2027 H1 (중기): 자체모델 고도화 · 검증 자동화 · 성능 최적화

**테마: 도메인 특화 정확도 끌어올리기 + 운영 안정성 확보.**

| 영역 | 과제 |
|------|------|
| **기획** | shadow run 기반 현장 검증 정례화. 재학습 주기/트리거(데이터 증가량 기준) 정의. |
| **UI/UX** | ① 트렌드 분석 뷰(시간대별 감지·오탐 추이) ② 멀티 영상 배치 분석 큐 관리 화면 ③ 모델 버전/실험 비교 UI ④ PPE 상태 표시(조끼·안전모 미착용 플래그) |
| **비즈니스 로직** | Active Learning → 재학습 → 배포의 **반자동 루프** 구축. confirm_window/N-of-M 현장 모션 기준 재튜닝. |
| **모델 투입** | ▶ **RT-DETRv2-L 자체 파인튜닝** 앙상블 강화(소형 객체 Recall) ▶ **합성 데이터**(`synthetic_gen.py`, SD+ControlNet) 투입(실:합성 ≤ 3:1) ▶ **PPE 서브검출기**(`ppe_train.py`, YOLO11s/n) 도입 → person crop 교차검증 ▶ **BoT-SORT** 추적기 옵션(드론 카메라 모션 대응) ▶ **TensorRT FP16/INT8** 변환으로 추론 지연 단축 |
| **데이터** | 합성+하드네거티브 혼합 데이터셋 정착. calibration set 구축(INT8용). |
| **산출물** | v2.0 앙상블 모델, TensorRT 엔진, 반자동 재학습 파이프라인, PPE 모듈 |
| **KPI 목표** | FAR −60%, Person Recall ≥0.90, 추론 지연 −20% |

---

### 🔵 Phase 3 — 2027 H2~ (장기): 오픈보캐뷸러리 · 운영 자동화 · 무인 운영

**테마: 신규 객체 대응력 + 사람 개입 최소화.**

| 영역 | 과제 |
|------|------|
| **기획** | 무인 24h 운영 목표. 신규 위험요소(낙석·연기·미인가 차량) 확장 정의. |
| **UI/UX** | ① 무인 운영 모드(자동 알람·에스컬레이션) ② 알림 연동(현장 경보·메신저) ③ 운영 리포트 자동 생성/내보내기 ④ 다국어/접근성 정비 |
| **비즈니스 로직** | 완전 자동 Active Learning→재학습→배포 루프. 드리프트 감지 시 자동 재학습 트리거. |
| **모델 투입** | ▶ **DINOv2 자기지도 사전학습**(`pretrain_dinov2.py`) → 도메인 적응 backbone으로 검출기 초기화 ▶ **YOLO-World v2**(오픈보캐뷸러리) → 학습 없이 신규 클래스 프롬프트 대응 ▶ **VLM 업그레이드**(대형 SigLIP / SAM 3 마스크 정제 라벨링) ▶ INT8 양자화 전면 적용 |
| **데이터** | 비라벨 드론 footage 50k+ 수집(자기지도용). 자동 큐레이션. |
| **산출물** | v3.0 오픈보캐뷸러리 스택, 무인 운영 모드, 자동 재학습 시스템 |
| **KPI 목표** | FAR −75%, Person Recall ≥0.95, 24h 무인 운영 |

---

## 5. 모델 투입 전략 (현재 → 향후) 종합표

| 레이어/기능 | 현재 (v0.9) | Phase 1 (2026 H2) | Phase 2 (2027 H1) | Phase 3 (2027 H2~) |
|-------------|-------------|--------------------|--------------------|---------------------|
| **L1 Primary** | YOLOv8l (COCO) | **YOLO11m 자체 파인튜닝** | + RT-DETRv2-L 자체 파인튜닝 | **DINOv2 backbone** 초기화 / **YOLO-World v2** |
| **L1 Secondary** | RT-DETR-L (COCO) | RT-DETR-L | RT-DETRv2-L 자체 | 오픈보캐뷸러리 앙상블 |
| **L1 Fast** | YOLOv8m | YOLO11m/s | TensorRT FP16 경량 | INT8 경량 |
| **L2 SAHI** | tile 768 / ov 0.25 | 파라미터 현장 튜닝 | 적응형 타일 | 유지 |
| **L3 VLM** | SigLIP-base-224 | 프롬프트·마진 튜닝 | 대형 SigLIP 검토 | SigLIP 대형 / SAM 3 연계 |
| **L4 Tracker** | ByteTrack | confirm 튜닝 | **+ BoT-SORT 옵션** | 모션보정 추적 |
| **L5 Active Learning** | 수집만 | 오탐신고 UI 연동 | 반자동 재학습 | 완전 자동 루프 |
| **PPE 서브태스크** | 없음 | 설계 | **YOLO11s/n 투입** | person 신뢰도 보정 정착 |
| **추론 최적화** | FP16(CUDA) | 측정·프로파일링 | **TensorRT FP16/INT8** | INT8 전면 |
| **자동 라벨링** | Grounding DINO + SAM | 정착 | 검수 효율화 | SAM 3 마스크 정제 |
| **합성 데이터** | 없음 | 설계 | **SD + ControlNet** 투입 | 자동 큐레이션 |

---

## 6. 데이터 전략

본 시스템의 장기 정확도 향상 엔진은 **데이터 루프**다.

```
현장 footage ─► 자동라벨(Grounding DINO+SAM) ─► 수동검수(CVAT) ─► 학습 데이터
      ▲                                                              │
      │                                                              ▼
오탐/하드네거티브 ◄── Active Learning(L5) ◄── 운영 중 불확실 검출 ◄── 재학습·배포
```

- **자동 라벨링**: `person / dump truck / tanker truck / truck` 프롬프트. 자동라벨은 부트스트랩일 뿐 — **반드시 검수 후 학습**.
- **하드네거티브**: 바위·그림자·장비 가장자리·역광·소형 작업자. 양성:하드네거티브 권장 비율 준수(positive 10장당 empty 3~5, 혼합난이도 2~3).
- **Active Learning**: confidence 0.30~0.60 + VLM reject + 미확정 트랙을 manifest로 저장 → 재학습 시드.
- **합성 데이터**: 희소 양성(특정 포즈 PPE 작업자)·통제된 하드네거티브 보강. 실:합성 ≤ 3:1.
- **데이터 충분 기준**: person 2,000+ / dump_truck 800+ / tanker_truck 800+ / truck 1,000+ boxes. 근접 중복보다 **다양한 소형 객체 예시**가 우선.

---

## 7. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 정량 베이스라인 부재 | 개선 효과 입증 불가 | **Phase 1 최우선**으로 검증셋·측정체계 구축 |
| 로컬 단일 GPU 자원 한계 | DINOv2 사전학습 등 대형 학습 제약 | 사전학습은 외부 클러스터/임대 GPU 활용, 추론은 TensorRT 경량화로 로컬 대응 |
| 자동라벨 오류 누적 | 학습 데이터 오염 | 검수 게이트 필수, 하드네거티브 루프로 보정 |
| 오탐 알람 피로 | 운영자 신뢰도 저하 | FAR을 1순위 KPI로 추적, 확정 트랙 게이팅 옵션 |
| 도메인 드리프트(계절·현장 변화) | 정확도 저하 | Active Learning 재학습 주기화, 드리프트 감지 |
| 모델 교체 시 회귀 | 기존 성능 저하 | 검증셋 회귀 테스트 + shadow run 후 배포 |

---

## 8. 부록

### 8.1 핵심 파일 맵
- `engine.py` — ZMQ 서버 + 세션 스레드 (검출 로직 없음)
- `detection/pipeline.py` — 5-Layer 오케스트레이터
- `detection/detector.py` (L1) · `sahi_runner.py` (L2) · `vlm_verifier.py` (L3) · `tracker.py` (L4) · `active_learning.py` (L5)
- `configs/runtime_detector.json` — 레이어별 런타임 설정(A/B 토글)
- `training/` — finetune / synthetic / ppe / dinov2 / tensorrt
- `docs/model-strategy.md` · `dataset-guide.md` · `hard-negative-guide.md` · `auto-labeling-guide.md`

### 8.2 용어
- **FAR**: False Alarm Rate per hour (시간당 오경보 수)
- **N-of-M**: 최근 M개 스냅샷 중 N회 이상 검출 시 확정 (깜빡임 제거)
- **SAHI**: Slicing Aided Hyper Inference (고해상도 타일링 추론)
- **Hard Negative**: 사람/차량과 혼동되는 음성 샘플(바위·그림자 등)
- **Active Learning**: 불확실 샘플을 우선 수집해 재학습 효율을 높이는 기법

### 8.3 다음 액션 (Phase 1 착수)
1. [ ] 검증셋 1,000+ 프레임 라벨링 & 베이스라인 mAP·FAR 측정
2. [ ] KPI 목표 수치 확정/합의 (3.3 표 갱신)
3. [ ] YOLO11m 자체 파인튜닝 1차 → L1 교체
4. [ ] UI 프로필 ↔ 엔진 파이프라인 연동
5. [ ] 오탐 리뷰 → Active Learning 큐 UI 연동
