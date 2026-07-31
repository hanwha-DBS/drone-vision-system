# 드론 비전 시스템(AeroSentinel) — Jira EPIC 정리 (개발 현황 기반)

> 본 문서는 실제 코드베이스(v0.9.0)에 구현된 내용을 근거로 10개 EPIC의 **현 상태**와 **남은 작업**을 분리해 정리한 것이다. Jira 등록 및 단계별 진척 보고용.

## 상태 범례
| 표기 | 의미 |
|------|------|
| ✅ **구현 완료** | 코드에 반영되어 동작. 검증·튜닝만 남음 |
| 🟡 **부분 구현** | 핵심 일부가 이미 존재. 통합/마무리 필요 |
| ⬜ **미착수** | 스크립트·폴더 구조만 있거나 없음 |

## EPIC 요약 & 권장 단계

| EPIC | 제목 | 상태 | 권장 Phase |
|------|------|------|------------|
| 01 | KPI 측정 체계 및 검증셋 구축 | ⬜ 미착수 (토글 인프라만 존재) | Phase 1 (2026 H2) |
| 02 | UI 프로필 ↔ 엔진 파이프라인 연동 | 🟡 부분 (UI·전송 O / 엔진 소비 X) | Phase 1 |
| 03 | 자체 학습 검출기 1차 도입 (YOLO11m) | 🟡 부분 (학습 스크립트 O / 학습·교체 X) | Phase 1 |
| 04 | 오탐 감소용 Hard Negative 루프 | 🟡 부분 (도구 O / 재학습 루프 X) | Phase 1 |
| 05 | 오탐 리뷰 UI 및 Active Learning 연동 | 🟡 부분 (백엔드 L5 O / UI X) | Phase 1 |
| 06 | 추적기 및 알람 확정 로직 | ✅ 대부분 완료 (튜닝·shadow run만 남음) | Phase 1 |
| 07 | TensorRT 기반 추론 최적화 | 🟡 부분 (export 스크립트·계측 UI O / 변환·적용 X) | Phase 2 (2027 H1) |
| 08 | PPE 서브 검출기 추가 | ⬜ 미착수 (학습 스크립트만 존재) | Phase 2 |
| 09 | 운영 분석 및 리포트 대시보드 | 🟡 일부 (레이어 처리시간 시각화만 존재) | Phase 2 |
| 10 | 오픈보캐뷸러리 및 차세대 모델 연구 | ⬜ 연구단계 (DINOv2 래퍼만 존재) | Phase 3 (2027 H2~) |

> **주요 용어 정정**
> - EPIC-02: "fast→L1 / balanced→L1+L2 / accurate→Full"은 *목표 매핑*이며 현재 코드에는 없음. **현재는 분석 간격(≤2초) 기준으로 fast(L1 단독)↔full(L1~L5) 자동 전환**됨. 프로필 셀렉트 값은 엔진에 전달되나 소비되지 않음.
> - EPIC-06: 추적기·N-of-M·확정트랙 알람은 *개선* 대상이 아니라 **이미 구현 완료**된 기능(튜닝/검증 단계).
> - EPIC-07: "inference runner 구현"이 아니라 — ultralytics가 `.engine`을 직접 로드하므로 **TensorRT 변환·calibration·config 적용**이 실제 작업.

---

# [EPIC-01] KPI 측정 체계 및 검증셋 구축  ⬜ 미착수

## 목적
객체 검출 성능(mAP / Recall / FAR)의 정량 측정 기반 구축 및 모델 회귀 테스트 환경 마련. **모든 후속 EPIC의 효과를 입증할 베이스라인 — 최우선.**

## 이미 구현된 것 (근거)
- 레이어 on/off 토글 인프라: `configs/runtime_detector.json` + `DetectionPipeline.describe_active_layers()` (A/B 측정의 전제 조건은 이미 존재)
- 데이터셋 폴더 구조 & 구조 검증 스크립트: `datasets/person_dump_tanker/`(train/val/test), `scripts/validate_dataset.py` — **단, 파일 개수·split 검증만 수행. 성능 측정 아님**
- 레이어별 처리시간 계측: `layer_timings_ms`(L1+L2 / VLM / Tracker)는 이미 측정·전송됨

## 남은 작업
- 광산/발파 현장 검증셋 1,000+ 프레임 수집 및 person/vehicle GT 라벨링 (비행·위치별 split으로 누수 방지)
- **mAP@0.5 / Recall / Precision / FAR(건/시) 측정 스크립트 신규 구현** (예: `scripts/evaluate.py`)
- 레이어 on/off 조합별 ablation 자동 측정 러너
- 결과 리포트 자동 생성(JSON/CSV) + 모델 회귀 테스트

## 완료 조건 (DoD)
- 기준 검증셋으로 베이스라인 수치 산출 가능
- Layer on/off 조합별 성능 비교 가능
- 회귀 테스트 자동 실행 가능

---

# [EPIC-02] UI 프로필 ↔ 엔진 파이프라인 연동  🟡 부분 구현

## 목적
UI의 fast / balanced / accurate 프로필을 실제 엔진 레이어 설정과 연동.

## 이미 구현된 것 (근거)
- UI 프로필 셀렉트 + 표시: `index.html` `#profileSelect`, `profileMeta`, `profileBadge`
- 프로필 값 전송: `main.js` / `index.html`이 `analysis_profile: profileSelect.value`를 `start_video` 요청에 포함해 전송
- **간격 기반 자동 모드 전환은 이미 동작**: `pipeline.py`에서 `interval_seconds ≤ 2` → fast 모드(L1 단독), 그 외 → full 스택(L1~L5)

## 남은 작업 (핵심: 엔진이 프로필을 소비하지 않음)
- `engine.py`의 `start_video` 핸들러가 `analysis_profile`을 **읽지 않음** → 프로필별 레이어 프리셋 적용 로직 추가
- 프로필 → 레이어 매핑 정의 및 구현 (제안):
  - **fast** → L1 단독
  - **balanced** → L1 + L2(SAHI) + L4(Tracker)
  - **accurate** → Full Stack (L1~L5, VLM 포함)
- 활성 레이어를 HUD/배지에 표시 (현재 `layersBadge`는 정적)
- 실시간 로그에서 적용된 프로필·활성 레이어 출력

## 완료 조건 (DoD)
- 프로필 변경 시 실제 엔진 레이어 변경 확인 가능
- 실시간 로그에서 활성 레이어 확인 가능

---

# [EPIC-03] 자체 학습 검출기 1차 도입 (YOLO11m)  🟡 부분 구현

## 목적
COCO 기반 범용 모델(현재 `yolov8l.pt` + `rtdetr-l.pt`)에서 광산 도메인 특화 검출기로 전환.

## 이미 구현된 것 (근거)
- 파인튜닝 스크립트: `training/finetune_detector.py` (YOLO11m 기본, RT-DETR 옵션, 드론 도메인 증강 포함)
- 데이터셋 정의/가이드: `configs/person_dump_tanker_dataset.yaml`, `docs/dataset-guide.md`
- 클래스 alias 정규화: `detection/detector.py` `FilterConfig.normalize()` (dump_truck/tanker_truck/truck/excavator → vehicle 등) **이미 동작**
- 모델 교체 가능 구조: `runtime_detector.json`의 `gpu.model_path` 교체만으로 L1 primary 전환 가능

## 남은 작업
- 학습 데이터셋 수집·정리 및 train/val/test split 구성
- YOLO11m 실제 파인튜닝 수행
- 기존 YOLOv8l 대비 검증셋 성능 비교 (EPIC-01 의존)
- 학습 모델로 L1 primary 교체 (`runtime_detector.json` 갱신)
- 모델 버전 관리 정책 수립 (체크포인트 명명·기록)

## 완료 조건 (DoD)
- 검증셋 기준 Recall 개선 확인
- 기존 대비 FAR 감소 확인
- 운영 환경 적용 가능 상태

---

# [EPIC-04] 오탐 감소용 Hard Negative 루프 구축  🟡 부분 구현

## 목적
광산 특화 오탐(바위/그림자/장비 가장자리/밝은 자갈) 감소.

## 이미 구현된 것 (근거)
- False Positive 리뷰셋 생성 스크립트: `scripts/generate_false_positive_review_set.py`, 산출물 `datasets/review_candidates/false_positive_review.csv`
- SigLIP(L3) 거부 튜닝 노브: `vlm_verifier.py` / `runtime_detector.json`의 `min_positive_margin`, `reject_if_top_is_negative`, `hard_negative_prompts` (바위/그림자/장비 가장자리 등 7종 프롬프트 이미 정의)
- 하드네거티브 운영 가이드: `docs/hard-negative-guide.md` (권장 데이터 혼합비 포함)

## 남은 작업
- 하드네거티브 데이터셋 별도 분리·관리 체계화
- SigLIP reject threshold 현장 데이터 기반 튜닝
- Hard Negative 반영 재학습 라운드 수행
- 재학습 전후 FAR 리포트 자동 비교 (EPIC-01 의존)

## 완료 조건 (DoD)
- FAR 감소 수치 확인 가능
- Hard Negative 데이터셋 운영 체계화

---

# [EPIC-05] 오탐 리뷰 UI 및 Active Learning 연동  🟡 부분 구현

## 목적
운영자가 오탐을 쉽게 피드백하고 재학습 루프로 연결 가능하도록 개선.

## 이미 구현된 것 (근거)
- **Active Learning 백엔드(L5) 이미 동작**: `detection/active_learning.py` — confidence 0.30~0.60 + VLM reject + 미확정 트랙을 ROI crop + `manifest.csv`로 **자동 저장** (추론 중 자동 실행)
- AL 저장 카운트 UI 노출: `index.html` `alBadge` (AL · N)
- 이벤트 아카이브 표시: `recentEvents` 그리드, 썸네일·위험도·프레임 표시 (단, **표시 전용**)

## 남은 작업 (핵심: UI 피드백 경로 부재)
- 이벤트 아카이브에 **"오탐 신고" 버튼** 추가 (현재 index.html에 해당 UI 없음)
- 이벤트 상태 관리(flagged / reviewed) 및 저장
- 운영자 신고분을 Active Learning manifest로 자동 반영
- 불확실 샘플 큐 시각화
- 재학습 export 기능 (라벨링 도구 import용)

## 완료 조건 (DoD)
- UI에서 오탐 등록 가능
- Active Learning 큐 자동 생성·반영 확인

---

# [EPIC-06] 추적기 및 알람 확정 로직  ✅ 대부분 완료

## 목적
깜빡임 검출 및 순간 오탐 감소를 위한 N-of-M 기반 확정 로직. **(개선이 아니라 이미 구현된 기능의 검증·튜닝 단계)**

## 이미 구현된 것 (근거)
- ByteTrack 래퍼 + 위치기반 fallback 추적기: `detection/tracker.py` `TemporalTracker` (`supervision.ByteTrack` 로드, 실패 시 IoU fallback)
- **N-of-M 확정 룰 구현 완료**: `confirm_window=5`, `confirm_min_hits=3`, `filter_confirmed()`
- **확정 트랙 기반 알람 옵션 구현 완료**: `require_confirmation_for_alarm` 플래그 (현재 false — 감지 즉시 저장, 확정은 ✓ 배지로 표시)
- 트랙 메타 직렬화 + UI 표시: `track_id`, `track_hits`, `track_confirmed` → UI `#id` 라벨 및 ✓ 배지(`index.html` line ~1848)
- 미관측 트랙 GC: `lost_track_buffer` 기반 정리

## 남은 작업
- `confirm_window` / `confirm_min_hits` 현장 데이터 기반 튜닝
- ByteTrack 파라미터 프로파일별 분리 (EPIC-02 연계)
- track lifecycle 상세 시각화 로그 보강
- shadow run(신·구 로직 병행) 비교 기능

## 완료 조건 (DoD)
- 순간 오탐 감소 정량 확인 (EPIC-01 의존)
- 확정 트랙 기반 이벤트 저장 모드 검증 완료

---

# [EPIC-07] TensorRT 기반 추론 최적화  🟡 부분 구현

## 목적
온프레미스 로컬 GPU 환경에서 추론 지연 감소 및 처리량 향상.

## 이미 구현된 것 (근거)
- TensorRT export 스크립트: `training/export_tensorrt.py` (INT8/FP16, calibration 인자 지원)
- CUDA/CPU fallback: `engine.py` device 자동 감지 + `pipeline.py` CPU 시 SAHI/VLM/secondary 자동 비활성
- FP16 추론: `runtime_detector.json` `fp16_on_cuda`, detector `half` 적용
- 레이어별 latency 계측·시각화 UI: `layer_timings_ms` → UI "5-Layer 처리 시간" 바 (이미 존재)

## 남은 작업
- YOLO11 → TensorRT **FP16 엔진 실제 변환·적용** (ultralytics가 `.engine`을 직접 로드 → `model_path`에 `.engine` 지정)
- INT8 calibration set 구성(200~500장, 배포 분포 일치)
- 변환 전후 latency·정확도 비교 검증
- (선택) CPU fallback 경로에서의 동작 재확인

## 완료 조건 (DoD)
- FP16 TensorRT 추론 적용
- 기존 대비 latency 감소 확인

---

# [EPIC-08] PPE 서브 검출기 추가  ⬜ 미착수

## 목적
안전모/안전조끼 미착용 감지 기능 추가.

## 이미 구현된 것 (근거)
- PPE 학습 스크립트(설계): `training/ppe_train.py` (YOLO11s/n, 클래스 helmet/vest/no_helmet/no_vest, person crop 기반 추론 전제)

## 남은 작업
- PPE 데이터셋 구성 (`configs/ppe_dataset.yaml` 신규 — 현재 미존재)
- YOLO11s/n PPE 모델 학습
- person crop 기반 secondary inference를 파이프라인에 통합 (현재 미통합)
- PPE 상태 UI 표시 (미착용 플래그)
- PPE 이벤트 저장 정책 추가

## 완료 조건 (DoD)
- PPE 상태 감지 가능
- person detection과 연계 동작 확인

---

# [EPIC-09] 운영 분석 및 리포트 대시보드  🟡 일부 구현

## 목적
운영 지표 및 탐지 추세 시각화 제공.

## 이미 구현된 것 (근거)
- 실시간 KPI 스트립: 현재 사람/차량, 이벤트, 스냅샷, 분석 완료, 원본/추론 FPS (`index.html` KPI 카드)
- 레이어 처리시간 시각화: "5-Layer 처리 시간" 바 (이미 존재)
- 이벤트 아카이브 그리드 (이력 표시)

## 남은 작업
- 시간대별 탐지 통계 (집계 저장 필요)
- FAR 추이 그래프
- 모델 버전별 성능 비교 뷰
- 운영 리포트 export (PDF/CSV)

## 완료 조건 (DoD)
- KPI 추세 확인 가능
- 리포트 export 가능

---

# [EPIC-10] 오픈보캐뷸러리 및 차세대 모델 연구  ⬜ 연구단계

## 목적
신규 위험 객체 대응력 확보 및 장기 모델 전략 검증.

## 이미 구현된 것 (근거)
- DINOv2 자기지도 사전학습 래퍼: `training/pretrain_dinov2.py` (공식 dinov2 entrypoint 호출, 클러스터 필요)
- 모델 전략 문서: `docs/model-strategy.md`의 "다음 업그레이드 사이클"(YOLO-World v2, DINOv2 backbone, INT8)

## 남은 작업
- YOLO-World v2 평가 (오픈보캐뷸러리, 학습 없이 신규 클래스 프롬프트 대응)
- DINOv2 backbone 사전실험 → 검출기 초기화
- SigLIP 대형 모델 테스트 (L3 검증 정확도)
- SAM 기반 자동 라벨 정제 실험 (auto-labeling 고도화)
- 신규 클래스 프롬프트 검증

## 완료 조건 (DoD)
- 차세대 모델 PoC 결과 확보
- 신규 클래스 탐지 가능 여부 검증

---

## 부록: 단계 보고용 한 줄 요약

- **Phase 1 (2026 H2)** — EPIC-06(완료 검증) · EPIC-02/03/04/05(부분→완성) · EPIC-01(베이스라인 신규 구축). *핵심: 측정 기반 + 자체모델 1차 + 오탐 루프.*
- **Phase 2 (2027 H1)** — EPIC-07(TensorRT 적용) · EPIC-08(PPE) · EPIC-09(리포트 대시보드). *핵심: 최적화 + 기능 확장.*
- **Phase 3 (2027 H2~)** — EPIC-10(차세대 모델 PoC). *핵심: 오픈보캐뷸러리 + 무인 운영 기반.*
