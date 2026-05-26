# Model Strategy — 5-Layer Safety Stack

이 시스템은 단일 detector 강화가 아니라 **레이어드 검출-검증-시간일치** 구조로 인식률과 오탐률을 동시에 개선한다. 발파 안전 도메인은 **누락(FN)** 과 **오탐(FP)** 둘 다 비용이 크기 때문이다.

## 추론 파이프라인

```
snapshot ─► L1 Detector ─► L2 SAHI 타일링 ─► L3 SigLIP 검증 ─► L4 ByteTrack(N-of-M) ─► 알람
                                                                       │
                                                                       └► L5 Active Learning
```

- **L1 Detector** (`detection/detector.py`) — YOLO/RT-DETR pluggable wrapper. 도메인 필터(클래스 별칭, 기하학 area/aspect, 클래스별 최소점수) 포함.
- **L2 SAHI** (`detection/sahi_runner.py`) — 고해상도 드론 영상을 타일로 잘라 추론 후 NMS. 작은 객체 recall의 핵심.
- **L3 SigLIP VLM** (`detection/vlm_verifier.py`) — 박스 crop을 zero-shot으로 검증. 광산 hard-negative(바위/그림자/장비 가장자리) 프롬프트와 비교해 reject.
- **L4 ByteTrack** (`detection/tracker.py`) — supervision의 ByteTrack 사용. N-of-M 룰(`confirm_window`, `confirm_min_hits`) 통과한 트랙만 알람.
- **L5 Active Learning** (`detection/active_learning.py`) — confidence 0.30~0.60 박스와 reject된 검출을 manifest로 저장 → 재학습 시드.

오케스트레이터는 `detection/pipeline.py`의 `DetectionPipeline`이다. `engine.py`는 ZMQ 서버와 세션 스레드만 담당.

## 권장 모델 스택

1. **GPU 운영** (권장 기본값)
   - L1 Primary: 자체 학습한 **YOLO11m** 또는 **RT-DETRv2-L** (`training/finetune_detector.py`로 학습)
   - L1 Secondary (앙상블): `rtdetr-l.pt`
   - Fast (간격 ≤2s): `yolov8m.pt`
   - L3: `google/siglip-base-patch16-224` (FP16, 224 입력)
   - L4: `supervision.ByteTrack` 백엔드

2. **CPU 폴백**
   - L1: `yolov8l.pt`
   - SAHI/VLM 비활성 권장 (`configs/runtime_detector.json`에서 `enabled: false`로 토글)
   - Tracker는 fallback IoU 매칭만 사용

3. **다음 업그레이드 사이클**
   - DINOv2 self-supervised 사전학습 → detector backbone 초기화
   - YOLO11/12 대신 YOLO-World v2로 open-vocabulary 전환
   - TensorRT INT8 변환 (`training/export_tensorrt.py`)

## 데이터 워크플로

1. **자동 라벨링** — Grounding DINO 1.5 + SAM 2 (`configs/auto_labeling.json`, `scripts/auto_label_bootstrap.py`).
2. **수동 리뷰** — CVAT 또는 Label Studio.
3. **재학습** — `python -m training.finetune_detector --data configs/person_dump_tanker_dataset.yaml`.
4. **Active Learning** — `datasets/active_learning/manifest.csv` 를 라벨링 도구에 import.
5. **Hard-negative refresh** — `scripts/generate_false_positive_review_set.py`로 오탐 프레임 추출.
6. **합성 데이터** — `python -m training.synthetic_gen --per-prompt 50` (실제:합성 = 3:1 이하 권장).

## 런타임 설정 (`configs/runtime_detector.json`)

각 레이어는 독립적으로 켜고 끌 수 있다 — A/B 비교가 가능해야 효과 측정이 신뢰성 있다.

| 키 | 기능 |
|----|------|
| `cpu` / `gpu` / `fast_sampling` | L1 모델 경로/해상도/신뢰도 |
| `ensemble.enabled` | primary+secondary 앙상블 NMS |
| `sahi.enabled`, `tile_size`, `overlap` | L2 타일링 |
| `vlm.enabled`, `model_id`, `hard_negative_prompts` | L3 VLM 검증 |
| `tracker.enabled`, `confirm_window`, `confirm_min_hits` | L4 N-of-M 룰 |
| `active_learning.enabled`, `confidence_min/max` | L5 ROI 캡처 |
| `require_confirmation_for_alarm` | 알람을 confirmed 트랙으로만 제한할지 |

## 추적기 권장

- **ByteTrack** — 기본 (안정성)
- **BoT-SORT** — 드론 카메라 모션이 큰 광산 현장에 더 적합

`tracker.backend`를 `bytetrack` → `botsort`로 바꾸려면 supervision 측 어댑터를 추가하면 된다 (TODO).

## 운영 모드

- **고정밀 모드**: 간격 3~10s, 모든 레이어 활성 → 발파 직전 안전 스윕에 권장
- **고속 미리보기**: 간격 ≤2s, L1만 사용 (자동) → 영상 미리보기·드론 비행 중 실시간 모니터링

## 다음 검증 단계

1. 광산 footage 1000+ 프레임 검증셋 구축
2. 각 레이어 토글 켜고/끄고 mAP·FAR(False Alarm Rate per hour) 측정
3. 현장 shadow run — 신·구 시스템 병행 운영 → 누락·오탐 차이 분석
