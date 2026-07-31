# 드론 비전 시스템 — 개발 티켓

## 벤치마킹 대상
- Komatsu KOMTRAX / DISPATCH (Modular Mining) — 실시간 차량 상태 대시보드, 무인 데이터 수집
- Caterpillar Cat MineStar — 카드 KPI, 다크 운전석, **Detect** 객체 안전 감지
- Hexagon Mining HxGN — 다크 운전석, 다중 검증 충돌방지
- Wenco (Hitachi) — 운전자 의사결정 보조 컬러 시그널

상태 표기: ✅ 완료 · 🟡 부분 · 🧩 예정

---

## TICKET-001 · UI/UX 개발  ✅
**벤치마킹** Komatsu DISPATCH, Caterpillar MineStar 운전석 디스플레이

**작업 내용**
- Electron 기반 다크 관제 콘솔 화면 구축 (`index.html`)
- KPI 카드 스트립, 위험도 신호등 4단계, 라이브 뷰어 HUD 오버레이, 이벤트 아카이브, 미션 바
- 시안/민트 액센트, CSS 변수 토큰화

**기대 효과**
- 광산 표준 운전석 UX 도입, 한 화면에서 현장 상태·위험도 즉시 인지

---

## TICKET-002 · AI 모델 연구·검토  ✅
**벤치마킹** Cat MineStar Detect, Hexagon 충돌방지 — 다중 검증 안전 감지

**작업 내용**
- 후보 모델 비교 검토: YOLO 계열(v8/11), RT-DETR, Grounding DINO+SAM, SigLIP, DINOv2, ByteTrack
- 각 모델 강점/약점, 광산 도메인 적합도 정리
- 결론: 단일 검출기 강화가 아니라 **검출-검증-시간일치 다단계** 구조 채택

**기대 효과**
- 누락(FN)·오탐(FP) 동시 감소를 위한 모델 스택 방향 확정

---

## TICKET-003 · 모델 선정 (5-Layer Safety Stack)  ✅
**벤치마킹** Hexagon Mining 다중 검증 안전 판정

**작업 내용**
- L1 검출: **YOLOv8l + RT-DETR-L 앙상블** (고속 모드 YOLOv8m)
- L2 **SAHI** 고해상도 타일링 (작은 객체 Recall)
- L3 **SigLIP VLM** zero-shot 검증 (광산 하드네거티브 7종 프롬프트)
- L4 **ByteTrack** N-of-M 시간일치 확정 (window=5, hits=3)
- L5 **Active Learning** 불확실 검출 자동 수집
- 모든 레이어 on/off 토글 (`runtime_detector.json`)

**기대 효과**
- 검출 한 방이 아닌 다단계로 오탐 직접 제거, 레이어별 A/B 측정 가능

---

## TICKET-004 · 로컬 저장 (이벤트·학습 데이터)  ✅
**벤치마킹** KOMTRAX 로컬 텔레메트리, 산업 HMI 로컬 캐시

**작업 내용**
- 감지 이벤트 주석 이미지 → `results/events/`
- Active Learning ROI crop + `manifest.csv` → `datasets/active_learning/`
- 외부 서버 미사용, 온프레미스 단독 구동

**기대 효과**
- 인터넷 단절에도 운영 가능, 보안 현장 적합, 재학습용 데이터 자동 축적

---

## TICKET-005 · 자동 데이터 수집 (폴더 감시)  ✅
**벤치마킹** KOMTRAX 무인 데이터 수집

**작업 내용**
- 지정 폴더 감시 → 신규 영상 자동 분석 투입
- 파일 안정화 2초 확인, 세션 간 중복 방지(localStorage)

**기대 효과**
- 사람 개입 없이 신규 영상 자동 처리, 무인 운영 기반

---

## TICKET-006 · 운영 인프라 (Electron ↔ Python ZMQ)  ✅
**벤치마킹** 산업 제어 모듈형 아키텍처 (UI/엔진 분리)

**작업 내용**
- Electron이 Python 엔진을 자식 프로세스로 spawn, ZeroMQ REQ/REP 통신
- 4스레드 세션(stream/스냅샷/추론/결과)으로 실시간성·분석 분리
- 디바이스 자동 감지, GPU FP16, CPU 안전망

**기대 효과**
- UI/엔진 독립 교체 가능, 추론 지연이 UI를 막지 않음

---

## TICKET-007 · API 연동  🧩 예정
**벤치마킹** MineStar / DISPATCH 외부 시스템 연동

**작업 내용**
- 현재는 로컬 단독 구동
- (예정) 알람 → 현장 경보/메신저 / 이벤트 → 본부 서버 업로드 / KPI·모델 버전 → 관리 콘솔 / 외부 GIS·디스패치 시스템 연동

**기대 효과**
- 단독 운영 → 본부 통합 관제로 확장

---

## 주간 보고 사용법
매주 처리한 티켓만 골라 표로 보고. 예: *W22 — TICKET-001, 003, 004 완료 / 다음 주 TICKET-007 설계.*
