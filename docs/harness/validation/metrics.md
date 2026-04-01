# Metrics — YouTube Appliance VoC Dashboard Harness (v0.1)

## 0. 목적(Purpose)
본 문서는 유튜브 가전 VoC 대시보드의 품질을 **계량 지표**로 관리하기 위한 메트릭 정의서이다.
메트릭은 다음 3가지를 보장해야 한다.

1. **무결성(Integrity)**: 라벨, 대표코멘트, 키워드, 유사댓글, 표출값이 서로 모순되지 않는다.
2. **대표성(Coverage)**: 중요한 영상/댓글이 누락되지 않거나, 누락 시 사유가 설명된다.
3. **안정성(Stability)**: 버전이 바뀌어도 결과가 불필요하게 흔들리지 않으며, 흔들리면 원인이 기록된다.

> NOTE: 본 metrics는 acceptance-criteria.md의 INV/AC 항목을 지속적으로 감시하는 "계기판"이다.

---

## 1. 원칙(Principles)
- **"0인데 1이 나오는 것"** (count=0인데 대표/키워드/표출 존재)은 P0 결함으로 간주한다.
- **설명 불가능한 결과** (reason_code 없음)는 P0 결함으로 간주한다.
- 메트릭은 항상 **동일 `dataset_fingerprint`** 기준으로 계산되어야 한다.
- 메트릭은 Audit Log의 event + reason_code로 역추적 가능해야 한다.
- 코드 내 실제 라벨 값(`"positive"` / `"negative"` / `"neutral"` / `"excluded"`)을 기준으로 계산한다.

---

## 2. 측정 범위(Measurement Scope)
| 스코프 | 설명 |
|--------|------|
| `per_run` | 파이프라인 실행(`run_id`) 단위 |
| `per_video` | 영상(`video_id`) 단위 |
| `per_segment` | 제품/브랜드/국가/기간/CEJ stage 등 필터 세그먼트 단위 |
| `per_release` | 배포 버전(`pipeline_version`/`rule_version`) 단위 |

---

## 3. 핵심 메트릭(Outcome Metrics)

### 3.1 Integrity Metrics (무결성)

#### M-INT-01: Count↔Representative Invariant Violation Rate
- **정의**: `count(label)==0`인데 대표코멘트가 존재하거나,
  `count>0`인데 대표가 임의 대체/오염된 경우의 비율
- **계산**:
  - 분모: `(video_id × 감성라벨)` 조합의 총 개수
  - 분자: 아래 중 하나라도 충족하는 케이스 수
    - `count(label)=0` AND `representative(label) != None`
    - `count(label)>0` AND 대표의 `감성` 값이 슬롯 라벨과 불일치
- **관련 acceptance**: INV-03, AC-03~AC-05

#### M-INT-02: Label↔Representative Mismatch Rate
- **정의**: 대표 코멘트의 실제 `감성` 값이 슬롯 라벨과 다른 비율
- **계산**:
  - 분모: 대표 코멘트가 생성된 슬롯 수
  - 분자: `reason_code=LABEL_MISMATCH` 케이스 수
- **관련 acceptance**: INV-01, AC-02

#### M-INT-03: Promo Leakage Rate
- **정의**: `promo_flag=True`인 콘텐츠가 대표/키워드/유사댓글에 영향을 준 비율
- **계산**:
  - 분모: 대표/키워드/유사댓글 산출 이벤트 수
  - 분자: promo 탐지됐는데 결과에 포함된 케이스 수
- **관련 acceptance**: INV-02, AC-01, AC-09

#### M-INT-04: Dataset Fingerprint Mismatch Rate
- **정의**: 같은 화면에서 counts와 대표/키워드가 서로 다른 `dataset_fingerprint`를 참조한 비율
- **계산 참조**: `audit-log-spec.md` — `dataset_fingerprint` 계산 방법
- **관련 acceptance**: INV-03, AC-05

---

### 3.2 Text Safety / Rendering Metrics

#### M-TXT-01: Garbled Output Rate
- **정의**: UI 표출 필드에서 Garbled Text가 탐지된 비율
- **계산**:
  - 분모: `RENDERING_GUARDRAIL` 이벤트 수
  - 분자: `reason_code=GARBLED_TEXT_DETECTED` 발생 수

#### M-TXT-02: Garbled Display Leakage Rate
- **정의**: Garbled Text가 탐지되었음에도 그대로 UI에 출력된 비율
- **목표**: **0%** (P0)

#### M-TXT-03: N/A Replacement Rate by Field
- **정의**: `field_name`별 `"N/A"` 치환 빈도
- **목적**: 특정 필드가 지속적으로 깨지는지 탐지 (예: `major_negative_points`)
- **목표**: 장기적으로 감소 추세

---

### 3.3 Keyword Quality Metrics

#### M-KWD-01: Keyword Noise Rate
- **정의**: 상위 K 키워드 중 `NOISE_TERMS` (acceptance-criteria.md INV-05 정의) 단어의 비율
- **계산**:
  - K=20 권장
  - 분모: 각 슬롯별 top-K 키워드 수
  - 분자: top-K 중 `NOISE_TERMS` 목록과 매칭되는 키워드 수
- **목표**: < **5%** (0%가 이상적)
- **관련 acceptance**: INV-05, AC-09

#### M-KWD-02: Label-Specific Corpus Purity
- **정의**: `"negative"` 키워드는 `감성=="negative"` 코퍼스,
  `"positive"` 키워드는 `감성=="positive"` 코퍼스에서만 나왔는지 확인
- **목표**: **100% purity**
- **관련 acceptance**: AC-10

#### M-KWD-03: Keyword Coverage
- **정의**: `count(label)>0`임에도 키워드 리스트가 비는 경우의 비율
- **목표**: 낮을수록 좋음 (단, `reason_code=KEYWORD_CORPUS_EMPTY`로 설명 필수)

---

### 3.4 Similarity Integrity Metrics

#### M-SIM-01: Similarity Relevance Pass Rate
- **정의**: 유사댓글 매칭 시 threshold/topic overlap gate를 통과한 비율
- **계산**:
  - 분모: `SIMILARITY_MATCHING` 이벤트 수
  - 분자: `matched_count>0` AND reason_code가 실패 코드가 아닌 케이스

#### M-SIM-02: Unrelated Match Detection Rate (Regression)
- **정의**: 골든셋에서 "유사하면 안 되는 케이스"에 유사댓글이 반환된 비율
- **목표**: **0%** (P0)
- **관련 acceptance**: INV-06, AC-11

#### M-SIM-03: Gibberish Eligibility Rate
- **정의**: 무의미/장난/나열형 텍스트가 similarity 대상이 되는 비율
- **목표**: 낮을수록 좋음
- **관련 acceptance**: AC-12

---

### 3.5 Representative Template Quality Metrics

#### M-REP-01: Template Compliance Rate
- **정의**: 대표 코멘트 출력이 템플릿 필수 항목을 모두 충족한 비율
- **필수 항목** (acceptance-criteria.md AC-08 기준):
  - Header: `Top{k} {stage} 단계 {sentiment} 코멘트`
  - Video title 라벨: `영상제목: ...`
  - 유사 의견 N건이면 최소 1개 이상 목록 제공
  - 요약 1~2문장 + 근거 발췌 1~2개
  - 메타라인 `???` 금지
- **목표**: **100%**

#### M-REP-02: Boilerplate Ratio
- **정의**: 대표 "설명"에서 중복/장문 boilerplate 비중
- **기준**: 설명 길이 상한 600 chars 초과 비율
- **목표**: 상한 초과 **0%**

---

## 4. 운영 메트릭(Operational Metrics)

### M-OPS-01: Exclude Rate by Reason
- `"excluded"` 비율과 reason_code 분포 (`PROMO_DETECTED`, `SPAM_GIBBERISH` 등)
- 목적: PROMO가 폭증하면 수집 소스 변화 신호

### M-OPS-02: Cache Staleness Rate
- `CACHE_STALE` 발생 빈도 (`CACHE_VERSION` 미업데이트 탐지)
- 목적: count/display mismatch의 전조 탐지

### M-OPS-03: Sanitization Action Frequency
- `REMOVE_CONTROL_CHARS` / `REPLACE_GARBLED` / `TRUNCATE` 수행 빈도
- 목적: 인코딩/데이터 오염 구간 탐지

---

## 5. Phase별 릴리즈 게이트(Release Gates)

> MVP 단계에서 모든 지표를 한 번에 통과하기는 어렵다.
> Phase별로 현실적인 기준을 단계적으로 적용한다.

### Phase 1 — MVP (현재 ~ 1개월)
배포 차단 조건 (P0):
| 메트릭 | 조건 |
|--------|------|
| M-INT-03 Promo Leakage | > 0 |
| M-INT-04 Dataset Fingerprint Mismatch | > 0 |
| M-TXT-02 Garbled Display Leakage | > 0 |

경고 조건 (P1):
| 메트릭 | 조건 |
|--------|------|
| M-INT-01 Count↔Rep Invariant | > 0 (경고 후 수정 진행) |
| M-INT-02 Label↔Rep Mismatch | > 0 (경고 후 수정 진행) |
| M-KWD-01 Keyword Noise | ≥ 10% |

---

### Phase 2 — 고도화 (1~3개월)
배포 차단 조건 (P0):
| 메트릭 | 조건 |
|--------|------|
| M-INT-01 Count↔Rep Invariant | > 0 |
| M-INT-02 Label↔Rep Mismatch | > 0 |
| M-INT-03 Promo Leakage | > 0 |
| M-INT-04 Dataset Fingerprint Mismatch | > 0 |
| M-TXT-02 Garbled Display Leakage | > 0 |
| M-SIM-02 Unrelated Match | > 0 |

경고 조건 (P1):
| 메트릭 | 조건 |
|--------|------|
| M-KWD-01 Keyword Noise | ≥ 5% |
| M-REP-01 Template Compliance | < 100% |

---

### Phase 3 — 인프라 (3개월 이후)
배포 차단 조건 (P0): Phase 2 전체 + 아래 추가
| 메트릭 | 조건 |
|--------|------|
| M-REP-01 Template Compliance | < 100% |
| M-KWD-02 Label Corpus Purity | < 100% |

---

## 6. 산출/리포팅 포맷(Reporting)
메트릭 리포트는 최소 2가지 형태를 제공한다.
1. **per_run 리포트**: `run_id` 기준 전체 메트릭 + reason_code top-N
2. **per_segment 리포트**: 제품/브랜드/CEJ/국가/기간 세그먼트별 핵심 메트릭 비교

---

## 7. Acceptance Criteria와의 연결(Traceability)
| Invariant/AC | 메트릭 |
|--------------|--------|
| INV-02/AC-01 (프로모 차단) | M-INT-03, M-OPS-01 |
| INV-03/AC-03~05 (count-display) | M-INT-01, M-INT-04 |
| INV-04/AC-06~07 (깨짐 차단) | M-TXT-01~03 |
| INV-05/AC-09~10 (키워드 품질) | M-KWD-01~03 |
| INV-06/AC-11~12 (유사댓글) | M-SIM-01~03 |
| AC-08 (대표 템플릿) | M-REP-01~02 |
