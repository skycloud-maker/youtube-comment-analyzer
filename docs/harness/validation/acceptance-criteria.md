# Acceptance Criteria — YouTube Appliance VoC Dashboard Harness (v0.1)

## 0. 목적(Purpose)
본 문서는 유튜브 가전 VoC 대시보드가 **대표성(coverage)**, **무결성(integrity)**,
**설명가능성(explainability)**, **가독성(readability)**을 만족하는지 검증하기 위한
**수용 기준(acceptance criteria)**을 정의한다.

본 기준은 기능 개발보다 상위의 "하네스 원칙(헌법)"이며,
모든 릴리즈/리빌드 시 회귀 테스트의 기준점으로 사용한다.

---

## 1. 범위(Scope)
다음 산출물에 적용한다.
- 댓글 감성 라벨 (부정/중립/긍정/제외) 결과
- 영상 단위 요약 (요약 테이블, 주요 부정 포인트 등)
- 대표 코멘트 (부정/긍정/중립) 선정 및 출력 템플릿
- 워드클라우드/도넛 (라벨별 대표 키워드)
- 유사 댓글 목록 (similar comments)
- 집계(count) ↔ 표출(display) 일관성
- 텍스트 인코딩/정제 및 UI 표출 안전장치

---

## 2. 용어 정의(Definitions)

> ⚠️ 코드 내 실제 값과 반드시 일치시킨다 (`SENTIMENT_LABELS` 상수 기준).

| 용어 | 코드 내 실제 값 | 설명 |
|------|----------------|------|
| **Label** | `"positive"` / `"negative"` / `"neutral"` / `"excluded"` | `감성` 컬럼 값 |
| **Promo/Commercial** | `"excluded"` (promo_flag=True) | 제휴/광고/링크유도/구매유도/쿠팡파트너스/고정댓글 |
| **Representative Comment** | `representative_comments.parquet` | 특정 라벨/CEJ stage 대표 코멘트 |
| **Counts** | `brand_ratio.parquet`, `reporting_summary.parquet` | 라벨별 댓글 집계 |
| **Display** | Streamlit UI 렌더링 결과 | 실제 화면에 표시되는 값 |
| **Garbled Text** | `???`, `\ufffd`, 제어문자 | 비정상 인코딩으로 인한 깨짐 문자열 |
| **Similarity List** | `representative_bundles.parquet` | 원문과 의미적으로 관련된 댓글 목록 |
| **CEJ Stage** | `고객여정단계` 컬럼 | 인지/탐색/결정/구매/배송/사용준비/사용/관리/교체/기타 |

---

## 3. 핵심 불변조건(Invariants) — "깨지면 즉시 실패"
아래 불변조건은 모든 화면/리포트/내보내기에서 항상 만족해야 한다.

### INV-01 (Label Integrity)
Representative Comment는 **반드시 동일 라벨 bucket**에서만 선택되어야 한다.
- `"positive"` 대표는 `감성 == "positive"` 댓글에서만 뽑는다.
- 라벨 불일치 대표 선정은 금지.

### INV-02 (Promo Exclusion)
Promo/Commercial (광고/제휴/링크유도/고정댓글 등)은
- 라벨링 단계에서 `"excluded"` + `promo_flag=True`로 분리되어야 하며,
- 어떤 경우에도 `positive`/`negative`/`neutral` 대표 코멘트 후보가 될 수 없다.

### INV-03 (Count ↔ Display Consistency)
라벨별 집계(count)와 대표 코멘트 표출(display)은 **동일 기준의 필터링/데이터셋**을 사용해야 한다.
- `count(label) == 0` → `representative(label)` MUST be None (표시 금지)
- `count(label) > 0` 이지만 대표 선정 실패 → `"대표 없음"` + `reason_code` 표기
  (임의 코멘트로 대체 금지)

### INV-04 (Text Safety)
UI에 출력되는 모든 텍스트는 Garbled Text를 포함하면 안 된다.
- 출력 전 sanitize/normalize 수행
- 여전히 비정상 문자열이면 `"N/A"`로 대체하고 `reason_code` 기록

### INV-05 (Keyword Relevance)
워드클라우드/도넛의 "부정/긍정 대표 키워드"는
- 해당 라벨 코퍼스에서만 추출되어야 하며,
- 아래 노이즈 단어 목록(`NOISE_TERMS`)이 상위 키워드에 노출되면 실패로 간주한다.

```python
# 코드에서 반드시 사용해야 할 최소 노이즈 단어 목록
NOISE_TERMS = [
    "클릭", "링크", "제품보기", "구입", "구매", "쿠팡", "파트너스",
    "네이버", "쇼핑", "커넥트", "수수료", "고정댓글", "할인",
    "체감가", "가격", "품절", "배송", "주문취소", "추천제품",
    "요약", "영상", "구독", "좋아요", "댓글", "채널", "문의",
]
# 위 목록은 최소값이며, 운영 중 확장 가능하다.
# src/analytics/keywords.py 의 filter_business_keywords() 와 연동할 것.
```

### INV-06 (Similarity Integrity)
유사 댓글 목록은 원문과 **의미적/주제적 연관성**이 최소 기준을 만족해야 한다.
- 기준 미달이면 `"유사 댓글 없음"` 처리
- 무의미/장난/나열형 텍스트는 유사도 계산에서 제외 또는 별도 처리한다.

---

## 4. 수용 기준(Acceptance Tests)
각 테스트는 최소 1개 이상의 자동화 테스트(pytest) 또는 재현 가능한 검증 스크립트로 구현되어야 한다.

### AC-01 Promo/Commercial Filtering
**Given** 고정댓글/제휴링크/구매유도/쿠팡파트너스/네이버쇼핑커넥트 문구가 포함된 댓글이 존재할 때
**When** 감성 라벨링 및 대표 코멘트 선정 파이프라인을 실행하면
**Then**
1. 해당 댓글은 `감성 == "excluded"` 로 분류되어야 하며
2. `positive`/`negative`/`neutral` 대표 후보군에 포함되면 안 된다.

*Failure linked*: Case #2 | *Invariant*: INV-02

---

### AC-02 Representative Label Match (Hard Constraint)
**Given** 라벨이 이미 부여된 댓글 집합이 있을 때
**When** 라벨별 대표 코멘트를 선정하면
**Then** 대표 코멘트의 `감성` 값은 반드시 해당 슬롯의 라벨과 동일해야 한다.

*Failure linked*: Case #7 | *Invariant*: INV-01

---

### AC-03 Count = 0 → No Representative
**Given** 특정 영상/필터 조건에서 `positive` count=0일 때
**When** 대표 코멘트 영역을 렌더링하면
**Then** `positive` 대표 코멘트는 표시되면 안 된다 (빈 값 또는 숨김).

*Failure linked*: Case #3 | *Invariant*: INV-03

---

### AC-04 Count > 0 but Representative Missing → Explain
**Given** `negative` count>0인데 대표 선정이 실패할 수 있는 조건이 있을 때
**When** UI가 대표 영역을 출력하면
**Then** `"대표 없음"` + `reason_code` (예: `NO_VALID_CANDIDATE` / `FILTERED_AS_PROMO` / `SANITIZED_EMPTY`)를
표기해야 하며, 임의 댓글을 대체로 끌고 오면 안 된다.

*Failure linked*: Case #3 | *Invariant*: INV-03

---

### AC-05 Same Dataset for Counts and Representatives
**Given** 특정 화면의 집계 테이블과 대표 코멘트가 같은 필터를 공유할 때
**When** 동일 조건으로 계산하면
**Then** counts 및 대표 후보군은 동일 `dataset_fingerprint`를 가진 데이터셋을 참조해야 한다.
- `dataset_fingerprint` 계산: `parquet 파일들의 mtime` + `적용 필터 파라미터`를 SHA-256 해시
- `load_dashboard_data()` 캐시와 대표 선정 로직이 동일 번들을 참조하는지 검증

*Failure linked*: Case #3 | *Invariant*: INV-03

---

### AC-06 Garbled Text Must Not Be Displayed (Major Negative Points)
**Given** "주요 부정 포인트" 요약 텍스트를 생성/표출할 때
**When** 결과 문자열에 `???`, `\ufffd`, 제어문자 등 Garbled Text 패턴이 포함되면
**Then** UI는 이를 직접 출력하지 말고 `"N/A"`로 대체 + `reason_code=GARBLED_TEXT_DETECTED`를 남겨야 한다.

*Failure linked*: Case #4 | *Invariant*: INV-04

---

### AC-07 Garbled Text Must Not Be Displayed (Representative Metadata)
**Given** 대표 코멘트 메타 라인 (제품/단계/브랜드/라벨 등)을 렌더링할 때
**When** 필드 결측/인코딩 문제로 `???` 같은 값이 생성되면
**Then** `"N/A"` 처리 또는 안전한 기본값을 사용하고, `???` 출력은 금지한다.

*Failure linked*: Case #6-2 | *Invariant*: INV-04

---

### AC-08 Representative Template Compliance
**Given** 대표 코멘트 출력 템플릿을 적용할 때
**Then** 반드시 아래 형식을 만족해야 한다:

```
Top{k} {stage} 단계 {sentiment} 코멘트
영상제목: {title}
{product} - 고객여정단계: {stage} - 브랜드: {brand} - 감성: {sentiment} - 유사 의견: {n}건
요약: (1~2문장)
근거 발췌: (1~2개 짧은 인용)
유사 의견 예시: (n>0이면 최대 3개 샘플/ID/발췌)
```

규칙:
1. Header: `Top{k} {stage} 단계 {sentiment} 코멘트`
2. Video title 라벨: `영상제목: {title}`
3. `유사 의견 N건`이면 **목록(최대 3개 샘플 또는 ID/발췌)** 제공
4. 설명 영역은 중복/장문 boilerplate 금지: **요약 1~2문장 + 근거 발췌 1~2개**
5. 원문 전체는 `"원문 전체 보기"`로 분리하고 대표 영역에는 **부분 발췌**만 표시
6. 메타 필드 결측 시 `"N/A"` 출력 (??? 금지)

*Failure linked*: Case #6-1~5 | *Invariant*: INV-04

---

### AC-09 Keyword Noise Removal (Wordcloud/Donut)
**Given** 라벨별 대표 키워드를 생성할 때
**When** 상위 키워드에 `NOISE_TERMS` (INV-05 정의) 목록의 단어가 포함되면
**Then** 해당 키워드들은 제거되거나 하위로 밀려야 하며, 상위 노출 시 실패로 간주한다.
- `src/analytics/keywords.py` 의 `filter_business_keywords()` 함수에 `NOISE_TERMS` 반영 필수

*Failure linked*: Case #5, Case #2 | *Invariant*: INV-05

---

### AC-10 Keyword Must Be Label-Specific
**Given** `negative` 키워드 / `positive` 키워드를 생성할 때
**Then** 각 키워드는 해당 라벨 코퍼스(`감성 == "negative"` 필터 후)에서만 추출되어야 하며,
다른 라벨 코퍼스의 단어가 혼입되면 실패로 간주한다.

*Failure linked*: Case #5, Case #7 | *Invariant*: INV-05

---

### AC-11 Similarity Must Be Semantically Related
**Given** 원문 댓글과 유사 댓글 목록을 생성할 때
**When** 유사 댓글이 주제/의미적 연관성 기준을 만족하지 못하면
**Then** 유사 댓글 목록은 빈 목록이어야 하고 `reason_code=SIMILARITY_BELOW_THRESHOLD` 사유를 제공해야 한다.

*Failure linked*: Case #8 | *Invariant*: INV-06

---

### AC-12 Noise/Gibberish Comments Handling (Similarity)
**Given** 무의미 나열/장난/스팸/의미 없는 텍스트가 입력일 때
**Then**
- 유사도 계산 대상에서 제외하거나 별도 처리한다.
- 의미적 연관성이 없는 유사 댓글을 강제로 매칭하면 실패로 간주한다.
- `reason_code=GIBBERISH_INPUT` 반환 필수

*Failure linked*: Case #8 | *Invariant*: INV-06

---

### AC-13 Coverage Sanity Check (Hot/Important Videos)
**Given** 특정 기간/제품군에서 "핫"한 영상이 존재할 때
**When** 대시보드 수집/분석 결과를 생성하면
**Then** 그 영상이 누락되는 경우, 반드시 `reason_code=COVERAGE_MISSING_HOT_VIDEO`가 기록되어야 하며
재수집/포함 여부를 판단할 수 있어야 한다.

*Failure linked*: Case #1

---

## 5. 실패 등급(Severity)

| 등급 | 조건 | 처리 |
|------|------|------|
| **P0** | INV-01~INV-06 위반, Case #2/#3/#4/#7/#8 | 즉시 수정, 배포 금지 |
| **P1** | 템플릿 미준수, 키워드 노이즈, 표기 혼란 | 릴리즈 전 수정 |
| **P2** | 커버리지 확장 (핫 영상 자동 포함) 등 | 운영 고도화 과제 |

### Phase별 릴리즈 게이트 (현실적 기준)

| Phase | P0 조건 | P1 조건 |
|-------|---------|---------|
| **Phase 1 (MVP)** | INV-02, INV-03, INV-04 위반 시 차단 | INV-01, INV-05, INV-06 경고 |
| **Phase 2 (고도화)** | INV-01~INV-06 전체 위반 시 차단 | 템플릿/키워드 기준 강화 |
| **Phase 3 (인프라)** | 전체 AC 100% 통과 필수 | — |

---

## 6. 필수 로깅(Audit / Reason Codes)
모든 제외/대표없음/정제대체 발생 시 다음을 기록한다.
- `comment_id` / `video_id` / `stage` / `sentiment_slot`
- `decision` (EXCLUDE/PROMO/SANITIZED/NONE)
- `reason_code` (아래 목록)
- `prompt_version` / `rule_version` / `pipeline_version`
- `timestamp`

세부 필드는 `audit-log-spec.md` 참조.

---

## 7. 추적성(Traceability)

| Case | Acceptance Test | Invariant |
|------|-----------------|-----------|
| Case #1 (핫 영상 누락) | AC-13 | — |
| Case #2 (Promo → Positive) | AC-01, AC-09 | INV-02, INV-05 |
| Case #3 (Count↔대표 불일치) | AC-03~AC-05 | INV-03 |
| Case #4 (텍스트 깨짐) | AC-06 | INV-04 |
| Case #5 (키워드 노이즈) | AC-09~AC-10 | INV-05 |
| Case #6 (대표 템플릿 미준수) | AC-07~AC-08 | INV-04 |
| Case #7 (라벨↔대표 불일치) | AC-02, AC-10 | INV-01 |
| Case #8 (유사댓글 무결성) | AC-11~AC-12 | INV-06 |
