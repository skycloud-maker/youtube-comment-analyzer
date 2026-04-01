# Golden Set — YouTube Appliance VoC Dashboard Harness (v0.1)

## 0. 목적(Purpose)
Golden Set은 유튜브 가전 VoC 대시보드가 버전업/재실행/프롬프트 변경을 하더라도
**핵심 원칙(acceptance-criteria)과 무결성(integrity)이 유지되는지**를 검증하기 위한
"정답/기준 샘플 묶음"이다.

> 핵심: Golden Set은 "통계가 좋아 보이는지"가 아니라
> "이런 실패(8사례)가 다시는 재발하지 않는지"를 보장한다.

---

## 1. 범위(Scope)
Golden Set은 아래 4종 유형을 반드시 포함한다.
1. **정상 케이스**: 대표 코멘트/키워드/유사댓글이 자연스럽게 잘 나오는 샘플
2. **경계 케이스**: 애매하지만 정책으로 처리해야 하는 샘플
3. **실패 재현 케이스**: 지금까지 발견된 8개 사례를 재현/고정하는 샘플
4. **악성/노이즈 케이스**: promo/스팸/장난/무의미 텍스트 등

---

## 2. 크기(Size) & 성장 전략
- **v0.x 초기**: 최소 10~20개로 시작
- 각 실패 유형당 최소 1개 (8개 사례 → 최소 8개) + 정상 케이스 2~12개
- **v1.0 이후**: 100~300개 (제품/국가/브랜드/CEJ/언어 균형 고려)

---

## 3. 선정 원칙(Selection Principles)

### G-PR-01 대표성
실제 의사결정에 중요한 유형을 포함한다.

### G-PR-02 실패 고정(Regression Anchoring)
"한 번이라도 문제였던 유형"은 반드시 Golden Set에 들어가야 한다.

### G-PR-03 최소정보/보안
- 원문 전체를 저장하지 않는다.
- `comment_id`로 참조하고, **부분 발췌(snippet, 최대 300 chars)**만 저장한다.
- PII (전화번호/이메일/계정명 등)는 반드시 마스킹한다.

---

## 4. 노이즈 단어 목록(NOISE_TERMS) — 공식 정의

아래 목록은 `acceptance-criteria.md INV-05`와 `goldenset_seed.yaml`의
`keyword_checks.noise_topk_ok` 판단에 사용한다.
`src/analytics/keywords.py`의 `filter_business_keywords()`와 반드시 연동한다.

```python
NOISE_TERMS = [
    # 상업성/광고
    "클릭", "링크", "제품보기", "구입", "구매", "쿠팡", "파트너스",
    "네이버", "쇼핑", "커넥트", "수수료", "고정댓글", "할인",
    "체감가", "품절", "주문취소", "추천제품",
    # 유튜브 UI 관련 (감성과 무관)
    "영상", "구독", "좋아요", "댓글", "채널", "문의",
    # 중립적 기능어
    "요약", "가격",
]
# 위 목록은 최소값이며, 운영 중 확장 가능하다.
# 변경 시 rule_version을 올리고 goldenset_seed.yaml의 관련 케이스를 재검증한다.
```

---

## 5. Golden Item 스키마(Schema)

`goldenset_seed.yaml` 형식 기준:

```yaml
golden_id: GS-XXXX                    # "GS-" + 4자리 숫자
video_id: <YouTube video ID>           # TBD_* 허용 (초기)
comment_id: <comment ID>               # TBD_* 허용 (초기)
language: ko | en | mixed | unknown
case_tags:                             # 아래 Tag Taxonomy 참고
  - TAG_NAME
input_snippet: "<원문 발췌, PII 마스킹, 최대 300 chars>"
expected:
  sentiment_label: positive|negative|neutral|excluded  # 코드 내 실제 값 사용
  promo_flag: true|false
  eligible_for_representative: true|false
  eligible_for_keywords: true|false
  eligible_for_similarity: true|false
  representative_slot: positive|negative|neutral|null
  representative_template_checks:
    header_ok: true|false         # "Top{k} {stage} 단계 {sentiment} 코멘트"
    video_title_label_ok: true|false  # "영상제목: ..."
    no_garbled_ok: true|false     # ??? 금지
    snippet_only_ok: true|false   # 원문 전체 아닌 발췌만
    similar_list_ok: true|false   # n>0이면 목록 존재
  keyword_checks:
    noise_topk_ok: true|false     # NOISE_TERMS가 top-K에 없어야 true
    label_purity_ok: true|false   # 라벨별 코퍼스 분리 여부
  similarity_checks:
    should_return_empty: true|false
    min_topic_overlap_ok: true|false
  reason_code: <Reason Code>      # audit-log-spec.md Taxonomy 참조
notes: "<사람이 보는 설명>"
```

### Tag Taxonomy (case_tags)
| Tag | 설명 | 관련 AC |
|-----|------|---------|
| `COVERAGE_MISSING_HOT_VIDEO` | 핫 영상 누락 | AC-13 |
| `PROMO` | 제휴/광고 | AC-01 |
| `PINNED_COMMERCIAL` | 고정댓글 상업성 | AC-01 |
| `LABEL_REP_MISMATCH` | 라벨↔대표 불일치 | AC-02 |
| `COUNT_REP_MISMATCH` | count↔대표 불일치 | AC-03~05 |
| `GARBLED_TEXT` | 텍스트 깨짐 | AC-06~07 |
| `KEYWORD_NOISE` | 키워드 노이즈 | AC-09 |
| `KEYWORD_LABEL_PURITY` | 키워드 라벨 순도 | AC-10 |
| `TEMPLATE_NON_COMPLIANCE` | 템플릿 미준수 | AC-08 |
| `SIMILARITY_IRRELEVANT` | 유사댓글 무관련 | AC-11 |
| `GIBBERISH_INPUT` | 무의미 텍스트 입력 | AC-12 |

---

## 6. 운영 프로세스(Governance & Workflow)

### 추가(Add)
1. 신규 이슈 발견 → `case_tags` 분류 → Golden Item 생성
2. `acceptance-criteria.md`에 AC 항목 매핑
3. `metrics.md`에서 관련 지표 감지 여부 확인
4. PR에 Golden Set 업데이트 포함 (테스트와 함께)

### 변경(Change)
- 변경 이유(정책 변경) 기록
- `rule_version` 상승
- 이전 결과와의 차이를 `notes`에 기록

### 삭제(Remove)
- 삭제는 원칙적으로 금지
- 불가피하면 `deprecated: true` 표시 후 유지

---

## 7. TBD_* 교체 방법
1. 대시보드 parquet에서 해당 케이스의 실제 ID 조회
   - `data/processed/{run_id}/comments.parquet` 에서 `comment_id` 확인
   - 영상은 `videos.parquet` 에서 `video_id` 확인
2. `goldenset_seed.yaml`의 `TBD_*` 값을 실제 ID로 치환
3. `input_snippet`을 실제 댓글 부분 발췌(PII 마스킹 후, 최대 300 chars)로 업데이트
4. `pytest -q tests/golden` 재실행으로 통과 확인

---

## 8. 파일 배치(Repository Layout)
```
docs/harness/validation/
├── acceptance-criteria.md   ← 불변조건 + AC 기준
├── audit-log-spec.md        ← 이벤트/필드/reason code
├── metrics.md               ← 계량 지표 + Phase별 릴리즈 게이트
├── goldenset.md             ← 본 문서 (정책/스키마/NOISE_TERMS)
└── goldenset_seed.yaml      ← 실제 seed 항목 (GS-0001~GS-0008)

tests/golden/
├── test_golden_regressions.py  ← 회귀 테스트 코드
└── conftest.py                 ← pytest 픽스처

.github/workflows/
└── tests.yml                   ← CI 자동화
```

---

## 9. 성공 기준(Success Criteria)
- 모든 릴리즈에서 Golden Set 회귀 테스트가 **100% 통과**해야 한다.
- 새 결함 발견 시 최소 1개 Golden Item을 추가하여 재발 방지를 "고정"한다.
- `NOISE_TERMS` 변경 시 관련 Golden Item(`KEYWORD_NOISE` 태그)을 재검증한다.
