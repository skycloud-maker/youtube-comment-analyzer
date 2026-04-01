# Audit Log Spec — YouTube Appliance VoC Dashboard Harness (v0.1)

## 0. 목적(Purpose)
본 문서는 유튜브 가전 VoC 대시보드의 분석 파이프라인에서 발생하는 모든 핵심
의사결정(라벨링/제외/대표선정/요약/키워드/유사댓글)을 **추적 가능(Traceable)** 하도록,
최소한의 감사로그(Audit Log) 필드/이벤트/사유코드(reason codes)를 정의한다.

본 로그는 다음을 보장한다.
- "왜 이 댓글/영상이 제외됐는가?"
- "왜 대표 코멘트가 비어 있는가?"
- "깨진 문자열/???/비정상 키워드/유사댓글 불일치" 문제를 **원인 단위로 분해** 가능하게 함

> NOTE: 본 스펙은 acceptance-criteria.md의 INV/AC 항목을 충족하기 위한 운영/회귀 기반이다.

---

## 1. 적용 범위(Scope)
Audit Log는 아래 이벤트를 포함한다.
1. **Ingest/Normalize**: 수집, 전처리, 언어감지, 중복 제거
2. **Labeling**: 감성(`positive`/`negative`/`neutral`/`excluded`) 및 promo/상업성 탐지
3. **Eligibility**: 대표/키워드/유사댓글 계산의 후보군 적격성 판단
4. **Representative**: 대표 코멘트 선택/부재 처리
5. **Summary**: 주요 부정 포인트 등 요약 텍스트 생성 및 정제/대체
6. **Keyword**: 워드클라우드/도넛 키워드 생성 및 노이즈 필터링
7. **Similarity**: 유사 댓글 산정, threshold 통과 여부
8. **Rendering**: UI 표출 안전장치(sanitize) 적용 및 출력 대체(N/A)

---

## 2. 개인정보/보안 원칙(Privacy & Safety)
- Audit Log에는 **원문 전체(comment_text full)**를 저장하지 않는다.
- 필요 시, 근거 확인을 위한 **부분 발췌(snippet)**만 저장하며, 다음을 반드시 적용:
  - PII 탐지/마스킹 후 저장 (전화번호/이메일/주소/계정 패턴 등)
  - 최대 길이 제한 (200~300 chars)
- 원문 조회는 별도 접근 제어된 스토리지(parquet/DB)에 남기고,
  Audit Log는 참조 키(`comment_id`)를 통해 연결한다.

---

## 3. 공통 필드(Common Fields) — 모든 이벤트 필수

### 3.1 식별자/추적 (Identity & Traceability)
| 필드 | 타입 | 설명 |
|------|------|------|
| `event_id` | UUID | 이벤트 고유 ID |
| `event_time` | ISO-8601 | 발생 시각 |
| `run_id` | str | 파이프라인 실행 단위 ID (`CACHE_VERSION`과 연동) |
| `pipeline_version` | str | git commit hash 권장 |
| `rule_version` | str | harness 룰 버전 (semver, 예: `0.1.0`) |
| `prompt_version` | str | LLM 프롬프트 버전 (사용 시) |
| `model_id` | str | LLM 모델/엔진 식별자 (옵션) |

### 3.2 데이터 스코프 (Data Scope)
| 필드 | 타입 | 설명 |
|------|------|------|
| `scope_type` | str | `"comment"` \| `"video"` \| `"channel"` \| `"aggregate"` |
| `video_id` | str | YouTube video id (해당 시) |
| `channel_id` | str | YouTube channel id (해당 시) |
| `comment_id` | str | comment unique id (해당 시) |

### 3.3 필터 컨텍스트 (Filter Context)
| 필드 | 타입 | 설명 |
|------|------|------|
| `filters_applied` | JSON | 적용된 필터 요약 |
| `dataset_fingerprint` | str | 집계/대표/키워드가 동일 데이터셋을 참조하는지 검증용 해시 |

#### dataset_fingerprint 계산 방법 (구현 규칙)
count/display mismatch 추적의 핵심이다. 아래 방식으로 계산한다.

```python
import hashlib, json
from pathlib import Path

def compute_dataset_fingerprint(
    parquet_dir: Path,
    filters: dict,
) -> str:
    """
    parquet 파일들의 수정 시각(mtime) + 적용 필터 파라미터를 합쳐 SHA-256 해시를 반환.
    집계(counts)와 대표 선정(representative)이 동일 fingerprint를 갖는지
    비교하여 데이터셋 불일치를 탐지한다.
    """
    parts = []

    # 1) parquet 파일들의 mtime (정렬하여 순서 고정)
    for p in sorted(parquet_dir.rglob("*.parquet")):
        parts.append(f"{p.name}:{p.stat().st_mtime_ns}")

    # 2) 필터 파라미터 (정렬하여 순서 고정)
    parts.append(json.dumps(filters, sort_keys=True, ensure_ascii=False))

    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]  # 앞 16자리만 사용
```

> 주의: `load_dashboard_data()`의 캐시(pickle)와 대표 선정 로직이
> 반드시 동일 `dataset_fingerprint`를 사용해야 한다.
> fingerprint가 다르면 `reason_code=DATASET_FINGERPRINT_MISMATCH`로 기록한다.

필터 예시:
```json
{
  "filters_applied": {
    "valid_only": true,
    "exclude_promo": true,
    "lang": ["ko", "en"],
    "date_range": "2026-01-01/2026-03-31",
    "product": "냉장고",
    "brand": "LG",
    "region": "KR"
  }
}
```

### 3.4 결과/사유 (Result & Reason)
| 필드 | 타입 | 설명 |
|------|------|------|
| `decision` | str | `PASS` \| `FAIL` \| `EXCLUDE` \| `SANITIZED` \| `SKIP` \| `SELECTED` \| `EMPTY` |
| `reason_code` | str | 아래 Reason Code Taxonomy 중 하나 |
| `reason_detail` | str | 한 줄 설명 (최대 200 chars) |
| `severity` | str | `P0` \| `P1` \| `P2` |

---

## 4. 이벤트 타입(Event Types) & 추가 필드

### EVT-01 INGEST_RESULT
| 필드 | 설명 |
|------|------|
| `source` | `"youtube_api"` \| `"scrape"` |
| `status` | `"SUCCESS"` \| `"ERROR"` |
| `error_type`, `error_message` | 에러 시 |
| `raw_length` | 원문 길이(문자수) |
| `lang_detected` | `"ko"` \| `"en"` \| `"unknown"` |

### EVT-02 PROMO_DETECTION
| 필드 | 설명 |
|------|------|
| `promo_flag` | bool |
| `promo_signals` | 탐지 근거 키워드/패턴 목록 |
| `label_override` | `"excluded"` \| null (코드 내 실제 값 사용) |

### EVT-03 SENTIMENT_LABEL
| 필드 | 설명 |
|------|------|
| `sentiment_label` | `"positive"` \| `"negative"` \| `"neutral"` \| `"excluded"` |
| `sentiment_confidence` | 0~1 (옵션) |
| `sentiment_rationale_snippet` | 근거 발췌 (마스킹/길이 제한 적용) |

### EVT-04 ELIGIBILITY_CHECK
| 필드 | 설명 |
|------|------|
| `target` | `"representative"` \| `"keyword"` \| `"similarity"` \| `"summary"` |
| `eligible` | bool |
| `ineligible_reasons` | reason_code list |
| `content_quality_flags` | `["GARBLED_TEXT","GIBBERISH","TOO_SHORT","ONLY_URL",...]` |

### EVT-05 REPRESENTATIVE_SELECTION
| 필드 | 설명 |
|------|------|
| `slot` | `"positive"` \| `"neutral"` \| `"negative"` (코드 내 실제 값 사용) |
| `stage` | CEJ stage (`고객여정단계` 컬럼 값) |
| `count_in_slot` | 해당 슬롯 댓글 수 |
| `candidate_count` | 후보 수 (필터 후) |
| `selected_comment_id` | 선택 시 |
| `selected_snippets` | 근거 발췌 1~2개 (마스킹/길이 제한) |
| `template_version` | 대표 템플릿 버전 |
| `template_fields_missing` | 누락된 필드 목록 |

> 대표 부재 시 (`decision=EMPTY`): `reason_code`는 반드시
> `NO_VALID_CANDIDATE` / `COUNT_ZERO` / `FILTERED_AS_PROMO` / `LABEL_MISMATCH` 중 하나.

### EVT-06 SUMMARY_TEXT_GENERATION
| 필드 | 설명 |
|------|------|
| `summary_type` | `"major_negative_points"` \| `"video_summary"` |
| `raw_summary` | 생성된 요약 (sanitize 후 최대 길이 제한) |
| `sanitized_summary` | 표출용 정제 텍스트 |
| `sanitization_actions` | `["REMOVE_CONTROL_CHARS","REPLACE_GARBLED","TRUNCATE",...]` |
| `rendered_value` | UI 최종 표시값 (`"N/A"` 포함) |

### EVT-07 KEYWORD_EXTRACTION
| 필드 | 설명 |
|------|------|
| `slot` | `"negative"` \| `"positive"` \| `"neutral"` |
| `corpus_size` | 토큰화 대상 댓글 수 |
| `noise_terms_removed` | 제거된 노이즈 단어 샘플 (상위 N개) |
| `top_keywords` | 최종 상위 키워드 (JSON) |
| `stopwords_version` | stopwords 리스트 버전 |

### EVT-08 SIMILARITY_MATCHING
| 필드 | 설명 |
|------|------|
| `algorithm` | `"tfidf_cosine"` \| `"embedding_cosine"` |
| `threshold` | 적용 임계치 |
| `candidate_pool_size` | 비교 대상 수 |
| `matched_count` | 매칭 수 |
| `matched_comment_ids` | 상위 k개 id |
| `matched_scores` | 점수 목록 |
| `gibberish_flag` | 원문이 무의미 텍스트인지 여부 |

### EVT-09 RENDERING_GUARDRAIL
| 필드 | 설명 |
|------|------|
| `field_name` | `"major_negative_points"` \| `"representative_metadata"` 등 |
| `input_value_sample` | sanitize/trim 후 저장 |
| `output_value` | UI 표출값 |
| `blocked` | bool |
| `block_reason` | reason_code |

---

## 5. Reason Code Taxonomy

### 커버리지/수집
| Code | 설명 |
|------|------|
| `COVERAGE_MISSING_HOT_VIDEO` | 핫 영상 누락 감지 |
| `CHANNEL_NOT_IN_SCOPE` | 채널이 수집 범위 밖 |
| `INGEST_API_ERROR` | API 오류로 수집 실패 |

### 상업성/스팸/제외
| Code | 설명 |
|------|------|
| `PROMO_DETECTED` | 제휴/광고/구매유도 탐지 |
| `PINNED_COMMERCIAL` | 고정댓글 상업성 |
| `SPAM_GIBBERISH` | 무의미/장난/스팸성 텍스트 |
| `EXCLUDED_BY_POLICY` | 정책적 제외 (룰 기반) |

### 감성/라벨 무결성
| Code | 설명 |
|------|------|
| `LABEL_ASSIGNED` | 라벨링 정상 (정보성) |
| `LABEL_MISMATCH` | 대표 슬롯 라벨과 후보 라벨 불일치 |
| `LABEL_OVERRIDDEN_PROMO` | promo 탐지로 `"excluded"` override |

### 집계-표출 불일치
| Code | 설명 |
|------|------|
| `COUNT_ZERO` | count=0 |
| `DATASET_FINGERPRINT_MISMATCH` | 집계/대표/키워드 참조 데이터셋 불일치 |
| `CACHE_STALE` | `CACHE_VERSION` 미업데이트로 캐시 오래됨 |
| `FILTER_ORDER_MISMATCH` | 필터 적용 순서가 달라 결과 어긋남 |

### 대표 선정
| Code | 설명 |
|------|------|
| `SELECTED_TOP_CANDIDATE` | 대표 선정 성공 |
| `NO_VALID_CANDIDATE` | 필터 후 후보 없음 |
| `FILTERED_AS_PROMO` | promo로 모두 제거됨 |
| `SANITIZED_EMPTY` | 정제 결과가 빈 값이 됨 |
| `TEMPLATE_FIELD_MISSING` | 템플릿 필수 필드 결측 |

### 텍스트 깨짐/정제
| Code | 설명 |
|------|------|
| `GARBLED_TEXT_DETECTED` | `???`/replacement/control chars 탐지 |
| `SANITIZED_REPLACED_WITH_NA` | `"N/A"`로 대체 |
| `ENCODING_ERROR` | 인코딩/디코딩 오류 |

### 키워드 품질
| Code | 설명 |
|------|------|
| `NOISE_TERM_REMOVED` | 노이즈 제거 수행 |
| `KEYWORD_CORPUS_EMPTY` | 해당 라벨 코퍼스가 비어 있음 |
| `KEYWORD_CONTAINS_NOISE` | 노이즈가 상위에 남아 품질 실패 |

### 유사댓글 무결성
| Code | 설명 |
|------|------|
| `SIMILARITY_BELOW_THRESHOLD` | 임계치 미달 |
| `TOPIC_OVERLAP_LOW` | 주제 겹침 낮음 |
| `GIBBERISH_INPUT` | 원문이 무의미 텍스트 |
| `NO_ELIGIBLE_CANDIDATES` | 비교 대상 없음 |

---

## 6. 저장 포맷(Storage Format)
권장: JSON Lines (1 event per line).

```json
{
  "event_id": "uuid-...",
  "event_time": "2026-03-31T12:34:56+09:00",
  "run_id": "run-20260331-001",
  "pipeline_version": "git:abcdef1",
  "rule_version": "0.1.0",
  "scope_type": "comment",
  "video_id": "abc123",
  "comment_id": "cmt987",
  "filters_applied": {"valid_only": true, "exclude_promo": true},
  "dataset_fingerprint": "9f2a3b4c5d6e7f8a",
  "event_type": "PROMO_DETECTION",
  "decision": "EXCLUDE",
  "reason_code": "PROMO_DETECTED",
  "reason_detail": "Pinned affiliate link detected (쿠팡파트너스/링크).",
  "severity": "P0",
  "promo_flag": true,
  "promo_signals": ["쿠팡파트너스", "링크", "고정댓글"]
}
```

---

## 7. Acceptance Criteria와의 연결(Traceability)
| 현상 | Reason Code |
|------|-------------|
| Promo 댓글이 대표/키워드에 섞임 | `PROMO_DETECTED` / `FILTERED_AS_PROMO` |
| count=0인데 대표 표시 | `COUNT_ZERO` + `DATASET_FINGERPRINT_MISMATCH` 의심 |
| ???/깨짐 문자열 표출 | `GARBLED_TEXT_DETECTED` + `SANITIZED_REPLACED_WITH_NA` |
| 유사 댓글 연관성 붕괴 | `GIBBERISH_INPUT` 또는 `SIMILARITY_BELOW_THRESHOLD` |
| 캐시 관련 불일치 | `CACHE_STALE` (`CACHE_VERSION` 확인 필요) |
