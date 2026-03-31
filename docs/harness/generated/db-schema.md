# Database Schema

> **현재 상태**: Parquet 파일 기반 (data/processed/{run_id}/)
> **이전 목표**: PostgreSQL (시점 미정)
> **이 문서의 역할**: 현재 데이터 구조 정의 + 향후 DB 스키마 설계 기준

---

## 현재 데이터 구조 (Parquet)

### comments.parquet — 핵심 테이블

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| text_display | str | 화면 표시용 댓글 원문 |
| text_original | str | 수집 원본 댓글 (가공 전) |
| title | str | 영상 제목 |
| video_url | str | 영상 링크 |
| like_count | int | 댓글 좋아요 수 |
| language_detected | str | 감지된 언어 코드 (ko, en 등) |
| removed_reason | str | 제거 사유 (없으면 null) |
| 감성 | str | positive / negative / neutral / excluded |
| 국가 | str | KR / US |
| 브랜드 | str | LG / Samsung / GE / Whirlpool 등 |
| 고객여정단계 | str | 인지 / 탐색 / 결정 / 구매 / 배송 / 사용 / 관리 / 교체 / 기타 |
| product | str | 냉장고 / 세탁기 / 건조기 등 |
| 작성일시 | datetime | 댓글 작성 시각 |
| 주차시작 | date | 해당 주 월요일 날짜 |
| 주차표기 | str | 화면 표시용 주차 레이블 |
| 연관 맥락 | str | 부모 댓글 또는 영상 맥락 |
| 연관 맥락 번역 | str | 맥락 한국어 번역 |
| run_id | str | 수집 회차 ID |

### videos.parquet — 영상 메타

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| video_id | str | YouTube 영상 ID |
| title | str | 영상 제목 |
| published_at | datetime | 영상 게시일 |
| view_count | int | 조회수 |
| like_count | int | 좋아요 수 |
| comment_count | int | 댓글 수 |
| 브랜드 | str | 채널 브랜드 |
| 국가 | str | KR / US |
| run_id | str | 수집 회차 ID |

### opinion_units.parquet — 의견 단위 (댓글 분할)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| source_comment_id | str | 원본 댓글 ID |
| opinion_text | str | 분리된 의견 단위 텍스트 |
| sentiment_label | str | positive / negative / neutral |
| product | str | 연관 제품 |
| run_id | str | 수집 회차 ID |

### 분석 집계 테이블들

| 파일명 | 설명 |
|--------|------|
| brand_ratio.parquet | 브랜드별 언급 비중 |
| cej_negative_rate.parquet | CEJ 단계별 부정 비율 |
| negative_density.parquet | 기간별 부정 밀도 |
| new_issue_keywords.parquet | 신규 이슈 키워드 |
| persistent_issue_keywords.parquet | 지속 이슈 키워드 |
| quality_summary.parquet | 수집 품질 요약 |
| representative_comments.parquet | 대표 댓글 |
| representative_bundles.parquet | 대표 댓글 묶음 |
| monitoring_summary.parquet | 모니터링 KPI 요약 |
| reporting_summary.parquet | 임원 보고용 KPI 요약 |

---

## 향후 PostgreSQL 스키마 설계

### 마이그레이션 원칙
- parquet 컬럼명 → snake_case 영문으로 전환
- run_id → `collection_runs` 테이블로 분리
- 한글 컬럼명 전면 영문화

### 테이블 구조 (목표)

```sql
-- 수집 회차
CREATE TABLE collection_runs (
    id          UUID PRIMARY KEY,
    run_at      TIMESTAMPTZ NOT NULL,
    brand       VARCHAR(50),
    region      VARCHAR(10),
    note        TEXT
);

-- 영상
CREATE TABLE videos (
    id              UUID PRIMARY KEY,
    youtube_id      VARCHAR(20) UNIQUE NOT NULL,
    title           TEXT,
    published_at    TIMESTAMPTZ,
    view_count      INT,
    like_count      INT,
    comment_count   INT,
    brand           VARCHAR(50),
    region          VARCHAR(10),
    run_id          UUID REFERENCES collection_runs(id)
);

-- 댓글 (핵심)
CREATE TABLE comments (
    id                  UUID PRIMARY KEY,
    video_id            UUID REFERENCES videos(id),
    text_display        TEXT,
    text_original       TEXT,
    language_detected   VARCHAR(10),
    like_count          INT,
    written_at          TIMESTAMPTZ,
    sentiment           VARCHAR(20),   -- positive/negative/neutral/excluded
    removed_reason      TEXT,
    region              VARCHAR(10),
    brand               VARCHAR(50),
    product             VARCHAR(50),
    cej_stage           VARCHAR(20),
    week_start          DATE,
    run_id              UUID REFERENCES collection_runs(id)
);

-- 의견 단위
CREATE TABLE opinion_units (
    id                  UUID PRIMARY KEY,
    comment_id          UUID REFERENCES comments(id),
    opinion_text        TEXT,
    sentiment_label     VARCHAR(20),
    product             VARCHAR(50),
    run_id              UUID REFERENCES collection_runs(id)
);
```

### 마이그레이션 시 주의사항
- `dashboard_bundle.pkl` 캐시 무효화 필요
- `load_dashboard_data()` 함수를 DB 쿼리로 교체
- Streamlit `@st.cache_data` TTL 전략 재수립 필요
- 한글 컬럼 상수(`COL_*`) → 영문으로 일괄 교체 필요
