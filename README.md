# YouTube Comment Analysis

공식 YouTube Data API v3만 사용해서 YouTube 영상 검색, 댓글/대댓글 전체 수집, 정규화, 분석, Excel 리포트, Markdown 요약 보고서를 생성하는 Python 프로젝트입니다.

## Overview
- 목적: 키워드 기반으로 YouTube 영상을 찾고, 비즈니스 사용자가 바로 볼 수 있는 댓글 분석 산출물을 만듭니다.
- 수집 범위: `search.list`, `videos.list`, `commentThreads.list`, `comments.list`
- 비목표: 브라우저 스크래핑, Selenium, Playwright, HTML 파싱, 비공식 API 사용
- 기본 사용자: 한국어 기반 비즈니스 분석가

## Architecture
- `src/api`: 공식 API 호출, 재시도, 쿼터 추적
- `src/collectors`: 비디오 검색과 댓글/대댓글 수집
- `src/pipelines`: 수집, 정규화, 분석, 내보내기 오케스트레이션
- `src/analytics`: 텍스트 정제, 감성, 키워드, 토픽, 클러스터링
- `src/exporters`: Excel 워크북과 Markdown 요약 보고서 작성
- `tests`: 핵심 변환 로직 테스트

## Project Structure
```text
project_root/
  README.md
  .env.example
  pyproject.toml
  config/
    settings.example.yaml
  data/
    raw/
    processed/
    exports/
    logs/
  src/
    __init__.py
    main.py
    cli.py
    config.py
    logger.py
    api/
      __init__.py
      youtube_client.py
      quota.py
    collectors/
      __init__.py
      video_search.py
      comments_collector.py
    pipelines/
      __init__.py
      ingest_pipeline.py
      normalize_pipeline.py
      analytics_pipeline.py
      export_pipeline.py
    models/
      __init__.py
      schemas.py
    analytics/
      __init__.py
      descriptive.py
      text_cleaning.py
      sentiment.py
      topics.py
      keywords.py
      clustering.py
    exporters/
      __init__.py
      excel_writer.py
      report_writer.py
    utils/
      __init__.py
      retry.py
      dates.py
      io.py
  tests/
    test_normalization.py
    test_text_cleaning.py
    test_quota_logic.py
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
copy .env.example .env
```

## Environment Variables
- `YOUTUBE_API_KEY`: YouTube Data API v3 key. 필수입니다.
- `YT_ANALYSIS_SETTINGS_PATH`: YAML 설정 경로. 기본값 `config/settings.example.yaml`
- `YT_ANALYSIS_ENV`: 실행 환경 식별자. 기본값 `local`

## How To Get A YouTube API Key
1. Google Cloud Console에서 새 프로젝트를 생성합니다.
2. `YouTube Data API v3`를 활성화합니다.
3. `Credentials`에서 API key를 생성합니다.
4. 생성한 값을 `.env`의 `YOUTUBE_API_KEY`에 넣습니다.
5. 필요하면 API key 제한을 프로젝트 환경에 맞게 설정합니다.

## Quota Caveats
- `search.list`는 1회 호출당 100 units라서 가장 비쌉니다.
- 이 프로젝트는 실행 전에 예상 쿼터를 계산하고, 실제 호출 건수도 메타데이터에 저장합니다.
- 검색 결과는 로컬 캐시에 저장해서 같은 조건 재실행 시 불필요한 검색 비용을 줄입니다.
- 대댓글은 `commentThreads.list`의 내장 응답이 불완전할 수 있어 `comments.list(parentId=...)`로 보완합니다.

## Commands
### Full run
```bash
python -m src.main run \
  --keyword "삼성 냉장고" \
  --max-videos 100 \
  --comments-per-video 1000 \
  --published-after 2025-01-01T00:00:00Z \
  --published-before 2026-01-01T00:00:00Z \
  --order relevance \
  --language ko \
  --region KR \
  --include-replies true \
  --output-prefix samsung_fridge
```

### Search only
```bash
python -m src.main search-videos --keyword "삼성 냉장고" --max-videos 100
```

### Collect comments for an existing run
```bash
python -m src.main collect-comments --run-id <RUN_ID> --comments-per-video 1000 --include-replies true
```

### Analyze existing collected data
```bash
python -m src.main analyze --run-id <RUN_ID>
```

### Export Excel and Markdown report
```bash
python -m src.main export --run-id <RUN_ID> --output-prefix samsung_fridge
```

## Outputs
- `data/raw/<run_id>/...`: 엔드포인트별 원본 JSON 감사 로그
- `data/processed/<run_id>/videos_raw.parquet`: 수집 비디오 원본 테이블
- `data/processed/<run_id>/comments_raw.parquet`: 댓글 원본 테이블
- `data/processed/<run_id>/videos_normalized.parquet`: 정규화 비디오 테이블
- `data/processed/<run_id>/comments_normalized.parquet`: 정규화 댓글 테이블
- `data/processed/<run_id>/run_manifest.json`: 실행 메타데이터
- `data/exports/<run_id>/<prefix>_<run_id>.xlsx`: Excel 산출물
- `data/exports/<run_id>/<prefix>_<run_id>_summary.md`: Markdown 요약 보고서

## Excel Workbook
다음 시트를 생성합니다.
- `README`
- `Runs_Summary`
- `Videos`
- `Comments`
- `Video_Ranking`
- `Keyword_Analysis`
- `Sentiment_Summary`
- `Topic_Summary`
- `Representative_Comments`
- `Errors_Log`

추가로 다음 서식을 적용합니다.
- freeze panes
- auto filter
- 자동 열 너비 조정
- 조건부 서식 기반 감성 강조
- 감성 분포 차트
- 워크북 메타데이터

## Testing
```bash
pytest
```

## Troubleshooting
- `Configuration error`: `.env`에 `YOUTUBE_API_KEY`가 없는 경우입니다.
- `403` 또는 `commentsDisabled`: 해당 영상은 댓글이 비활성화되었거나 접근 권한이 제한된 경우입니다. 오류 로그 시트를 확인하세요.
- `parquet` 저장 실패: 선택 의존성이 없는 환경이면 CSV로 자동 폴백합니다.
- 일일 쿼터 초과 우려: 검색 건수와 영상 수를 줄이거나 기존 검색 캐시를 재사용하세요.

## Compliance Note
이 프로젝트는 공식 YouTube Data API v3만 사용합니다. 브라우저 자동화, 스크래핑, HTML 파싱, 비공식 API 호출은 포함하지 않습니다.

## Future Extensions
- OpenAI 기반 요약/분류 모듈 추가
- 도메인별 위험 키워드 사전 확장
- 브랜드/채널 간 비교 리포트 고도화


## Harness Validation & Golden Set
- Harness validation docs live under `docs/harness/validation/`.
- Golden set seeds live in `docs/harness/validation/goldenset_seed.yaml`.
- Golden regression tests live in `tests/golden/`.
- Local commands:
  - `pytest -q`
  - `pytest -q tests/golden`
  - `pytest --cov=src -q`
  - `python scripts/validate_dashboard.py`
  - `python scripts/validate_coverage.py`
- Replace `TBD_*` placeholders in `goldenset_seed.yaml` with real `video_id` / `comment_id` values once they are confirmed from processed data.
