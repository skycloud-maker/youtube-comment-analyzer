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

`nlp_analyzer`를 함께 쓰는 로컬 환경이라면 아래를 추가로 설치하세요.
```bash
pip install -r requirements.nlp-analyzer.txt
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

## Dashboard Modes (Sample vs Real)

Streamlit dashboard now has two intentionally separated flows:

- `샘플 로드`:
  - 빠른 UI/차트/필터 검증용
  - 최신 run 일부를 축약 로드하거나, 결과 파일이 없으면 내장 synthetic sample을 로드
  - 긴 수집/분석 작업 없이 즉시 확인 가능
- `실데이터 새로고침`:
  - 실제 `data/processed/*` 결과를 다시 읽어 최신 상태 반영
  - 캐시를 무효화한 뒤 재로드
- `실분석 실행`:
  - 대시보드에서 `python -m src.main run ...`을 실행해 실제 수집+분석 수행
  - 완료 후 실데이터 모드로 자동 갱신

### Dashboard entrypoints

- 기본(Strategy-first 레이어): `streamlit run streamlit_app.py`
  - 새 정보 구조: `전략 이슈 → 우선순위/액션 → 근거 드릴다운`
- 레거시(검증/비교 레이어): `streamlit run streamlit_app_legacy.py`
  - 기존 `src/dashboard_app.py` 화면을 그대로 유지

### Startup loading behavior

- 결과 파일(`videos_normalized.parquet`, `comments.parquet`)이 존재하면 앱 시작 시 자동 로딩합니다.
- 데이터는 `st.session_state`에 실제 bundle(dict[str, DataFrame]) 자체를 저장해 rerun 시 유지합니다.
- 버튼은 보조 동작(샘플 전환/실데이터 새로고침/실분석 실행)이며, 버튼 내부 임시 변수에만 의존하지 않습니다.
- 실데이터 모드에서 필수 결과 파일이 없으면 누락 파일명을 포함한 경고를 표시합니다.

### NLP responsibility split

- Layer 1 (`src/pipelines/analytics_pipeline.py` + `src/analytics/*`):
  - 댓글 단위 분류(감성, 문의 여부 등) 단일 기준 산출
- Layer 2 (Legacy UI: `src/dashboard_app.py`):
  - 검증/회귀 체크용 레거시 대시보드
- Layer 3 (Strategy-first UI: `src/dashboard_v2.py`):
  - 전략 이슈를 랜딩 객체로 먼저 제시하고, 대표/유사/원천 댓글은 근거 드릴다운으로 제공
  - 댓글 단위 재분석 금지 (대시보드 내부 별도 NLP 분기 제거)

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

대시보드 아키텍처 관련 스모크 테스트만 빠르게 돌리려면:
```bash
pytest tests/test_dashboard_architecture.py tests/test_streamlit_ui_smoke.py
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
