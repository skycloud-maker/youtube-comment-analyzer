# docs/exec-plans/tech-debt-tracker.md

> 기술 부채 목록입니다.
> 하네스 작업 시 이 목록을 참고하여 개선 작업을 병행하세요.
> 신규 부채 발견 시 이 파일에 추가합니다.

---

## 우선순위 기준

- 🔴 High — 지금 당장 해결 안 하면 운영 리스크
- 🟡 Medium — Phase 2 전까지 해결 권장
- 🟢 Low — Phase 3 또는 여유 있을 때

---

## 🔴 High

### TD-001 — API 키 하드코딩 위험
- **현상**: YouTube API 키 및 채널 ID가 코드에 직접 박혀있을 가능성
- **위험**: 레포 공개 시 키 노출 → API 할당량 탈취
- **해결**: 전체 코드에서 API 키 검색 후 환경변수(`YOUTUBE_API_KEY`)로 전환
- **참조**: AGENTS.md 2번 규칙

### TD-002 — CACHE_VERSION 수동 관리
- **현상**: parquet 구조 변경 시 `CACHE_VERSION` 문자열을
  개발자가 직접 수정해야 함 (`CACHE_VERSION = "2026-03-27-0215"`)
- **위험**: 업데이트 누락 시 구버전 캐시로 대시보드 오작동
- **해결**: parquet 파일들의 mtime 해시를 자동 계산하여
  `_latest_signature()`로 대체
- **참조**: ARCHITECTURE.md — Known Issues

### TD-003 — data/ 폴더 .gitignore 미확인
- **현상**: `data/processed/`, `data/dashboard_cache/` 등
  댓글 원문 데이터가 실수로 커밋될 위험
- **위험**: 개인정보 포함 댓글 원문 GitHub 노출
- **해결**: `.gitignore` 즉시 확인 및 아래 항목 추가
  ```
  data/processed/
  data/dashboard_cache/
  data/mplcache/
  .env
  ```
- **참조**: AGENTS.md 1번 규칙

---

## 🟡 Medium

### TD-004 — 한글 컬럼명 혼용
- **현상**: `COL_SENTIMENT = "감성"`, `COL_COUNTRY = "국가"` 등
  DataFrame 컬럼명이 한글
- **위험**: DB 이전(PostgreSQL) 시 전면 리팩터링 필요,
  영문 환경 도구와 연동 시 인코딩 오류 가능
- **해결**: Phase 3 DB 이전 시점에 영문으로 일괄 전환
  (`sentiment`, `country`, `brand` 등)
  그 전까지는 `COL_*` 상수 패턴 철저히 유지
- **참조**: db-schema.md — 마이그레이션 주의사항

### TD-005 — 브랜드/제품 목록 하드코딩
- **현상**: `BRAND_FILTER_OPTIONS = ["LG", "Samsung", ...]`,
  `PRIMARY_PRODUCT_FILTERS = ["냉장고", "세탁기", ...]`가
  코드 상단에 고정값으로 선언
- **위험**: 경쟁사 추가/제거 시 코드 배포 필요
- **해결**: 설정 파일(`config.yaml`) 또는 환경변수로 외부화
- **참조**: AGENTS.md 6번 규칙

### TD-006 — 수집 파이프라인 수동 실행
- **현상**: 댓글 수집을 로컬에서 수동으로 실행 후
  parquet을 레포에 업로드하는 방식
- **위험**: 수집 누락, 데이터 신선도 저하
- **해결**: 대시보드 내 "수집 실행" 버튼 구현
- **참조**: Phase 2 — data-pipeline-automation.md

### TD-007 — 폰트 경로 중복 선언
- **현상**: `FONT_CANDIDATES` 리스트와 `_get_font_file()` 함수가
  사실상 동일한 역할을 코드 두 곳에서 중복 수행
- **해결**: `_get_font_file()` 단일 함수로 통합,
  `FONT_CANDIDATES` 상수 제거

### TD-008 — 캐시 fallback 로직 복잡도
- **현상**: `load_dashboard_data()` 안에
  로컬 캐시 → 구버전 캐시 → Cloud 캐시 → parquet 직접 로드
  순서로 4단계 fallback이 중첩
- **위험**: 디버깅 어려움, 의도치 않은 구버전 데이터 로드
- **해결**: fallback 단계를 명확히 로깅하고
  단계별 실패 사유를 `st.warning()`으로 노출

---

## 🟢 Low

### TD-009 — 타입힌트 누락
- **현상**: 일부 함수에 입력/반환 타입힌트 없음
- **해결**: `ruff check` + 타입힌트 일괄 추가
- **참조**: AGENTS.md 3번 규칙

### TD-010 — 테스트 코드 부재
- **현상**: `src/analytics/` 모듈에 단위 테스트 없음
- **위험**: 감성 분석 로직 변경 시 회귀 감지 불가
- **해결**: Phase 2 완료 후 핵심 함수부터 pytest 작성
  우선순위: `analyze_comment_with_context`,
  `build_keyword_counter`, `filter_business_keywords`

### TD-011 — Streamlit Cloud 타임아웃 미대응
- **현상**: 대용량 수집 실행 시 10분+ 소요되면
  Streamlit Cloud 타임아웃 발생 가능
- **해결**: 수집을 청크 단위로 나누거나
  Phase 3 서버 이전 시 GitHub Actions로 전환
- **참조**: data-pipeline-automation.md — 제약 사항

### TD-012 — PDF 출력 미구현
- **현상**: 임원 보고용 PDF 출력 기능 없음
- **해결**: Phase 2에서 구현
- **참조**: pdf-export.md

---

## 해결 이력

| ID | 해결일 | 내용 |
|----|--------|------|
| - | - | - |

> 해결된 항목은 위 표로 이동 후 목록에서 제거
