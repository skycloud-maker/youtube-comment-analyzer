# AGENTS.md

> 이 파일은 AI(하네스)와 사람이 함께 이 레포를 작업할 때
> 반드시 따라야 할 규칙을 정의합니다.
> 코드 작업 전 반드시 읽으세요.

## 1. 절대 규칙 (AI·사람 모두 적용)

- `data/` 폴더 하위 파일은 **절대 커밋하지 않는다**
  - 댓글 원문, parquet, pickle 캐시 모두 포함
  - `.gitignore`에 명시되어 있어야 함
- API 키, 유튜브 채널 ID, 개인식별 정보는 **코드에 하드코딩 금지**
  - 반드시 환경변수 또는 `.env` 파일 사용
- 댓글 작성자 개인정보(닉네임, 프로필 등)는 **분석·저장·노출 금지**

## 2. 개인정보 취급 규칙

- 수집 대상: 댓글 텍스트, 작성일시, 좋아요 수, 영상 ID만 허용
- 수집 금지: 작성자 채널 ID, 닉네임, 프로필 이미지, 이메일
- 번역된 댓글도 원문과 동일한 보안 수준으로 취급
- 로그에 댓글 원문을 출력하는 코드 작성 금지
  - 디버깅 시 `comment[:20]` 등 앞부분만 사용할 것

## 3. 코드 스타일

### Python
- 포맷터: `ruff format` (Black 호환)
- 린터: `ruff check`
- 타입힌트: 모든 함수에 입력/반환 타입 명시
  ```python
  # Good
  def build_keyword_summary(texts: tuple[str, ...], top_n: int = 6) -> pd.DataFrame:

  # Bad
  def build_keyword_summary(texts, top_n=6):
  ```
- 상수는 파일 상단에 대문자로 선언 (기존 COL_*, CACHE_* 패턴 유지)
- 컬럼명 문자열은 반드시 상수 사용, 하드코딩 금지
  ```python
  # Good
  df[COL_SENTIMENT]

  # Bad
  df["감성"]
  ```

### 네이밍
- 함수: `snake_case` (동사로 시작 — `build_`, `render_`, `load_`, `filter_`)
- 상수: `UPPER_SNAKE_CASE`
- Streamlit 렌더링 함수: `render_` 접두사 필수

## 4. PR / 코드 리뷰 규칙

- PR 제목 형식: `[타입] 설명`
  - 타입: `feat` / `fix` / `refactor` / `docs` / `chore`
  - 예: `[feat] 경쟁사 채널 브랜드 필터 추가`
- PR 본문에 반드시 포함할 것:
  - 변경 이유
  - 영향받는 화면/기능
  - 테스트한 방법 (스크린샷 권장)
- 데이터 파이프라인(`src/analytics/`) 수정 시
  → 반드시 사람이 리뷰 후 머지
- 대시보드 UI만 수정 시
  → AI 단독 작업 허용, 사람 확인 후 머지

## 5. 배포 전 체크리스트

### 필수 확인
- [ ] `data/` 관련 파일이 커밋에 포함되지 않았는가
- [ ] `.env` 또는 API 키가 코드에 노출되지 않았는가
- [ ] `ruff check` 통과했는가
- [ ] Streamlit Cloud 환경변수에 필요한 키가 모두 등록되어 있는가
- [ ] `CACHE_VERSION` 업데이트가 필요한 변경인가 (parquet 구조 변경 시)

### 기능 확인
- [ ] 국가 필터 (KR/US) 정상 동작
- [ ] 브랜드 필터 정상 동작
- [ ] 엑셀 다운로드 정상 동작
- [ ] 한글 폰트 Cloud 환경에서 깨지지 않는가

## 6. AI(하네스)에게 특별 지시

- 기존 상수(`COL_*`, `CACHE_*`, `SENTIMENT_LABELS` 등)는
  **이름 변경 금지** — 다른 모듈과 연결되어 있음
- `st.cache_data` / `st.cache_resource` 데코레이터
  **임의 제거 금지** — 성능에 직접 영향
- 새로운 parquet 컬럼 추가 시
  → `load_dashboard_data()`의 `buckets` 딕셔너리도 함께 업데이트
- 브랜드/제품 목록 변경은 상수 수정으로만 처리
  (`BRAND_FILTER_OPTIONS`, `PRIMARY_PRODUCT_FILTERS`)
