# PLANS.md

## 전체 로드맵 개요

```
Phase 1 — MVP        현재 ~ 1개월
Phase 2 — 고도화     1개월 ~ 3개월
Phase 3 — 인프라     3개월 이후 (시점 미정)
```

---

## Phase 1 — MVP (현재 ~ 1개월)

**목표**: 기존 코드 기반을 문서화된 구조로 정리하고,
핵심 대시보드를 Streamlit Cloud에서 안정적으로 운영

### 완료 기준
- [ ] 감성 분석 대시보드 — 필터 + KPI + 차트 정상 동작
- [ ] 엑셀 Raw Data 다운로드 정상 동작
- [ ] CEJ 단계별 신뢰도 테이블 정상 렌더링
- [ ] 브랜드별 언급 비중 테이블 정상 렌더링
- [ ] 자사 2~3개 채널 + 국내 경쟁사 댓글 수집 가능
- [ ] Streamlit Cloud 배포 안정화
- [ ] AGENTS.md 기준 코드 스타일 정리
- [ ] 환경변수로 API 키 관리 (하드코딩 제거)
- [ ] 한글 폰트 Cloud 환경에서 정상 적용

### 이번 단계에서 하지 않는 것
- PDF 출력 (Phase 2)
- 경쟁사 해외 채널 (Phase 2)
- 수집 자동화 (Phase 2)
- DB 이전 (Phase 3)

---

## Phase 2 — 고도화 (1개월 ~ 3개월)

**목표**: 경쟁사 비교 + PDF 출력 + CEJ 고도화로
임원 보고 수준의 대시보드 완성

### 기능 목록 (우선순위 순)

| 우선순위 | 기능 | 참조 스펙 |
|---------|------|-----------|
| 1 | 경쟁사 채널 비교 화면 | competitor-comparison.md |
| 2 | CEJ × 제품 히트맵 | cej-analysis.md |
| 3 | PDF 보고서 출력 | pdf-export.md |
| 4 | 수집 파이프라인 자동화 | data-pipeline-automation.md |
| 5 | 해외 채널 다국어 댓글 수집 | data-pipeline-automation.md |

### 완료 기준
- [ ] 브랜드별 감성 비율 바 차트
- [ ] 부정 키워드 비교 테이블 (공통 키워드 하이라이트)
- [ ] CEJ × 브랜드 히트맵
- [ ] 시계열 감성 트렌드 라인 차트
- [ ] 섹션 선택 기반 PDF 생성 + 한글 폰트 정상 적용
- [ ] 대시보드 내 "수집 실행" 버튼 + 진행 상태 표시
- [ ] GE, Whirlpool 등 해외 채널 댓글 수집 + 번역

---

## Phase 3 — 인프라 (3개월 이후)

**목표**: Parquet 기반 구조를 DB로 이전하여
대용량 데이터와 다중 사용자 환경 대응

### 기능 목록

| 기능 | 참조 문서 |
|------|-----------|
| PostgreSQL 이전 | db-schema.md |
| 수집 스케줄러 자동화 (GitHub Actions 또는 별도 서버) | data-pipeline-automation.md |
| 캐시 전략 재수립 (TTL 기반) | architecture.md |
| 한글 컬럼명 → 영문 전면 리팩터링 | db-schema.md |

### 진입 조건
- Phase 2 완료 후 데이터 누적량이 parquet으로
  감당하기 어려운 수준에 도달했을 때
- 또는 동시 접속 사용자가 늘어 성능 이슈 발생 시

---

## 현재 상태 스냅샷

| 항목 | 상태 |
|------|------|
| 대시보드 메인 코드 | ✅ 구현됨 |
| 감성 분석 | ✅ 구현됨 |
| CEJ 테이블 | ✅ 구현됨 |
| 키워드 분석 | ✅ 구현됨 |
| 엑셀 다운로드 | ✅ 구현됨 |
| 경쟁사 비교 화면 | 🔜 Phase 2 |
| PDF 출력 | 🔜 Phase 2 |
| CEJ 히트맵 | 🔜 Phase 2 |
| 수집 자동화 | 🔜 Phase 2 |
| PostgreSQL 이전 | 🔜 Phase 3 |

---

## 하네스 작업 순서 가이드

```
지금 당장 시작할 것 (Phase 1)
1. AGENTS.md 규칙 숙지
2. 환경변수 정리 (API 키 하드코딩 제거)
3. 코드 스타일 정리 (ruff, 타입힌트)
4. Streamlit Cloud 배포 안정화 확인

Phase 1 완료 후 시작할 것 (Phase 2)
1. competitor-comparison.md 스펙 기준 경쟁사 비교 화면
2. cej-analysis.md 기준 히트맵 추가
3. pdf-export.md 기준 PDF 출력 구현
4. data-pipeline-automation.md 기준 수집 버튼 구현
```
