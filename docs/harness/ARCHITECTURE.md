# Architecture

## System Overview

YouTube Data API → 수집 파이프라인 → Parquet 저장 → Streamlit 대시보드
                                                              ↓
                                                    Streamlit Cloud 배포

## Tech Stack

| 레이어 | 기술 |
|--------|------|
| 데이터 수집 | YouTube Data API v3 (Python) |
| 데이터 저장 | Parquet 파일 (로컬 / 레포 내) |
| 캐싱 | Pickle 기반 dashboard_bundle 캐시 |
| 분석 | Pandas, 감성분석 모델 (다국어 지원) |
| 시각화 | Altair, Matplotlib (한글 폰트: NanumGothic) |
| 대시보드 | Streamlit |
| 배포 | Streamlit Cloud |

## Directory Structure

```
project/
├── src/
│   ├── analytics/
│   │   ├── comment_analysis.py   # 감성 분석 (긍/부정/중립/제외)
│   │   └── keywords.py           # 키워드 추출/정규화
│   └── utils/
│       └── translation.py        # 다국어 → 한국어 번역
├── data/
│   ├── processed/
│   │   └── {run_id}/             # 수집 회차별 폴더
│   │       ├── comments.parquet
│   │       ├── videos.parquet
│   │       ├── brand_ratio.parquet
│   │       ├── cej_negative_rate.parquet
│   │       ├── opinion_units.parquet
│   │       └── ... (총 14종 parquet)
│   ├── dashboard_cache/
│   │   ├── dashboard_bundle.pkl       # 로컬 캐시
│   │   └── dashboard_bundle_cloud.pkl # Cloud 캐시
│   └── mplcache/                      # Matplotlib 캐시
├── fonts/
│   └── NanumGothic.ttf
└── dashboard.py                       # 메인 Streamlit 앱
```

## Data Pipeline

```
1. 수집
   YouTube Data API v3
   → 채널별 영상 목록 + 댓글 수집
   → 대상: 자사 2~3개 브랜드 + 경쟁사 (LG, Samsung, GE, Whirlpool + 해외)

2. 전처리
   → 언어 감지 (다국어)
   → 한국어 번역
   → 스팸/노이즈 제거 → removed_reason 기록

3. 분석
   → 감성 분류: positive / negative / neutral / excluded
   → CEJ 단계 태깅: 인지→탐색→결정→구매→배송→사용→관리→교체→기타
   → 제품 분류: 냉장고, 세탁기, 건조기, 식기세척기, 청소기, 오븐 등
   → 키워드 추출: 신규 이슈 / 지속 이슈 분리
   → 의견 단위(opinion_unit) 분리: 댓글 1개 → 복수 의견으로 분할

4. 저장
   → data/processed/{run_id}/ 하위 14종 parquet으로 저장
   → 수집 회차(run_id)별로 누적 관리

5. 캐싱
   → 최초 로드 시 전체 parquet 병합 → pickle 캐시 생성
   → CACHE_VERSION + 파일 signature로 캐시 유효성 검증
```

## Dashboard Screens

```
메인 대시보드
├── 필터: 국가(KR/US) / 브랜드 / 제품 / 기간
├── Tab 1: 댓글 VoC 대시보드
│   ├── KPI 카드 (분석댓글수 / 유효댓글수 / 긍부정 비율)
│   ├── 주차별 긍정/부정 트렌드 차트 (Altair)
│   ├── 부정/긍정 키워드 패널
│   ├── CEJ 단계별 고객 신뢰도 지수
│   ├── 브랜드 언급 비중 + 신뢰도
│   └── 대표 댓글 뷰어
└── Tab 2: 영상 분석 요약
    └── 영상별 댓글 현황

출력
└── 전체 Raw Data 엑셀 다운로드 (.xlsx)
```

## Key Constraints

- **캐시 의존성**: parquet이 바뀌면 CACHE_VERSION 수동 업데이트 필요
- **폰트 이중화**: Cloud(NanumGothic.ttf) + 로컬(malgun.ttf) 둘 다 지원
- **다국어 처리**: 해외 채널 댓글은 번역 후 감성 분석 적용
- **수집 주기**: 현재 수동 실행 (자동화 미적용 — 향후 개선 대상)

## Known Issues / Tech Debt

- [ ] 수집 파이프라인 자동화 (현재 수동)
- [ ] 캐시 버전 관리 수동화 → 자동 감지로 개선 필요
- [ ] 경쟁사 채널 확장 시 브랜드 필터 하드코딩 수정 필요
