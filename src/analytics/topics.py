"""CEJ stage inference for appliance comments."""

from __future__ import annotations

CEJ_STAGE_KEYWORDS = {
    "인지": ["추천", "비교", "광고", "리뷰", "후기", "review", "compare", "worth", "브랜드"],
    "구매": ["구매", "결제", "가격", "할인", "쿠폰", "사은품", "order", "buy", "price", "deal"],
    "배송": ["배송", "도착", "출고", "shipping", "delivery", "late", "delay"],
    "설치": ["설치", "기사", "방문", "installation", "install", "setup", "mount"],
    "사용준비": ["설명서", "연결", "앱", "등록", "와이파이", "초기", "manual", "connect", "wifi", "app"],
    "사용": ["사용", "소음", "냄새", "성능", "세척", "건조", "세탁", "불편", "works", "noise", "smell", "performance"],
    "관리교체": ["청소", "필터", "교체", "수리", "as", "서비스", "관리", "clean", "replace", "repair", "service"],
}



def infer_topic(text: str) -> str:
    lowered = (text or "").lower()
    best_label = "기타"
    best_score = 0
    for label, keywords in CEJ_STAGE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label
