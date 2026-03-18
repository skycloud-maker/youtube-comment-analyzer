"""Keyword extraction utilities with appliance-domain normalization."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

import pandas as pd

KOREAN_STOPWORDS = {
    "그리고", "그러나", "근데", "그런데", "하지만", "또한", "또", "그래서", "그래도", "왜", "정말", "진짜", "너무",
    "약간", "조금", "그냥", "영상", "댓글", "제품", "경우", "부분", "정도", "관련", "사용", "이거", "저거", "그거",
    "이건", "저는", "우리는", "진짜로", "같아요", "합니다", "했어요", "합니다만", "하는데", "있어요", "없어요",
    "같음", "같다", "같은", "때문", "정도", "하나", "둘", "셋", "이번", "저번", "오늘", "내일", "어제",
    "여기", "저기", "그냥요", "이런", "그런", "저런", "좀", "많이", "매우", "혹시", "아마", "일단",
    "그리고요", "그래요", "네", "아니요", "입니다", "있습니다", "없습니다", "하세요", "해요", "했음",
}

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "did", "do", "does",
    "done", "for", "from", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "on", "or", "our", "ours", "ourselves", "out", "over", "really", "so", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves", "years", "year", "month", "months",
    "day", "days", "thing", "things", "something", "anything", "everything", "review", "reviews", "video", "appliance",
    "brand", "washer", "dryer", "fridge", "refrigerator", "dishwasher", "machine", "machines", "appliances",
    "im", "its", "dont", "cant", "didnt", "doesnt", "wasnt", "werent", "shouldnt", "wouldnt", "couldnt", "ive",
    "youre", "theyre", "thats", "theres", "whats", "ill", "id", "weve", "isnt", "arent", "wont", "hadnt",
}

GENERIC_EXCLUDE = {"문제", "이슈", "상태", "부분", "경험", "느낌", "후기"}

STOPWORDS = KOREAN_STOPWORDS | ENGLISH_STOPWORDS | GENERIC_EXCLUDE

KEYWORD_TRANSLATIONS = {
    "noise": "소음",
    "noisy": "소음",
    "loud": "소음",
    "quiet": "정숙성",
    "vibration": "진동",
    "vibrates": "진동",
    "delivery": "배송",
    "shipping": "배송",
    "install": "설치",
    "installation": "설치",
    "setup": "설치",
    "price": "가격",
    "pricing": "가격",
    "expensive": "가격부담",
    "refund": "환불",
    "return": "반품",
    "repair": "수리",
    "service": "고객지원",
    "support": "고객지원",
    "broken": "고장",
    "issue": "문제",
    "issues": "문제",
    "problem": "문제",
    "problems": "문제",
    "defect": "불량",
    "defective": "불량",
    "leak": "누수",
    "capacity": "용량",
    "smell": "냄새",
    "odor": "냄새",
    "cleaning": "세척",
    "drying": "건조성능",
    "washing": "세탁성능",
    "energy": "전기세",
    "bill": "전기세",
    "filter": "필터",
    "door": "도어",
    "rack": "선반",
    "cycle": "코스",
    "stain": "얼룩",
    "performance": "성능",
    "quality": "품질",
    "manual": "설명서",
    "wifi": "와이파이",
    "app": "앱연결",
    "cold": "냉각",
    "cooling": "냉각",
    "ice": "얼음",
    "drain": "배수",
    "draining": "배수",
}

BUSINESS_KEEPWORDS = {
    "가격", "가격부담", "가성비", "배송", "설치", "사용성", "편의성", "세척", "건조성능", "세탁성능", "냄새",
    "소음", "진동", "발열", "고장", "불량", "문제", "누수", "환불", "반품", "수리", "고객지원", "전기세",
    "도어", "필터", "용량", "코스", "크기", "디자인", "내구성", "마감", "설치기사", "A/S", "AS", "브랜드",
    "얼룩", "품질", "성능", "앱연결", "와이파이", "설명서", "청소", "교체", "선반", "건조시간", "세탁시간",
    "소독", "살균", "건조", "세척력", "냉각", "온도", "얼음", "소비전력", "정숙성", "배수", "물튐",
    "설치비", "소비전력", "보증", "AS", "A/S", "앱", "연동", "물샘", "탈수", "소음문제", "용량부족",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+|[0-9A-Za-z가-힣+/]+")


def normalize_keyword(token: str) -> str:
    token = str(token or "").strip().lower()
    if not token:
        return ""
    token = token.strip(".,!?()[]{}\"'`“”’:;/-_")
    token = token.replace("'", "")
    token = KEYWORD_TRANSLATIONS.get(token, token)
    if token in STOPWORDS or token.isdigit():
        return ""
    if len(token) < 2:
        return ""
    if re.search(r"[^a-z0-9가-힣+/]", token):
        return ""
    if re.fullmatch(r"[a-z]+", token):
        return ""
    return token



def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(str(text or "")):
        token = normalize_keyword(raw_token)
        if token:
            tokens.append(token)
    return tokens



def build_keyword_counter(texts: Iterable[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text(str(text or "")))
    return counter



def filter_business_keywords(counter: Counter[str], top_n: int) -> list[tuple[str, int]]:
    filtered = [(keyword, count) for keyword, count in counter.items() if keyword in BUSINESS_KEEPWORDS or bool(re.search(r"[가-힣]", keyword))]
    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered[:top_n]



def extract_keywords(texts: pd.Series, top_n: int = 30) -> pd.DataFrame:
    counter = build_keyword_counter(texts.fillna(""))
    rows = [{"keyword": token, "count": count} for token, count in filter_business_keywords(counter, top_n)]
    return pd.DataFrame(rows)
