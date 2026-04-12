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

# UI/interaction noise: YouTube UI terms + common non-business comment patterns
UI_NOISE_TERMS = {
    # UI/interaction
    "링크", "제품보기", "바로가기", "클릭", "상세보기", "자세히보기",
    "상단", "하단", "좌측", "우측", "중앙", "왼쪽", "오른쪽",
    "있음", "없음", "가능", "불가", "추천이유", "체감가",
    "보기", "구독", "좋아요", "알림", "추천", "공유", "재생",
    "채널", "동영상", "영상보기", "확인", "누르", "눌러",
    # Emoticons/reactions
    "ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "ㅎㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㅠㅠㅠ", "ㅜㅜㅜ",
    "ㄱㄱ", "ㄴㄴ", "ㄹㄹ", "ㅇㅇ", "ㅎㅎㅎㅎ", "ㅋㅋㅋㅋ",
    # Polite/reaction phrases (not business insight)
    "감사", "감사합니다", "감사해요", "고맙습니다", "수고하셨습니다", "수고하세요",
    "잘봤습니다", "잘봤어요", "잘보고", "잘보겠습니다", "잘보고갑니다",
    "맞습니다", "맞아요", "그렇죠", "그렇네요", "그렇구나", "그렇군요",
    "궁금합니다", "궁금해요", "궁금한데", "알겠습니다", "알겠어요",
    "주세요", "해주세요", "알려주세요", "보여주세요",
    # Common verb/adjective stems (no business meaning alone)
    "좋은", "좋아", "좋고", "좋다", "좋네", "좋겠", "좋을", "좋아요", "좋습니다",
    "보고", "봤는데", "봤어", "봤다", "보면", "봐서",
    "쓰고", "쓰는", "쓰면", "써보", "써본",
    "하고", "했는데", "했다", "했어", "해서", "하니", "하면서", "한번",
    # Intensifiers/adverbs (no business insight as standalone)
    "아주", "엄청", "완전", "되게", "거의", "별로", "딱히", "꽤", "꽤나",
    # Temporal/demonstrative (not product keywords)
    "이번에", "요즘", "최근", "계속", "다시", "아직", "먼저",
    "처음", "다음", "마지막", "이후", "이전", "앞으로", "앞으로도",
    # Verbal endings and connectives
    "아닌", "보는", "있는", "되는", "하는", "없는", "했는",
    "하면", "되면", "나는", "됩니다", "했습니다", "습니다",
    "하나요", "인가요", "건가요", "인데요", "거든요", "잖아요", "인데",
    "근데요", "그러면", "그래서요", "그런거", "이런거",
    "다만", "단지", "역시", "물론",
    # People/pronoun noise
    "사람", "분들", "여러분", "우리", "저도", "나도", "자기", "누구",
    # Non-specific terms
    "생각", "가지", "나오", "들어", "시작", "해보", "해야",
    "리뷰", "광고", "광고했", "광고했다", "광고했다고", "모델",
    # YouTuber/person names (not product keywords)
    "잇섭", "잇섭님", "곽튜브", "곽튜브님", "유튜버", "유튜브",
    # Non-actionable descriptors
    "확실히", "확실", "자주", "바로", "자가", "고려",
    "사용중", "구입", "구입했", "고장나서", "귀찮겠네요",
    # User-reported noisy tokens
    "샀는데", "샀다", "샀어요", "있는데", "없이", "없어서", "있는데요", "없는데",
}

# Brand/product terms that shouldn't be separate keywords (already in brand filter)
BRAND_NOISE_TERMS = {
    "삼성", "엘지", "지이", "월풀", "다이슨", "위닉스", "쿠쿠", "코웨이",
    "보쉬", "밀레", "일렉트로룩스", "필립스", "아이로봇", "발뮤다",
}

STOPWORDS = KOREAN_STOPWORDS | ENGLISH_STOPWORDS | GENERIC_EXCLUDE | UI_NOISE_TERMS | BRAND_NOISE_TERMS

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

# Suffix-based noise filter: tokens ending with these are UI/navigation noise
_NOISE_SUFFIXES = ("보기", "클릭", "링크", "바로가기")

# Suffix-based noise filter: common Korean verb/adjective endings that are not business keywords
_NOISE_VERB_SUFFIXES = (
    "했는데", "했어요", "합니다", "하는데", "있어요", "없어요", "됩니다",
    "인데요", "거든요", "잖아요", "나요", "ㅂ니다", "습니다",
    "해서요", "인가요", "건가요", "인데", "네요", "더라고",
    "더라구", "겠네요", "겠다", "었는데", "겠어요", "는데", "어서", "다고", "했고", "했다고", "지만",
)

# Korean particle/postposition suffixes — strip to get the root noun
_KR_PARTICLE_SUFFIXES = (
    "님", "을", "를", "이", "가", "에서", "에게", "으로", "에는",
    "에도", "에서도", "이나", "이라", "이고", "이랑", "처럼", "만큼",
    "까지", "부터", "보다", "한테", "에서의", "으로도", "이면",
)


def _strip_kr_particles(token: str) -> str:
    """Strip Korean particles/postpositions to get the root noun."""
    for suffix in _KR_PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def normalize_keyword(token: str) -> str:
    token = str(token or "").strip().lower()
    if not token:
        return ""
    token = token.strip(".,!?()[]{}\"'`\u201c\u201d\u2018:;/-_")
    token = token.replace("\u2019", "")
    token = KEYWORD_TRANSLATIONS.get(token, token)
    if token in STOPWORDS or token.isdigit():
        return ""
    # Suffix-based filtering (from Codex): catches compound words ending in noise suffixes
    if token.endswith(_NOISE_SUFFIXES):
        return ""
    if token.endswith(_NOISE_VERB_SUFFIXES) and token not in BUSINESS_KEEPWORDS:
        return ""
    # Strip Korean particles (e.g. 잇섭님→잇섭, 자가를→자가)
    stripped = _strip_kr_particles(token)
    if stripped != token:
        # Re-check the stripped form against stopwords
        if stripped in STOPWORDS or stripped in BRAND_NOISE_TERMS:
            return ""
        # If the stripped form is a BUSINESS_KEEPWORD, use it
        if stripped in BUSINESS_KEEPWORDS:
            return stripped
        token = stripped
    if len(token) < 2:
        return ""
    if re.search(r"[^a-z0-9\uac00-\ud7a3+/]", token):
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
    filtered = [(keyword, count) for keyword, count in counter.items() if keyword in BUSINESS_KEEPWORDS or bool(re.search(r"[\uac00-\ud7a3]", keyword))]
    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered[:top_n]



def extract_keywords(texts: pd.Series, top_n: int = 30) -> pd.DataFrame:
    counter = build_keyword_counter(texts.fillna(""))
    rows = [{"keyword": token, "count": count} for token, count in filter_business_keywords(counter, top_n)]
    return pd.DataFrame(rows)
