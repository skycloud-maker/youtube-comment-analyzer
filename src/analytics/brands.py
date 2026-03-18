"""Brand mention detection for appliance comments."""

from __future__ import annotations

BRAND_PATTERNS = {
    "삼성": ["삼성", "samsung", "비스포크", "bespoke"],
    "LG": ["lg", "엘지", "트롬", "오브제"],
    "월풀": ["whirlpool"],
    "GE": ["ge", "ge profile"],
    "보쉬": ["bosch"],
    "키친에이드": ["kitchenaid"],
    "메이태그": ["maytag"],
}



def infer_brand(text: str) -> str:
    lowered = (text or "").lower()
    for brand, keywords in BRAND_PATTERNS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return brand
    return "미언급"
