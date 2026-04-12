"""Preset batch runner for appliance VoC collection."""

from __future__ import annotations

from types import SimpleNamespace

from src.cli import run_pipeline

BATCH_PRESETS = {
    "세탁기": {
        "KR": [
            {"keyword": "세탁기 추천", "language": "ko", "region": "KR", "max_videos": 25},
            {"keyword": "세탁기 고장", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "세탁기 비교 리뷰", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "세탁기 후기", "language": "ko", "region": "KR", "max_videos": 15},
        ],
        "US": [
            {"keyword": "washer review", "language": "en", "region": "US", "max_videos": 25},
            {"keyword": "washer problems", "language": "en", "region": "US", "max_videos": 15},
            {"keyword": "best washing machine", "language": "en", "region": "US", "max_videos": 15},
        ],
    },
    "냉장고": {
        "KR": [
            {"keyword": "냉장고 추천", "language": "ko", "region": "KR", "max_videos": 25},
            {"keyword": "냉장고 고장", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "냉장고 비교 리뷰", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "냉장고 후기", "language": "ko", "region": "KR", "max_videos": 15},
        ],
        "US": [
            {"keyword": "refrigerator review", "language": "en", "region": "US", "max_videos": 25},
            {"keyword": "refrigerator problems", "language": "en", "region": "US", "max_videos": 15},
            {"keyword": "best refrigerator", "language": "en", "region": "US", "max_videos": 15},
        ],
    },
    "건조기": {
        "KR": [
            {"keyword": "건조기 추천", "language": "ko", "region": "KR", "max_videos": 25},
            {"keyword": "건조기 고장", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "건조기 비교 리뷰", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "건조기 후기", "language": "ko", "region": "KR", "max_videos": 15},
        ],
        "US": [
            {"keyword": "dryer review", "language": "en", "region": "US", "max_videos": 25},
            {"keyword": "dryer problems", "language": "en", "region": "US", "max_videos": 15},
            {"keyword": "best dryer", "language": "en", "region": "US", "max_videos": 15},
        ],
    },
    "식기세척기": {
        "KR": [
            {"keyword": "식기세척기 추천", "language": "ko", "region": "KR", "max_videos": 25},
            {"keyword": "식기세척기 고장", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "식기세척기 비교 리뷰", "language": "ko", "region": "KR", "max_videos": 15},
            {"keyword": "식기세척기 후기", "language": "ko", "region": "KR", "max_videos": 15},
        ],
        "US": [
            {"keyword": "dishwasher review", "language": "en", "region": "US", "max_videos": 25},
            {"keyword": "dishwasher problems", "language": "en", "region": "US", "max_videos": 15},
            {"keyword": "best dishwasher", "language": "en", "region": "US", "max_videos": 15},
        ],
    },
}



def build_args(*, product: str, keyword: str, language: str, region: str, max_videos: int, suffix: str) -> SimpleNamespace:
    slug = f"{region.lower()}_{product}_{suffix}"
    return SimpleNamespace(
        command="run",
        keyword=keyword,
        keywords=[],
        product=product,
        max_videos=max_videos,
        comments_per_video=0,
        comments_order="time",
        published_after=None,
        published_before=None,
        order="relevance",
        language=language,
        region=region,
        channel_id=None,
        include_replies="true",
        refresh_existing="false",
        search_oversample_factor=None,
        output_prefix=f"voc_{slug}",
        run_id=None,
    )



def main() -> None:
    results = []
    failures = []
    for product, countries in BATCH_PRESETS.items():
        for country_code, presets in countries.items():
            for index, preset in enumerate(presets, start=1):
                args = build_args(product=product, suffix=str(index), **preset)
                try:
                    result = run_pipeline(args)
                    result["product"] = product
                    result["country"] = country_code
                    result["keyword"] = preset["keyword"]
                    results.append(result)
                    print(f"[완료] {product} / {country_code} / {preset['keyword']} / run_id={result['run_id']}")
                except Exception as exc:
                    failures.append({
                        "product": product,
                        "country": country_code,
                        "keyword": preset["keyword"],
                        "error": str(exc),
                    })
                    print(f"[실패] {product} / {country_code} / {preset['keyword']} / error={exc}")
    print({"results": results, "failures": failures})


if __name__ == "__main__":
    main()
