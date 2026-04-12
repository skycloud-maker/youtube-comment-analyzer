"""Preset batch runner for appliance VoC collection."""

from __future__ import annotations

from types import SimpleNamespace

from src.cli import run_pipeline

BATCH_PRESETS = {
    "세탁기": {
        "KR": [
            {"keyword": "세탁기 추천", "language": "ko", "region": "KR", "max_videos": 10},
            {"keyword": "세탁기 고장", "language": "ko", "region": "KR", "max_videos": 10},
        ],
        "US": [
            {"keyword": "washer review", "language": "en", "region": "US", "max_videos": 10},
            {"keyword": "washer problems", "language": "en", "region": "US", "max_videos": 10},
        ],
    },
    "냉장고": {
        "KR": [
            {"keyword": "냉장고 추천", "language": "ko", "region": "KR", "max_videos": 10},
            {"keyword": "냉장고 고장", "language": "ko", "region": "KR", "max_videos": 10},
        ],
        "US": [
            {"keyword": "refrigerator review", "language": "en", "region": "US", "max_videos": 10},
            {"keyword": "refrigerator problems", "language": "en", "region": "US", "max_videos": 10},
        ],
    },
    "건조기": {
        "KR": [
            {"keyword": "건조기 추천", "language": "ko", "region": "KR", "max_videos": 10},
            {"keyword": "건조기 고장", "language": "ko", "region": "KR", "max_videos": 10},
        ],
        "US": [
            {"keyword": "dryer review", "language": "en", "region": "US", "max_videos": 10},
            {"keyword": "dryer problems", "language": "en", "region": "US", "max_videos": 10},
        ],
    },
    "식기세척기": {
        "KR": [
            {"keyword": "식기세척기 추천", "language": "ko", "region": "KR", "max_videos": 10},
            {"keyword": "식기세척기 고장", "language": "ko", "region": "KR", "max_videos": 10},
        ],
        "US": [
            {"keyword": "dishwasher review", "language": "en", "region": "US", "max_videos": 10},
            {"keyword": "dishwasher problems", "language": "en", "region": "US", "max_videos": 10},
        ],
    },
}



def build_args(*, product: str, keyword: str, language: str, region: str, max_videos: int, suffix: str) -> SimpleNamespace:
    slug = f"{region.lower()}_{product}_{suffix}"
    return SimpleNamespace(
        command="run",
        keyword=keyword,
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
    for product, countries in BATCH_PRESETS.items():
        for country_code, presets in countries.items():
            for index, preset in enumerate(presets, start=1):
                args = build_args(product=product, suffix=str(index), **preset)
                result = run_pipeline(args)
                result["product"] = product
                result["country"] = country_code
                results.append(result)
                print(f"[완료] {product} / {country_code} / {preset['keyword']} / run_id={result['run_id']}")
    print(results)


if __name__ == "__main__":
    main()
