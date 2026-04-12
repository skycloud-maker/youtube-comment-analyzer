"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseModel):
    base_url: str = "https://www.googleapis.com/youtube/v3"
    timeout_seconds: int = 30
    max_retries: int = 5
    backoff_seconds: float = 1.5
    max_results_per_page: int = 50


class StorageSettings(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    exports_dir: Path = Path("data/exports")
    logs_dir: Path = Path("data/logs")


class VideoSelectionSettings(BaseModel):
    enabled: bool = True
    search_oversample_factor: int = 3
    require_product_term: bool = True
    require_target_brand: bool = False
    exclude_competitor_brands: bool = True
    target_brand_terms: list[str] = Field(
        default_factory=lambda: ["lg", "엘지", "오브제", "디오스", "트롬", "휘센", "lg전자"]
    )
    competitor_brand_terms: list[str] = Field(
        default_factory=lambda: [
            "samsung",
            "삼성",
            "bespoke",
            "whirlpool",
            "ge",
            "electrolux",
            "bosch",
            "haier",
            "miele",
            "frigidaire",
        ]
    )
    product_terms: list[str] = Field(
        default_factory=lambda: [
            "가전",
            "냉장고",
            "세탁기",
            "건조기",
            "식기세척기",
            "청소기",
            "오븐",
            "정수기",
            "인덕션",
            "washer",
            "dryer",
            "refrigerator",
            "fridge",
            "dishwasher",
            "appliance",
        ]
    )
    preferred_channel_keywords: list[str] = Field(default_factory=list)
    blocked_channel_keywords: list[str] = Field(default_factory=list)


class CollectionSettings(BaseModel):
    video_selection: VideoSelectionSettings = Field(default_factory=VideoSelectionSettings)


class NlpAnalyzerSettings(BaseModel):
    enabled: bool = True
    provider: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    fallback_provider: str = "openai"
    fallback_model: str = "gpt-4o-mini"
    batch_size: int = 50
    use_fallback_on_failure: bool = True


class AnalyticsSettings(BaseModel):
    top_n_keywords: int = 30
    max_representative_comments: int = 5
    topic_count: int = 6
    risk_keywords: list[str] = Field(default_factory=lambda: ["배송", "불량", "가격", "AS", "소음", "발열", "품질"])
    nlp_analyzer: NlpAnalyzerSettings = Field(default_factory=NlpAnalyzerSettings)


class ExcelSettings(BaseModel):
    datetime_format: str = "yyyy-mm-dd hh:mm:ss"
    freeze_panes: str = "A2"


class DashboardSettings(BaseModel):
    supported_regions: list[str] = Field(default_factory=lambda: ["KR", "US"])
    primary_product_filters: list[str] = Field(
        default_factory=lambda: ["냉장고", "세탁기", "건조기", "식기세척기", "청소기", "오븐"]
    )
    extra_product_filters: list[str] = Field(
        default_factory=lambda: ["정수기", "인덕션", "와인셀러", "컵세척기", "실내식물재배기(틔운)"]
    )
    brand_filter_options: list[str] = Field(default_factory=lambda: ["LG", "Samsung", "GE", "Whirlpool"])
    build_label: str = "stable"
    cache_schema_version: str = "dashboard-v2"
    lite_comments_limit: int = 1000
    weekly_chart_page_min: int = 4
    weekly_chart_page_max: int = 24
    weekly_chart_page_default: int = 12
    weekly_chart_page_step: int = 2
    mpl_font_candidates: list[Path] = Field(
        default_factory=lambda: [
            Path("fonts/NanumGothic.ttf"),
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("C:/Windows/Fonts/NanumGothic.ttf"),
        ]
    )


class YamlSettings(BaseModel):
    project_name: str = "youtube-comment-analysis"
    log_level: str = "INFO"
    default_language: str = "ko"
    default_region: str = "KR"
    daily_quota_limit: int = 10000
    api: ApiSettings = Field(default_factory=ApiSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    collection: CollectionSettings = Field(default_factory=CollectionSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    excel: ExcelSettings = Field(default_factory=ExcelSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    youtube_api_key: SecretStr
    yt_analysis_settings_path: Path = Path("config/settings.example.yaml")
    yt_analysis_env: str = "local"


class AppConfig(BaseModel):
    env: EnvSettings
    yaml: YamlSettings

    @model_validator(mode="after")
    def ensure_directories(self) -> "AppConfig":
        for path in (
            self.yaml.storage.raw_dir,
            self.yaml.storage.processed_dir,
            self.yaml.storage.exports_dir,
            self.yaml.storage.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Settings file must contain a YAML mapping: {path}")
    return payload


def load_yaml_settings_optional(path: Path | None = None) -> YamlSettings:
    settings_path = path or Path(os.getenv("YT_ANALYSIS_SETTINGS_PATH", "config/settings.example.yaml"))
    if not settings_path.exists():
        return YamlSettings()
    return YamlSettings.model_validate(_load_yaml(settings_path))


def load_config() -> AppConfig:
    load_dotenv()
    try:
        env = EnvSettings()
    except ValidationError as exc:
        raise RuntimeError("Configuration error: YOUTUBE_API_KEY must be provided via environment variables.") from exc

    yaml_settings = load_yaml_settings_optional(env.yt_analysis_settings_path)
    return AppConfig(env=env, yaml=yaml_settings)
