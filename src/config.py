"""Application configuration."""

from __future__ import annotations

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


class AnalyticsSettings(BaseModel):
    top_n_keywords: int = 30
    max_representative_comments: int = 5
    topic_count: int = 6
    risk_keywords: list[str] = Field(default_factory=lambda: ["배송", "불량", "가격", "AS", "소음", "발열", "품질"])


class ExcelSettings(BaseModel):
    datetime_format: str = "yyyy-mm-dd hh:mm:ss"
    freeze_panes: str = "A2"


class YamlSettings(BaseModel):
    project_name: str = "youtube-comment-analysis"
    log_level: str = "INFO"
    default_language: str = "ko"
    default_region: str = "KR"
    daily_quota_limit: int = 10000
    api: ApiSettings = Field(default_factory=ApiSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    analytics: AnalyticsSettings = Field(default_factory=AnalyticsSettings)
    excel: ExcelSettings = Field(default_factory=ExcelSettings)


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



def load_config() -> AppConfig:
    load_dotenv()
    try:
        env = EnvSettings()
    except ValidationError as exc:
        raise RuntimeError("Configuration error: YOUTUBE_API_KEY must be provided via environment variables.") from exc

    yaml_settings = YamlSettings.model_validate(_load_yaml(env.yt_analysis_settings_path))
    return AppConfig(env=env, yaml=yaml_settings)
