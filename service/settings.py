from typing import List

from pydantic import BaseSettings, SecretStr


class Config(BaseSettings):
    class Config:
        case_sensitive = False


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {"env": ["log_level"]},
        }


class ServiceConfig(Config):
    access_token: SecretStr
    service_name: str = "reco_service"
    k_recs: int = 10
    log_config: LogConfig
    rec_models: List[str] = ["test"]


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
