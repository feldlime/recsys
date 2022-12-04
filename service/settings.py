from typing import Dict

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
    models: Dict = {"userknn": r"./data/models/userknn_tined-3.joblib"}
    dataset_path: str = r"./data/datasets/interactions_processed.csv"


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
