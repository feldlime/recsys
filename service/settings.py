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
    models: Dict = {
        # "userknn": r"./data/models/userknn_tined.joblib",
        # "userknn2": r"./data/models/userknn_1W.joblib",
        # "userknn3": r"./data/models/userknn_1D.joblib",
        "userknn4": r"./data/models/knn_20.joblib",
    }


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
