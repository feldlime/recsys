from os import getenv as env

from pydantic import BaseSettings

ACCESS_TOKEN: str = env("ACCESS_TOKEN")


class Config(BaseSettings):

    class Config:
        case_sensitive = False


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {
                "env": ["log_level"]
            },
        }


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10
    log_config: LogConfig


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
