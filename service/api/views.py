import os
import sys
from typing import Dict, List

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from recmodels.reco import RecModel
from recmodels.rmodels import load_model
from service.api.exceptions import (
    AuthError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.log import app_logger
from service.settings import ServiceConfig, get_config

sys.path.append(os.path.join(os.path.dirname("./recmodels"), "recmodels"))
sys.path.append(os.path.join(os.path.dirname("./data/models"), "models"))


def load_reco_models(
    service_config: ServiceConfig = get_config(),
) -> Dict[str, RecModel]:
    rec_models: Dict[str, RecModel] = {}
    for model_name in service_config.rec_models:
        try:
            rec_models[model_name] = load_model(model_name)
        except ValueError:
            app_logger.error(f"Model {model_name} not found")
            raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    return rec_models


reco_models = load_reco_models()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()

api_key = HTTPBearer(auto_error=False)


async def get_api_key(
    token: HTTPAuthorizationCredentials = Security(api_key),
) -> str:
    if (
        token is not None
        and token.credentials == get_config().access_token.get_secret_value()
    ):
        return token.credentials
    raise AuthError(error_message="Authorization error")


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        403: {"description": "Authorization error"},
        404: {"description": "Not found"},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_token: str = Depends(get_api_key),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    try:
        current_model = reco_models[model_name]
    except KeyError:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k = request.query_params.get("k", 10)
    try:
        reco = current_model.predict(user_id, k)
    except Exception as e:
        app_logger.error(e)
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
