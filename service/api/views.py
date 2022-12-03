import os
import sys
from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from recmodels.reco import RecModel
from service.api.exceptions import (
    AuthError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.log import app_logger
from service.settings import get_config

sys.path.append(os.path.join(os.path.dirname("./recmodels"), "recmodels"))


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()

api_key = HTTPBearer(auto_error=False)


def get_rmodel(
    model_path: str,
    dataset_path: str = get_config().dataset_path,
) -> RecModel:
    try:
        rmodel = RecModel(model_path, dataset_path)
    except FileNotFoundError:
        raise ModelNotFoundError(error_message="Model load error")
    return rmodel


rmodels = {
    model_name: get_rmodel(model_path)
    for model_name, model_path in get_config().models.items()
}


# rmodel = get_rmodel()
# recos = rmodel.predict_all()


async def get_api_key(
    token: HTTPAuthorizationCredentials = Security(api_key),
):
    api_tok = get_config().access_token
    if token is not None and token.credentials == api_tok.get_secret_value():
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
        current_model = rmodels[model_name]
    except KeyError:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    # current_model.k = request.app.state.config.k_recs
    try:
        reco = current_model.predict(user_id)
        # reco = recos.loc[user_id]["item_id"].tolist()
    except KeyError:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
