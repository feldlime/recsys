from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from recmodels import rmodels
from service.api.exceptions import (
    AuthError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.log import app_logger
from service.settings import ACCESS_TOKEN


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()

api_key = HTTPBearer(auto_error=False)


async def get_api_key(
    token: HTTPAuthorizationCredentials = Security(api_key),
):
    if token is not None and token.credentials == ACCESS_TOKEN:
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
        current_model = rmodels.to_prod[model_name]
    except KeyError:
        raise ModelNotFoundError(error_message=f'Model {model_name} not found')

    current_model.k = request.app.state.k_recs
    reco = current_model.predict(user_id)

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
