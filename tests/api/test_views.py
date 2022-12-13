from http import HTTPStatus

from requests.structures import CaseInsensitiveDict
from starlette.testclient import TestClient

from service.settings import ServiceConfig, get_config

GET_RECO_PATH = "/reco/{model_name}/{user_id}"


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig = get_config(),
) -> None:
    user_id: int = 651
    path = GET_RECO_PATH.format(model_name="test", user_id=user_id)
    api_token = service_config.access_token
    client.headers = CaseInsensitiveDict(
        {"Authorization": f"Bearer {api_token.get_secret_value()}"}
    )
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    client: TestClient,
    service_config: ServiceConfig = get_config(),
) -> None:
    user_id = 10**10
    path = GET_RECO_PATH.format(model_name="test", user_id=user_id)
    api_token = service_config.access_token
    client.headers = CaseInsensitiveDict(
        {"Authorization": f"Bearer {api_token.get_secret_value()}"}
    )
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_reco_for_anauth_user(
    client: TestClient,
) -> None:
    user_id = 3444
    path = GET_RECO_PATH.format(model_name="test", user_id=user_id)
    client.headers = CaseInsensitiveDict({"Authorization": "Bearer "})
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json()["errors"][0]["error_key"] == "auth_error"


def test_get_reco_for_unknown_model(
    client: TestClient,
    service_config: ServiceConfig = get_config(),
) -> None:
    user_id = 4566
    path = GET_RECO_PATH.format(model_name="some_model", user_id=user_id)
    api_token = service_config.access_token
    client.headers = CaseInsensitiveDict(
        {"Authorization": f"Bearer {api_token.get_secret_value()}"}
    )
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"
