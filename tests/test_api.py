"""Testes de integração da API (rotas, validação e tratamento de erros)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_recommendation_engine
from src.api.main import app
from src.api.schemas.errors import ModelNotLoadedError
from src.models.inference import RecommendationEngine


@pytest.fixture
def client(engine: RecommendationEngine) -> Iterator[TestClient]:
    """TestClient com o RecommendationEngine real substituído pelo sintético."""
    app.dependency_overrides[get_recommendation_engine] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check_returns_healthy(client: TestClient) -> None:
    """GET /health/status deve responder 200 com status healthy."""
    response = client.get("/health/status")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_recommendations_for_known_user(client: TestClient) -> None:
    """POST /recommendations/ para usuário conhecido retorna recomendações do modelo."""
    response = client.post("/recommendations/", json={"user_id": 10, "top_k": 2})

    body = response.json()
    assert response.status_code == 200
    assert body["is_cold_start"] is False
    assert [item["product_id"] for item in body["recommendations"]] == [100, 200]


def test_recommendations_for_unknown_user_is_cold_start(client: TestClient) -> None:
    """Usuário desconhecido em POST /recommendations/ usa o fallback de popularidade."""
    response = client.post("/recommendations/", json={"user_id": 999, "top_k": 2})

    body = response.json()
    assert response.status_code == 200
    assert body["is_cold_start"] is True
    assert body["model_type"] == "popularity"


def test_recommendations_applies_exclude_items(client: TestClient) -> None:
    """exclude_items informado na requisição não deve aparecer no resultado."""
    response = client.post(
        "/recommendations/",
        json={"user_id": 10, "top_k": 2, "exclude_items": [100]},
    )

    body = response.json()
    product_ids = [item["product_id"] for item in body["recommendations"]]
    assert 100 not in product_ids


def test_recommendations_invalid_top_k_returns_422(client: TestClient) -> None:
    """top_k fora do intervalo permitido deve falhar na validação do Pydantic."""
    response = client.post("/recommendations/", json={"user_id": 10, "top_k": 0})

    assert response.status_code == 422


def test_recommendations_missing_user_id_returns_422(client: TestClient) -> None:
    """Requisição sem user_id deve falhar na validação do Pydantic."""
    response = client.post("/recommendations/", json={"top_k": 5})

    assert response.status_code == 422


def test_model_not_loaded_error_returns_503(client: TestClient) -> None:
    """ModelNotLoadedError deve ser convertida em uma resposta 503 padronizada."""

    def _raise_model_not_loaded() -> RecommendationEngine:
        raise ModelNotLoadedError("artefato ausente")

    app.dependency_overrides[get_recommendation_engine] = _raise_model_not_loaded

    response = client.post("/recommendations/", json={"user_id": 10, "top_k": 2})

    body = response.json()
    assert response.status_code == 503
    assert body["error_code"] == "MODEL_NOT_LOADED"
