"""Configurações da API de recomendação, externalizadas via .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Configurações da API via variáveis de ambiente.

    Attributes:
        app_name: Nome da aplicação exibido no OpenAPI.
        app_version: Versão semântica da API.
        debug: Habilita modo debug/reload do uvicorn.
        host: Endereço de bind do servidor.
        port: Porta de bind do servidor.
        recommender_type: Modelo registrado na ModelFactory usado em produção.
        similarity_path: Caminho do artefato de similaridade item-item.
        interactions_path: Caminho do histórico de compras (sparse).
        popularity_path: Caminho do ranking de popularidade (fallback cold-start).
        vocab_path: Caminho dos vocabulários de usuário/produto.
        default_top_k: Quantidade padrão de recomendações por requisição.
        max_top_k: Quantidade máxima de recomendações permitida por requisição.
        log_level: Nível de log do structlog.
        random_seed: Seed fixada no startup da aplicação.
    """

    app_name: str = "Recommendation API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    recommender_type: str = "item_based_cf"
    similarity_path: Path = Path("models/item_based_cf/item_similarity.npz")
    interactions_path: Path = Path("data/processed/interactions_prior.npz")
    popularity_path: Path = Path("models/baseline_popularity/ranking.pkl")
    vocab_path: Path = Path("data/processed/vocabularies.pkl")

    default_top_k: int = 10
    max_top_k: int = 100

    log_level: str = "INFO"
    random_seed: int = 42

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = APISettings()
