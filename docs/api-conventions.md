# Convenções de API - Sistema de Recomendação

Documento que padroniza a estrutura, nomeação, validação e tratamento de erros para a API REST de recomendação de produtos.

---

## 1. Estrutura de Diretórios

```
src/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── main.py                 # Aplicação FastAPI principal
│   ├── config.py               # Configurações (Settings com Pydantic)
│   ├── dependencies.py         # Injeção de dependências
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── logging.py          # Middleware de logging estruturado
│   │   ├── error_handler.py    # Middleware de tratamento de erros
│   │   └── rate_limit.py       # Middleware de rate limiting (opcional)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py           # Health check
│   │   ├── recommendations.py  # Endpoints de recomendação
│   │   └── models.py           # Endpoints de gerenciamento de modelos
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py          # Schemas de requisição (Pydantic)
│   │   ├── response.py         # Schemas de resposta
│   │   └── errors.py           # Schemas de erro
│   ├── services/
│   │   ├── __init__.py
│   │   ├── recommendation_service.py    # Lógica de recomendação
│   │   ├── model_service.py             # Gerenciamento de modelos
│   │   └── preprocessing_service.py     # Pré-processamento de dados
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Configuração de structlog
│       └── cache.py            # Utilitários de cache
└── models/
    ├── inference.py            # Carregamento e inference de modelos
    └── model_loader.py         # Factory para carregar modelos
```

---

## 2. Configuração da Aplicação

### 2.1 Settings com Pydantic

**Arquivo:** `src/api/config.py`

```python
from pydantic_settings import BaseSettings
from typing import Optional

class APISettings(BaseSettings):
    """Configurações da API via variáveis de ambiente."""
    
    app_name: str = "Recommendation API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Modelos
    model_path: str = "models/neural_network/model.pt"
    device: str = "cpu"  # ou "cuda"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommendation_api"
    
    # Logging
    log_level: str = "INFO"
    structlog_format: str = "json"
    
    # Cache
    cache_ttl_seconds: int = 3600
    enable_cache: bool = True
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # Random Seed
    random_seed: int = 42
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = APISettings()
```

### 2.2 Variáveis de Ambiente

**Arquivo:** `.env.example`

```env
APP_NAME=Recommendation API
APP_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000

MODEL_PATH=models/neural_network/model.pt
DEVICE=cpu

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=recommendation_api

LOG_LEVEL=INFO
STRUCTLOG_FORMAT=json

CACHE_TTL_SECONDS=3600
ENABLE_CACHE=true

RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

RANDOM_SEED=42
```

---

## 3. Logging Estruturado

### 3.1 Configuração de Structlog

**Arquivo:** `src/api/utils/logger.py`

```python
import structlog
from src.api.config import settings

def configure_logging():
    """Configura structlog para logging estruturado em toda a API."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
            if settings.structlog_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger():
    """Obtém logger estruturado."""
    return structlog.get_logger()
```

### 3.2 Uso de Logging

**Regra Obrigatória:** Proibido usar `print()` no código da API. Sempre usar structlog.

```python
from src.api.utils.logger import get_logger

logger = get_logger()

# Correto
logger.info("user_recommendation_requested", user_id=user_id, top_k=top_k)

# Proibido
# print(f"Recomendação solicitada para usuário {user_id}")
```

---

## 4. Schemas de Requisição e Resposta

### 4.1 Schemas de Requisição

**Arquivo:** `src/api/schemas/request.py`

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class RecommendationRequest(BaseModel):
    """Schema de requisição para recomendações.
    
    Attributes:
        user_id: ID do usuário (inteiro positivo)
        top_k: Número de recomendações desejadas (padrão: 5)
        exclude_items: Lista de IDs de itens a excluir (opcional)
        min_score_threshold: Score mínimo para recomendação (opcional)
    """
    user_id: int = Field(..., gt=0, description="ID do usuário")
    top_k: int = Field(default=5, ge=1, le=100, description="Quantidade de recomendações")
    exclude_items: Optional[List[int]] = Field(None, description="Itens a excluir")
    min_score_threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "top_k": 10,
                "exclude_items": [456, 789],
                "min_score_threshold": 0.3
            }
        }
    
    @validator("top_k")
    def validate_top_k(cls, v):
        """Valida que top_k não é muito grande."""
        if v > 1000:
            raise ValueError("top_k não pode exceder 1000")
        return v
```

### 4.2 Schemas de Resposta

**Arquivo:** `src/api/schemas/response.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class RecommendationItem(BaseModel):
    """Item individual na recomendação.
    
    Attributes:
        product_id: ID do produto
        score: Score de recomendação (0-1)
        rank: Posição na ranking
    """
    product_id: int = Field(..., description="ID do produto")
    score: float = Field(..., ge=0.0, le=1.0, description="Score de recomendação")
    rank: int = Field(..., ge=1, description="Posição na ranking")

class RecommendationResponse(BaseModel):
    """Response de recomendações.
    
    Attributes:
        user_id: ID do usuário
        recommendations: Lista de recomendações
        model_version: Versão do modelo utilizado
        timestamp: Timestamp da requisição
        processing_time_ms: Tempo de processamento em ms
    """
    user_id: int
    recommendations: List[RecommendationItem]
    model_version: str
    timestamp: datetime
    processing_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "recommendations": [
                    {"product_id": 456, "score": 0.95, "rank": 1},
                    {"product_id": 789, "score": 0.87, "rank": 2}
                ],
                "model_version": "v1.0.0",
                "timestamp": "2026-06-08T10:30:00Z",
                "processing_time_ms": 45.5
            }
        }

class ErrorResponse(BaseModel):
    """Schema de erro padrão.
    
    Attributes:
        error_code: Código de erro interno
        message: Mensagem descritiva
        details: Detalhes adicionais (opcional)
        timestamp: Quando o erro ocorreu
    """
    error_code: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime
```

---

## 5. Tratamento de Erros

### 5.1 Exceções Customizadas

**Arquivo:** `src/api/schemas/errors.py`

```python
class APIException(Exception):
    """Exceção base da API."""
    
    def __init__(self, error_code: str, message: str, details: dict = None):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class UserNotFoundError(APIException):
    """Erro quando usuário não é encontrado."""
    def __init__(self, user_id: int):
        super().__init__(
            error_code="USER_NOT_FOUND",
            message=f"Usuário {user_id} não encontrado",
            details={"user_id": user_id}
        )

class ModelNotLoadedError(APIException):
    """Erro quando modelo não pode ser carregado."""
    def __init__(self, reason: str):
        super().__init__(
            error_code="MODEL_NOT_LOADED",
            message="Modelo não disponível",
            details={"reason": reason}
        )

class ValidationError(APIException):
    """Erro de validação de entrada."""
    def __init__(self, field: str, reason: str):
        super().__init__(
            error_code="VALIDATION_ERROR",
            message=f"Erro de validação no campo '{field}'",
            details={"field": field, "reason": reason}
        )
```

### 5.2 Exception Handler Middleware

**Arquivo:** `src/api/middleware/error_handler.py`

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from datetime import datetime
from src.api.schemas.errors import APIException
from src.api.utils.logger import get_logger

logger = get_logger()

async def api_exception_handler(request: Request, exc: APIException):
    """Handler para exceções de API."""
    logger.error(
        "api_exception_occurred",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções genéricas."""
    logger.exception(
        "unexpected_error",
        path=request.url.path,
        exception=str(exc)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Erro interno do servidor",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

---

## 6. Endpoints

### 6.1 Health Check

**Arquivo:** `src/api/routes/health.py`

```python
from fastapi import APIRouter
from src.api.utils.logger import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger()

@router.get("/status")
async def health_check():
    """Verifica se a API está operacional.
    
    Returns:
        dict: Status da API e informações de saúde
    """
    logger.info("health_check_requested")
    return {
        "status": "healthy",
        "service": "recommendation-api",
        "version": "1.0.0"
    }
```

### 6.2 Endpoint de Recomendação

**Arquivo:** `src/api/routes/recommendations.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas.request import RecommendationRequest
from src.api.schemas.response import RecommendationResponse
from src.api.services.recommendation_service import RecommendationService
from src.api.utils.logger import get_logger
from datetime import datetime
import time

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = get_logger()

@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    service: RecommendationService = Depends()
):
    """Gera recomendações para um usuário.
    
    Args:
        request: Schema de requisição com user_id e parâmetros
        service: Serviço de recomendação (injeção de dependência)
    
    Returns:
        RecommendationResponse: Recomendações com score e ranking
    
    Raises:
        HTTPException: Se usuário não encontrado ou erro no modelo
    """
    start_time = time.time()
    
    logger.info(
        "recommendation_request_received",
        user_id=request.user_id,
        top_k=request.top_k
    )
    
    recommendations = await service.get_recommendations(request)
    processing_time_ms = (time.time() - start_time) * 1000
    
    logger.info(
        "recommendations_generated",
        user_id=request.user_id,
        count=len(recommendations),
        processing_time_ms=round(processing_time_ms, 2)
    )
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version="v1.0.0",
        timestamp=datetime.utcnow(),
        processing_time_ms=round(processing_time_ms, 2)
    )
```

---

## 7. Services e Lógica de Negócio

### 7.1 Padrão Factory para Modelos

**Arquivo:** `src/models/model_loader.py`

```python
from typing import Protocol
from abc import ABC, abstractmethod

class ModelFactory:
    """Factory Pattern para carregar diferentes modelos.
    
    Implementa o padrão Factory para abstrair a criação e
    carregamento de diferentes tipos de modelos de ML.
    """
    
    _models = {}
    
    @classmethod
    def register(cls, name: str, model_class):
        """Registra um novo tipo de modelo."""
        cls._models[name] = model_class
    
    @classmethod
    def create(cls, model_type: str, **kwargs):
        """Cria e retorna uma instância do modelo."""
        model_class = cls._models.get(model_type)
        if not model_class:
            raise ValueError(f"Modelo '{model_type}' não registrado")
        return model_class(**kwargs)

# Registro de modelos
class NeuralNetworkModel:
    """Modelo de rede neural em PyTorch."""
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        # Carregamento do modelo

ModelFactory.register("neural_network", NeuralNetworkModel)
```

### 7.2 Service de Recomendação

**Arquivo:** `src/api/services/recommendation_service.py`

```python
from src.api.schemas.request import RecommendationRequest
from src.api.schemas.response import RecommendationItem
from src.api.schemas.errors import UserNotFoundError, ModelNotLoadedError
from src.models.model_loader import ModelFactory
from src.api.utils.logger import get_logger

logger = get_logger()

class RecommendationService:
    """Service para gerar recomendações.
    
    Implementa a lógica de negócio para recomendações,
    incluindo carregamento de modelos e processamento de requisições.
    """
    
    def __init__(self):
        self.model = self._load_model()
    
    def _load_model(self):
        """Carrega o modelo via Factory."""
        try:
            return ModelFactory.create("neural_network")
        except Exception as e:
            logger.error("model_loading_failed", reason=str(e))
            raise ModelNotLoadedError(str(e))
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> list[RecommendationItem]:
        """Gera recomendações para um usuário.
        
        Args:
            request: Requisição com parâmetros de recomendação
        
        Returns:
            Lista de itens recomendados com scores
        
        Raises:
            UserNotFoundError: Se usuário não existe nos dados
            ModelNotLoadedError: Se modelo não está disponível
        """
        # Validação de usuário
        if not self._user_exists(request.user_id):
            raise UserNotFoundError(request.user_id)
        
        # Geração de recomendações
        scores = self.model.predict(request.user_id)
        
        # Filtragem e ranking
        recommendations = self._process_recommendations(
            scores,
            request.top_k,
            request.exclude_items,
            request.min_score_threshold
        )
        
        return recommendations
    
    def _user_exists(self, user_id: int) -> bool:
        """Verifica se usuário existe nos dados."""
        # Implementação específica do projeto
        return True
    
    def _process_recommendations(
        self,
        scores: dict,
        top_k: int,
        exclude_items: list = None,
        min_score: float = 0.0
    ) -> list[RecommendationItem]:
        """Processa scores em recomendações ranqueadas.
        
        Args:
            scores: Dicionário {item_id: score}
            top_k: Quantidade de recomendações
            exclude_items: IDs a excluir
            min_score: Score mínimo
        
        Returns:
            Lista de RecommendationItem ordenada por score
        """
        exclude_set = set(exclude_items or [])
        
        filtered = [
            (item_id, score)
            for item_id, score in scores.items()
            if item_id not in exclude_set and score >= min_score
        ]
        
        ranked = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            RecommendationItem(
                product_id=item_id,
                score=float(score),
                rank=idx + 1
            )
            for idx, (item_id, score) in enumerate(ranked)
        ]
```

---

## 8. Injeção de Dependências

**Arquivo:** `src/api/dependencies.py`

```python
from fastapi import Depends
from src.api.services.recommendation_service import RecommendationService

_recommendation_service: RecommendationService = None

def get_recommendation_service() -> RecommendationService:
    """Injeção de dependência do serviço de recomendação."""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service
```

---

## 9. Aplicação Principal

**Arquivo:** `src/api/main.py`

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.config import settings
from src.api.utils.logger import get_logger, configure_logging
from src.api.routes import health, recommendations
from src.api.middleware.error_handler import (
    api_exception_handler,
    general_exception_handler
)
from src.api.schemas.errors import APIException
import numpy as np
import torch

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    # Startup
    logger.info("application_startup", version=settings.app_version)
    np.random.seed(settings.random_seed)
    torch.manual_seed(settings.random_seed)
    
    yield
    
    # Shutdown
    logger.info("application_shutdown")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Incluir routers
app.include_router(health.router)
app.include_router(recommendations.router)

if __name__ == "__main__":
    import uvicorn
    configure_logging()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
```

---

## 10. Type Hints e Docstrings

### 10.1 Obrigatoriedades

- [ ] **Type hints:** Obrigatório em TODAS as funções públicas
- [ ] **Return types:** Sempre especificar tipo de retorno
- [ ] **Docstrings:** Padrão Google Style em funções públicas
- [ ] **Máximo 20 linhas:** Por função (refatorar em sub-funções se necessário)

### 10.2 Exemplo

```python
def process_user_data(
    user_id: int,
    items: list[int]
) -> dict[str, float]:
    """Processa dados de usuário para recomendação.
    
    Realiza validação, normalização e extração de features
    do usuário para alimentar o modelo de recomendação.
    
    Args:
        user_id: ID único do usuário (positivo)
        items: Lista de IDs de itens comprados
    
    Returns:
        Dicionário com features normalizadas do usuário
    
    Raises:
        ValueError: Se user_id é inválido
        KeyError: Se items contém IDs não conhecidos
    """
    # Máximo 20 linhas
    return {}
```

---

## 11. Testes

### 11.1 Estrutura

```
tests/
├── __init__.py
├── test_api.py
├── test_services.py
├── test_schemas.py
└── conftest.py              # Fixtures do pytest
```

### 11.2 Exemplo de Teste

**Arquivo:** `tests/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

@pytest.fixture
def valid_request():
    """Fixture com requisição válida."""
    return {"user_id": 123, "top_k": 5}

def test_health_check():
    """Testa endpoint de health check."""
    response = client.get("/health/status")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_recommendation_valid_request(valid_request):
    """Testa geração de recomendação com requisição válida."""
    response = client.post("/recommendations/", json=valid_request)
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_recommendation_invalid_user():
    """Testa erro para usuário inexistente."""
    response = client.post(
        "/recommendations/",
        json={"user_id": -1, "top_k": 5}
    )
    assert response.status_code != 200
```

---

## 12. Versionamento de API

### 12.1 Estrutura com Versionamento

```
src/api/
├── v1/
│   ├── __init__.py
│   ├── routes/
│   │   ├── recommendations.py
│   └── schemas/
│       └── response.py
└── v2/
    ├── __init__.py
    └── routes/
        └── recommendations.py
```

### 12.2 Inclusão de Versões

```python
from src.api.v1.routes import recommendations as v1_recommendations
from src.api.v2.routes import recommendations as v2_recommendations

app.include_router(v1_recommendations.router, prefix="/api/v1")
app.include_router(v2_recommendations.router, prefix="/api/v2")
```

---

## 13. Checklist de Qualidade

- [ ] Nenhum `print()` no código; usar structlog
- [ ] RANDOM_SEED = 42 fixado no startup
- [ ] Type hints em todas as funções públicas
- [ ] Docstrings em padrão Google Style
- [ ] Funções com máximo 20 linhas
- [ ] Design Pattern implementado (Factory, Strategy, etc)
- [ ] Exceções customizadas para erros de negócio
- [ ] Logging estruturado em pontos críticos
- [ ] Validação com Pydantic em todas as entradas
- [ ] Tratamento de erros com middleware
- [ ] Testes unitários para services
- [ ] Testes de integração para endpoints
- [ ] Ruff linting sem erros
- [ ] Pre-commit hooks configurados
- [ ] Commits em padrão semântico

---

## 14. Referências e Integração

- Consultar [REQUIREMENTS.md](REQUIREMENTS.md) para requisitos gerais
- Consultar [naming-conventions.md](naming-conventions.md) para convenções de código
- Consultar [design-pattern.md](design-pattern.md) para padrões de projeto
- Consultar [NOTEBOOKS.md](NOTEBOOKS.md) para pipeline de modelagem
