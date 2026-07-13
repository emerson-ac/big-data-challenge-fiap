"""Estágio 1 do pipeline: pré-processamento (split, vocabulários, matriz esparsa).

Porta a lógica de ``notebooks/02_preprocessing.ipynb`` para um script reproduzível
via DVC. Lê ``data/raw/*.csv`` e escreve ``data/processed/*``.

Uso:
    uv run python -m src.pipeline.preprocess
"""

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
import structlog
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.pipeline.common import (
    compute_dataset_hash,
    load_config,
    set_seed,
    setup_mlflow,
)

logger = structlog.get_logger()


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, ...]:
    """Carrega os CSVs brutos do Instacart com dtypes enxutos.

    Args:
        raw_dir: Diretório contendo os CSVs brutos.

    Returns:
        Tupla (orders, order_products_prior, order_products_train, products).
    """
    orders = pd.read_csv(
        raw_dir / "orders.csv",
        dtype={"order_id": "int32", "user_id": "int32", "eval_set": "category"},
        usecols=["order_id", "user_id", "eval_set"],
    )
    prior = pd.read_csv(
        raw_dir / "order_products__prior.csv",
        dtype={"order_id": "int32", "product_id": "int32"},
        usecols=["order_id", "product_id"],
    )
    train = pd.read_csv(
        raw_dir / "order_products__train.csv",
        dtype={"order_id": "int32", "product_id": "int32"},
        usecols=["order_id", "product_id"],
    )
    products = pd.read_csv(raw_dir / "products.csv", dtype={"product_id": "int32"})
    return orders, prior, train, products


def build_catalog(prior: pd.DataFrame, top_n: int) -> tuple[dict, np.ndarray, float]:
    """Seleciona o catálogo dos top-N produtos mais comprados.

    Args:
        prior: Interações históricas (eval_set=prior).
        top_n: Tamanho do catálogo restrito.

    Returns:
        Tupla (product_id_to_idx, idx_to_product_id, cobertura de volume).
    """
    counts = prior["product_id"].value_counts()
    top_products = counts.head(top_n).index
    product_id_to_idx = {pid: idx for idx, pid in enumerate(top_products)}
    idx_to_product_id = np.array(top_products, dtype=np.int32)
    coverage = counts.head(top_n).sum() / counts.sum()
    return product_id_to_idx, idx_to_product_id, float(coverage)


def build_future_baskets(
    orders: pd.DataFrame, train: pd.DataFrame, product_id_to_idx: dict
) -> tuple[pd.Series, np.ndarray]:
    """Constrói o cesto-rótulo (eval_set=train) restrito ao catálogo.

    Args:
        orders: Tabela de pedidos com eval_set.
        train: Itens dos pedidos rotulados (eval_set=train).
        product_id_to_idx: Mapa product_id -> índice do catálogo.

    Returns:
        Tupla (future_baskets por user_id, array de eval_users).
    """
    order_to_user = orders.loc[orders["eval_set"] == "train"].set_index("order_id")[
        "user_id"
    ]
    labeled = train[train["product_id"].isin(product_id_to_idx)].copy()
    labeled["user_id"] = labeled["order_id"].map(order_to_user)
    future_baskets = labeled.groupby("user_id")["product_id"].apply(set)
    return future_baskets, future_baskets.index.to_numpy()


def segment_users(orders_per_user: pd.Series) -> pd.Series:
    """Segmenta usuários por número total de pedidos (para estratificar o split).

    Args:
        orders_per_user: Série com a contagem de pedidos por usuário.

    Returns:
        Série categórica ocasional/regular/super_user.
    """
    bins = [0, 3, 10, orders_per_user.max()]
    labels = ["ocasional", "regular", "super_user"]
    return pd.cut(orders_per_user, bins=bins, labels=labels, include_lowest=True)


def split_users(
    eval_users: np.ndarray, segments: pd.Series, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide usuários em treino/val/teste, estratificado por segmento.

    Args:
        eval_users: Usuários avaliáveis (com rótulo).
        segments: Segmento de cada user_id.
        cfg: Bloco ``preprocessing`` da configuração.
        seed: Semente de aleatoriedade.

    Returns:
        Tupla (train_users, val_users, test_users).
    """
    train_users, rest = train_test_split(
        eval_users,
        train_size=cfg["train_ratio"],
        random_state=seed,
        stratify=segments.loc[eval_users],
    )
    val_frac = cfg["val_ratio"] / (cfg["val_ratio"] + cfg["test_ratio"])
    val_users, test_users = train_test_split(
        rest, train_size=val_frac, random_state=seed, stratify=segments.loc[rest]
    )
    return train_users, val_users, test_users


def build_interaction_matrix(
    interactions: pd.DataFrame, user_map: dict, item_map: dict, shape: tuple[int, int]
) -> sp.csr_matrix:
    """Constrói a matriz esparsa binária usuário-item (CSR, int8).

    Args:
        interactions: DataFrame com colunas user_id e product_id.
        user_map: Mapa user_id -> índice de linha.
        item_map: Mapa product_id -> índice de coluna.
        shape: Dimensões (n_users, n_items) da matriz.

    Returns:
        Matriz esparsa CSR binária.
    """
    rows = interactions["user_id"].map(user_map).to_numpy()
    cols = interactions["product_id"].map(item_map).to_numpy()
    data = np.ones(len(rows), dtype=np.int8)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    return (matrix > 0).astype(np.int8)


def build_pairs(
    users: np.ndarray, baskets: pd.Series, user_map: dict, item_map: dict
) -> pd.DataFrame:
    """Constrói os pares positivos (user_idx, item_idx) de um split.

    Args:
        users: user_ids pertencentes ao split.
        baskets: Cesto futuro (set de product_ids) por user_id.
        user_map: Mapa user_id -> índice.
        item_map: Mapa product_id -> índice.

    Returns:
        DataFrame com colunas user_idx e item_idx.
    """
    pairs = [(user_map[uid], item_map[pid]) for uid in users for pid in baskets[uid]]
    return pd.DataFrame(pairs, columns=["user_idx", "item_idx"])


def _filter_prior(
    orders: pd.DataFrame, prior: pd.DataFrame, item_map: dict, user_map: dict
) -> pd.DataFrame:
    """Filtra interações prior ao catálogo e aos usuários avaliáveis."""
    order_to_user = orders.loc[orders["eval_set"] == "prior"].set_index("order_id")[
        "user_id"
    ]
    filtered = prior[prior["product_id"].isin(item_map)].copy()
    filtered["user_id"] = filtered["order_id"].map(order_to_user)
    return filtered[filtered["user_id"].isin(user_map)]


def _save_artifacts(out_dir: Path, vocab: dict, matrix, splits: dict) -> None:
    """Persiste vocabulários, matriz de interações e pares por split."""
    with open(out_dir / "vocabularies.pkl", "wb") as f:
        pickle.dump(vocab, f)
    sp.save_npz(out_dir / "interactions_prior.npz", matrix)
    for name, pairs in splits.items():
        pairs.to_pickle(out_dir / f"{name}_pairs.pkl")


def main() -> None:
    """Executa o pré-processamento completo e rastreia no MLflow."""
    settings = get_settings()
    config = load_config()
    pre = config["preprocessing"]
    set_seed(settings.random_seed)
    setup_mlflow("preprocessing")
    out_dir = settings.data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    orders, prior, train, products = load_raw(settings.data_raw_dir)
    item_map, idx_to_product_id, coverage = build_catalog(prior, pre["top_n_products"])
    baskets, eval_users = build_future_baskets(orders, train, item_map)

    segments = segment_users(orders.groupby("user_id", observed=True).size())
    train_users, val_users, test_users = split_users(
        eval_users, segments, pre, settings.random_seed
    )

    idx_to_user_id = np.sort(eval_users).astype(np.int32)
    user_map = {uid: idx for idx, uid in enumerate(idx_to_user_id)}
    shape = (len(idx_to_user_id), len(idx_to_product_id))
    matrix = build_interaction_matrix(
        _filter_prior(orders, prior, item_map, user_map), user_map, item_map, shape
    )

    vocab = {
        "user_id_to_idx": user_map,
        "idx_to_user_id": idx_to_user_id,
        "product_id_to_idx": item_map,
        "idx_to_product_id": idx_to_product_id,
        "idx_to_product_name": products.set_index("product_id")
        .loc[idx_to_product_id, "product_name"]
        .to_numpy(),
    }
    splits = {
        "train": build_pairs(train_users, baskets, user_map, item_map),
        "val": build_pairs(val_users, baskets, user_map, item_map),
        "test": build_pairs(test_users, baskets, user_map, item_map),
    }
    _save_artifacts(out_dir, vocab, matrix, splits)

    raw_files = [settings.data_raw_dir / n for n in _RAW_FILES]
    meta = _build_meta(pre, settings, shape, coverage, compute_dataset_hash(raw_files))
    with open(out_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log_run(meta)
    logger.info("preprocessing_done", **{k: meta[k] for k in ("n_items_catalog",)})


_RAW_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
]


def _build_meta(pre, settings, shape, coverage, dataset_hash) -> dict:
    """Monta o dicionário split_meta.json com metadados do split."""
    return {
        "random_seed": settings.random_seed,
        "top_n_products": pre["top_n_products"],
        "train_ratio": pre["train_ratio"],
        "val_ratio": pre["val_ratio"],
        "test_ratio": pre["test_ratio"],
        "n_items_catalog": shape[1],
        "catalog_coverage": coverage,
        "dataset_hash": dataset_hash,
    }


def _log_run(meta: dict) -> None:
    """Rastreia parâmetros, métricas e o split_meta no MLflow."""
    out_dir = get_settings().data_processed_dir
    with mlflow.start_run(run_name="preprocessing_v1"):
        mlflow.log_params({k: meta[k] for k in ("top_n_products", "random_seed")})
        mlflow.log_param("dataset_hash", meta["dataset_hash"])
        mlflow.log_metrics(
            {
                "n_items_catalog": meta["n_items_catalog"],
                "coverage": meta["catalog_coverage"],
            }
        )
        mlflow.log_artifact(str(out_dir / "split_meta.json"))


if __name__ == "__main__":
    main()
