"""Pipeline stage 1: preprocessing (split, vocabularies, sparse matrix).

Ports ``notebooks/02_preprocessing.ipynb`` into a DVC-reproducible script.
Reads ``data/raw/*.csv`` and writes ``data/processed/*``.

Usage:
    uv run python -m src.pipeline.preprocess
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import structlog
from sklearn.model_selection import StratifiedKFold

from src.config import get_settings
from src.pipeline.common import (
    compute_dataset_hash,
    load_config,
    set_seed,
    setup_mlflow,
)

logger = structlog.get_logger()

_RAW_FILES = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
]


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, ...]:
    """Loads the raw Instacart CSVs with compact dtypes.

    Args:
        raw_dir: Directory holding the raw CSVs.

    Returns:
        Tuple (orders, order_products_prior, order_products_train, products).
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
    """Selects the catalog of the top-N most purchased products.

    Args:
        prior: Historical interactions (eval_set=prior).
        top_n: Restricted catalog size.

    Returns:
        Tuple (product_id_to_idx, idx_to_product_id, volume coverage).
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
    """Builds the label basket (eval_set=train) restricted to the catalog.

    Args:
        orders: Orders table with eval_set.
        train: Labeled-order items (eval_set=train).
        product_id_to_idx: Map product_id -> catalog index.

    Returns:
        Tuple (future_baskets per user_id, eval_users array).
    """
    order_to_user = orders.loc[orders["eval_set"] == "train"].set_index("order_id")[
        "user_id"
    ]
    labeled = train[train["product_id"].isin(product_id_to_idx)].copy()
    labeled["user_id"] = labeled["order_id"].map(order_to_user)
    future_baskets = labeled.groupby("user_id")["product_id"].apply(set)
    return future_baskets, future_baskets.index.to_numpy()


def segment_users(orders_per_user: pd.Series) -> pd.Series:
    """Segments users by total order count (to stratify the split).

    Args:
        orders_per_user: Series with the order count per user.

    Returns:
        Categorical series occasional/regular/super_user.
    """
    bins = [0, 3, 10, orders_per_user.max()]
    labels = ["ocasional", "regular", "super_user"]
    return pd.cut(orders_per_user, bins=bins, labels=labels, include_lowest=True)


def _folds_for_ratio(test_ratio: float) -> int:
    """Calculates the number of StratifiedKFold splits from the test ratio."""
    return max(3, int(round(1.0 / test_ratio)))


def split_users(
    eval_users: np.ndarray, segments: pd.Series, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits users into train/val/test via StratifiedKFold (stratified CV).

    Uses StratifiedKFold to ensure each split preserves the segment
    distribution. Folds are assigned proportionally: last fold = test,
    second-to-last = val, rest = train.

    Args:
        eval_users: Evaluable users (with a label).
        segments: Segment of each user_id.
        cfg: The ``preprocessing`` config block.
        seed: Random seed.

    Returns:
        Tuple (train_users, val_users, test_users).
    """
    n_splits = _folds_for_ratio(cfg["test_ratio"])
    seg = segments.loc[eval_users].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        eval_users[test_idx]
        for _, test_idx in skf.split(np.zeros(len(eval_users)), seg)
    ]
    test_users = folds[-1]
    val_users = folds[-2]
    train_users = np.concatenate(folds[:-2])
    return train_users, val_users, test_users


def build_interaction_matrix(
    interactions: pd.DataFrame, user_map: dict, item_map: dict, shape: tuple[int, int]
) -> sp.csr_matrix:
    """Builds the binary sparse user-item matrix (CSR, int8).

    Args:
        interactions: DataFrame with user_id and product_id columns.
        user_map: Map user_id -> row index.
        item_map: Map product_id -> column index.
        shape: Matrix dimensions (n_users, n_items).

    Returns:
        Binary CSR sparse matrix.
    """
    rows = interactions["user_id"].map(user_map).to_numpy()
    cols = interactions["product_id"].map(item_map).to_numpy()
    data = np.ones(len(rows), dtype=np.int8)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    return (matrix > 0).astype(np.int8)


def build_pairs(
    users: np.ndarray, baskets: pd.Series, user_map: dict, item_map: dict
) -> pd.DataFrame:
    """Builds the positive (user_idx, item_idx) pairs of a split.

    Args:
        users: user_ids belonging to the split.
        baskets: Future basket (set of product_ids) per user_id.
        user_map: Map user_id -> index.
        item_map: Map product_id -> index.

    Returns:
        DataFrame with user_idx and item_idx columns.
    """
    pairs = [(user_map[uid], item_map[pid]) for uid in users for pid in baskets[uid]]
    return pd.DataFrame(pairs, columns=["user_idx", "item_idx"])


def _filter_prior(
    orders: pd.DataFrame, prior: pd.DataFrame, item_map: dict, user_map: dict
) -> pd.DataFrame:
    """Filters prior interactions to the catalog and evaluable users."""
    order_to_user = orders.loc[orders["eval_set"] == "prior"].set_index("order_id")[
        "user_id"
    ]
    filtered = prior[prior["product_id"].isin(item_map)].copy()
    filtered["user_id"] = filtered["order_id"].map(order_to_user)
    return filtered[filtered["user_id"].isin(user_map)]


def _save_artifacts(out_dir: Path, vocab: dict, matrix, splits: dict) -> None:
    """Persists vocabularies, interaction matrix and per-split pairs."""
    with open(out_dir / "vocabularies.pkl", "wb") as f:
        pickle.dump(vocab, f)
    sp.save_npz(out_dir / "interactions_prior.npz", matrix)
    for name, pairs in splits.items():
        pairs.to_pickle(out_dir / f"{name}_pairs.pkl")


def _build_meta(pre, settings, shape, coverage, dataset_hash) -> dict:
    """Assembles the split_meta.json dictionary with split metadata."""
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


def _build_vocab(products, sd) -> dict:
    """Assembles the vocabulary dictionary persisted to vocabularies.pkl."""
    idx = sd["idx_to_product_id"]
    return {
        "user_id_to_idx": sd["user_map"],
        "idx_to_user_id": sd["idx_to_user_id"],
        "product_id_to_idx": sd["item_map"],
        "idx_to_product_id": idx,
        "idx_to_product_name": products.set_index("product_id")
        .loc[idx, "product_name"]
        .to_numpy(),
    }


def _build_split_data(orders, prior, train, pre, seed) -> dict:
    """Builds the catalog, split, maps and interaction matrix."""
    item_map, idx_p, cov = build_catalog(prior, pre["top_n_products"])
    baskets, eval_users = build_future_baskets(orders, train, item_map)
    segments = segment_users(orders.groupby("user_id", observed=True).size())
    tu, vu, wu = split_users(eval_users, segments, pre, seed)
    idx_u = np.sort(eval_users).astype(np.int32)
    umap = {uid: idx for idx, uid in enumerate(idx_u)}
    matrix = _build_matrix(orders, prior, item_map, umap, idx_u, idx_p)
    return _split_dict(item_map, idx_p, cov, baskets, tu, vu, wu, idx_u, umap, matrix)


def _build_matrix(orders, prior, item_map, umap, idx_u, idx_p) -> sp.csr_matrix:
    """Builds the interaction matrix from filtered prior interactions."""
    shape = (len(idx_u), len(idx_p))
    return build_interaction_matrix(
        _filter_prior(orders, prior, item_map, umap), umap, item_map, shape
    )


def _split_dict(item_map, idx_p, cov, baskets, tu, vu, wu, idx_u, umap, m):
    """Assembles the split-data dictionary returned by _build_split_data."""
    return {
        "item_map": item_map,
        "idx_to_product_id": idx_p,
        "coverage": cov,
        "baskets": baskets,
        "train_users": tu,
        "val_users": vu,
        "test_users": wu,
        "idx_to_user_id": idx_u,
        "user_map": umap,
        "shape": (len(idx_u), len(idx_p)),
        "matrix": m,
    }


def _compute_meta(settings, pre, sd) -> dict:
    """Computes the dataset hash and assembles split_meta."""
    raw_files = [settings.raw_data_dir / n for n in _RAW_FILES]
    return _build_meta(
        pre, settings, sd["shape"], sd["coverage"], compute_dataset_hash(raw_files)
    )


def _write_meta(meta, out_dir) -> None:
    """Writes split_meta.json."""
    with open(out_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info("preprocessing_done", n_items_catalog=meta["n_items_catalog"])


def _log_run(meta, settings) -> None:
    """Logs preprocessing params, metrics and split_meta to MLflow."""
    import mlflow

    with mlflow.start_run(run_name="preprocessing_v1"):
        mlflow.log_param("top_n_products", meta["top_n_products"])
        mlflow.log_param("random_seed", meta["random_seed"])
        mlflow.log_param("dataset_hash", meta["dataset_hash"])
        mlflow.log_metric("n_items_catalog", meta["n_items_catalog"])
        mlflow.log_metric("coverage", meta["catalog_coverage"])
        mlflow.log_artifact(str(settings.processed_data_dir / "split_meta.json"))


def main() -> None:
    """Runs the full preprocessing and writes split metadata."""
    settings = get_settings()
    config = load_config()
    pre = config["preprocessing"]
    set_seed(settings.random_seed)
    setup_mlflow("preprocessing")
    out_dir = settings.processed_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    orders, prior, train, products = load_raw(settings.raw_data_dir)
    sd = _build_split_data(orders, prior, train, pre, settings.random_seed)
    vocab = _build_vocab(products, sd)
    splits = _build_splits(sd)
    _save_artifacts(out_dir, vocab, sd["matrix"], splits)
    meta = _compute_meta(settings, pre, sd)
    _write_meta(meta, out_dir)
    _log_run(meta, settings)


def _build_splits(sd) -> dict:
    """Builds the train/val/test positive-pair DataFrames."""
    baskets, umap, imap = sd["baskets"], sd["user_map"], sd["item_map"]
    return {
        "train": build_pairs(sd["train_users"], baskets, umap, imap),
        "val": build_pairs(sd["val_users"], baskets, umap, imap),
        "test": build_pairs(sd["test_users"], baskets, umap, imap),
    }


if __name__ == "__main__":
    main()
