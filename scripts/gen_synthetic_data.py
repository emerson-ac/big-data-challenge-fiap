"""Generate a small synthetic Instacart-format dataset to validate the pipeline.

This does NOT replace the real (Kaggle) data. It only exercises the
preprocess -> train -> evaluate flow end-to-end in seconds.

Usage:
    uv run python scripts/gen_synthetic_data.py [--out DIR] [--users N] [--products N]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def _products(n_products: int) -> pd.DataFrame:
    """Creates the products table with names and decreasing popularity."""
    return pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "product_name": [f"product_{i}" for i in range(1, n_products + 1)],
            "aisle_id": np.random.randint(1, 20, n_products),
            "department_id": np.random.randint(1, 10, n_products),
        }
    )


def _sample_products(n: int, n_products: int, weights: np.ndarray) -> np.ndarray:
    """Samples n product_ids weighted by popularity (long tail)."""
    size = min(n, n_products)
    return np.random.choice(n_products, size=size, replace=False, p=weights) + 1


def _user_orders(user_id: int, n_products: int, weights, order_id: int) -> tuple:
    """Generates prior orders for one user; returns rows and next order_id."""
    orders, prior_rows = [], []
    n_orders = np.random.randint(1, 16)  # 1..15 -> covers all 3 segments
    for _ in range(n_orders):
        orders.append((order_id, user_id, "prior"))
        for pid in _sample_products(np.random.randint(3, 12), n_products, weights):
            prior_rows.append((order_id, pid))
        order_id += 1
    return orders, prior_rows, order_id


def _eval_order(user_id: int, n_products: int, weights, order_id: int) -> tuple:
    """Generates the labeled (train) order for an evaluable user."""
    orders, train_rows = [], []
    orders.append((order_id, user_id, "train"))
    for pid in _sample_products(np.random.randint(2, 8), n_products, weights):
        train_rows.append((order_id, pid))
    return orders, train_rows, order_id + 1


def generate(out: Path, n_users: int, n_products: int) -> None:
    """Generates and writes the synthetic CSVs to ``out``."""
    np.random.seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    weights = 1.0 / np.arange(1, n_products + 1)
    weights /= weights.sum()
    orders, prior_rows, train_rows = [], [], []
    order_id = 1
    for user_id in range(1, n_users + 1):
        u_orders, u_prior, order_id = _user_orders(
            user_id, n_products, weights, order_id
        )
        orders += u_orders
        prior_rows += u_prior
        if user_id <= int(n_users * 0.85):  # 85% become evaluable (train) users
            e_orders, e_train, order_id = _eval_order(
                user_id, n_products, weights, order_id
            )
            orders += e_orders
            train_rows += e_train
    _write_csvs(out, orders, prior_rows, train_rows, n_users, n_products, order_id)


def _write_csvs(
    out, orders, prior_rows, train_rows, n_users, n_products, order_id
) -> None:
    """Writes the four synthetic CSV files to the output directory."""
    pd.DataFrame(orders, columns=["order_id", "user_id", "eval_set"]).to_csv(
        out / "orders.csv", index=False
    )
    pd.DataFrame(prior_rows, columns=["order_id", "product_id"]).to_csv(
        out / "order_products__prior.csv", index=False
    )
    pd.DataFrame(train_rows, columns=["order_id", "product_id"]).to_csv(
        out / "order_products__train.csv", index=False
    )
    _products(n_products).to_csv(out / "products.csv", index=False)
    print(
        f"[OK] Synthetic dataset in {out}: {n_users} users, "
        f"{n_products} products, {order_id - 1} orders."
    )


def main() -> None:
    """Parses arguments and generates the synthetic dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--products", type=int, default=200)
    args = parser.parse_args()
    generate(args.out, args.users, args.products)


if __name__ == "__main__":
    main()
