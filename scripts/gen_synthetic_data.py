"""Gera um mini-dataset sintético no formato Instacart para validar o pipeline.

NÃO substitui os dados reais (Kaggle). Serve apenas para exercitar
preprocess -> train -> evaluate ponta a ponta em segundos.

Uso:
    uv run python scripts/gen_synthetic_data.py [--out DIR] [--users N] [--products N]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def _products(n_products: int) -> pd.DataFrame:
    """Cria a tabela de produtos com nomes e popularidade decrescente."""
    return pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "product_name": [f"produto_{i}" for i in range(1, n_products + 1)],
            "aisle_id": np.random.randint(1, 20, n_products),
            "department_id": np.random.randint(1, 10, n_products),
        }
    )


def _sample_products(n: int, n_products: int, weights: np.ndarray) -> np.ndarray:
    """Amostra n product_ids ponderados por popularidade (com cauda longa)."""
    size = min(n, n_products)
    return np.random.choice(n_products, size=size, replace=False, p=weights) + 1


def generate(out: Path, n_users: int, n_products: int) -> None:
    """Gera e escreve os CSVs sintéticos em ``out``."""
    np.random.seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    weights = 1.0 / np.arange(1, n_products + 1)
    weights /= weights.sum()

    orders, prior_rows, train_rows = [], [], []
    order_id = 1
    for user_id in range(1, n_users + 1):
        n_orders = np.random.randint(1, 16)  # 1..15 -> cobre os 3 segmentos
        for _ in range(n_orders):
            orders.append((order_id, user_id, "prior"))
            for pid in _sample_products(np.random.randint(3, 12), n_products, weights):
                prior_rows.append((order_id, pid))
            order_id += 1
        if user_id <= int(n_users * 0.85):  # 85% viram usuários avaliáveis (train)
            orders.append((order_id, user_id, "train"))
            for pid in _sample_products(np.random.randint(2, 8), n_products, weights):
                train_rows.append((order_id, pid))
            order_id += 1

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
        f"[OK] Dataset sintético em {out}: {n_users} usuários, "
        f"{n_products} produtos, {order_id - 1} pedidos."
    )


def main() -> None:
    """Parseia argumentos e gera o dataset sintético."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--products", type=int, default=200)
    args = parser.parse_args()
    generate(args.out, args.users, args.products)


if __name__ == "__main__":
    main()
