"""Treino do Neural Collaborative Filtering (porta do notebook 07, modelo principal)."""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.ncf import NeuralCollaborativeFiltering, score_all_items
from src.models.training.data import (
    ProcessedData,
    evaluate_score_fn,
    save_metrics,
    save_pickle,
)

RANDOM_SEED = 42


def build_epoch_dataset(
    pairs: pd.DataFrame, n_items: int, negative_ratio: int, rng
) -> TensorDataset:
    """Monta o dataset de uma época: positivos + negativos uniformes.

    Args:
        pairs: Pares positivos (user_idx, item_idx) de treino.
        n_items: Tamanho do catálogo.
        negative_ratio: Negativos amostrados por positivo.
        rng: Gerador numpy (seed fixa por época).

    Returns:
        TensorDataset com colunas (user_idx, item_idx, label).
    """
    pos_users, pos_items = pairs["user_idx"].to_numpy(), pairs["item_idx"].to_numpy()
    neg_users = np.repeat(pos_users, negative_ratio)
    neg_items = rng.integers(0, n_items, size=len(neg_users))
    users = np.concatenate([pos_users, neg_users])
    items = np.concatenate([pos_items, neg_items])
    labels = np.concatenate([np.ones(len(pos_users)), np.zeros(len(neg_users))])
    return TensorDataset(
        torch.tensor(users, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def _train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    """Executa uma época de treino e devolve a loss média."""
    model.train()
    total = 0.0
    for users, items, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(users, items), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(users)
    return total / len(loader.dataset)


def _eval_sample(model, users, ground_truth, n_items, k) -> dict:
    """Avalia o modelo numa amostra via score_all_items."""
    return evaluate_score_fn(
        users, lambda u: score_all_items(model, u, n_items), ground_truth, k
    )


def run_training(spec: dict, ctx: dict) -> tuple:
    """Treina um NCF do zero com early stopping por recall na amostra de validação.

    Args:
        spec: Hiperparâmetros (embedding_dim, hidden_dims, learning_rate,
            epochs, patience).
        ctx: Contexto de treino (train_pairs, n_users, n_items, val_sample,
            gt, negative_ratio, batch_size, k).

    Returns:
        Tupla (modelo com melhores pesos, histórico por época, melhor recall).
    """
    model = NeuralCollaborativeFiltering(
        ctx["n_users"], ctx["n_items"], spec["embedding_dim"], spec["hidden_dims"]
    )
    optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])
    loss_fn = nn.BCELoss()
    history, best_recall, best_state, patience = [], -1.0, None, 0
    for epoch in range(1, spec["epochs"] + 1):
        dataset = build_epoch_dataset(
            ctx["train_pairs"],
            ctx["n_items"],
            ctx["negative_ratio"],
            np.random.default_rng(RANDOM_SEED + epoch),
        )
        loader = DataLoader(dataset, batch_size=ctx["batch_size"], shuffle=True)
        train_loss = _train_one_epoch(model, loader, optimizer, loss_fn)
        val = _eval_sample(
            model, ctx["val_sample"], ctx["val_sample_gt"], ctx["n_items"], ctx["k"]
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        if val["recall_at_k"] > best_recall:
            best_recall, best_state, patience = (
                val["recall_at_k"],
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            patience += 1
            if patience >= spec["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history, best_recall


def _search(ctx: dict, cfg: dict) -> list[dict]:
    """Random search de hiperparâmetros do NCF (poucas épocas por trial)."""
    rng = random.Random(RANDOM_SEED)
    combos = [
        (lr, dim, tuple(hd))
        for lr in cfg["learning_rate_choices"]
        for dim in cfg["embedding_dim_choices"]
        for hd in cfg["hidden_dims_choices"]
    ]
    results = []
    for lr, dim, hd in rng.sample(combos, k=min(cfg["n_trials"], len(combos))):
        spec = {
            "embedding_dim": dim,
            "hidden_dims": hd,
            "learning_rate": lr,
            "epochs": cfg["search_epochs"],
            "patience": cfg["search_early_stopping_patience"],
        }
        _, history, recall = run_training(spec, ctx)
        results.append(
            {
                "learning_rate": lr,
                "embedding_dim": dim,
                "hidden_dims": list(hd),
                "best_val_recall_at_k": recall,
                "epochs_run": len(history),
            }
        )
    return results


def _build_ctx(
    data: ProcessedData, processed_dir: Path, ncf_cfg: dict, k: int, seed: int
) -> dict:
    """Monta o contexto de treino (dados + amostra de validação)."""
    train_pairs = pd.read_pickle(processed_dir / "train_pairs.pkl")
    n_users, n_items = data.interactions.shape
    rng = np.random.default_rng(seed)
    val_all = list(data.val_ground_truth.keys())
    sample = rng.choice(
        val_all, size=min(ncf_cfg["val_sample_size"], len(val_all)), replace=False
    ).tolist()
    return {
        "train_pairs": train_pairs,
        "n_users": n_users,
        "n_items": n_items,
        "val_sample": sample,
        "val_sample_gt": {u: data.val_ground_truth[u] for u in sample},
        "negative_ratio": ncf_cfg["negative_ratio"],
        "batch_size": ncf_cfg["batch_size"],
        "k": k,
    }


def train(
    data: ProcessedData,
    k: int,
    ncf_cfg: dict,
    out_dir: Path,
    seed: int,
    processed_dir: Path,
) -> dict:
    """Treina o NCF (random search + treino final) e persiste os artefatos.

    Args:
        data: Artefatos processados.
        k: Tamanho do top-k.
        ncf_cfg: Bloco ``ncf`` da configuração.
        out_dir: Diretório de saída (models/neural_network).
        seed: Semente de aleatoriedade.
        processed_dir: Diretório ``data/processed`` (para carregar train_pairs).

    Returns:
        Payload de métricas com params, search_results, validation, test.
    """
    ctx = _build_ctx(data, processed_dir, ncf_cfg, k, seed)
    search = _search(ctx, ncf_cfg["search"])
    best = max(search, key=lambda r: r["best_val_recall_at_k"])
    spec = {
        "embedding_dim": best["embedding_dim"],
        "hidden_dims": tuple(best["hidden_dims"]),
        "learning_rate": best["learning_rate"],
        "epochs": ncf_cfg["epochs"],
        "patience": ncf_cfg["early_stopping_patience"],
    }
    model, history, _ = run_training(spec, ctx)

    n_items = ctx["n_items"]
    val = _eval_sample(
        model, list(data.val_ground_truth), data.val_ground_truth, n_items, k
    )
    test = _eval_sample(
        model, list(data.test_ground_truth), data.test_ground_truth, n_items, k
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_pickle(out_dir / "training_history.pkl", history)
    params = {
        "embedding_dim": best["embedding_dim"],
        "hidden_dims": list(best["hidden_dims"]),
        "learning_rate": best["learning_rate"],
        "negative_ratio": ncf_cfg["negative_ratio"],
        "batch_size": ncf_cfg["batch_size"],
    }
    payload = {
        "k": k,
        "params": params,
        "search_results": search,
        "validation": val,
        "test": test,
    }
    save_metrics(out_dir, payload)
    return payload
