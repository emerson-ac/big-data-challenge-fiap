"""Neural Collaborative Filtering training (port of notebook 07, main model)."""

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
    """Builds one epoch's dataset: positives plus uniform negatives.

    Args:
        pairs: Positive (user_idx, item_idx) training pairs.
        n_items: Catalog size.
        negative_ratio: Negatives sampled per positive.
        rng: numpy generator (seeded per epoch).

    Returns:
        TensorDataset with columns (user_idx, item_idx, label).
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
    """Runs one training epoch and returns the mean loss."""
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
    """Evaluates the model on a sample via score_all_items."""
    return evaluate_score_fn(
        users, lambda u: score_all_items(model, u, n_items), ground_truth, k
    )


def _run_epoch(model, spec, ctx, optimizer, loss_fn, epoch) -> dict:
    """Runs a single epoch: train, evaluate, return the history entry."""
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
    return {"epoch": epoch, "train_loss": train_loss, **val}


def run_training(spec: dict, ctx: dict) -> tuple:
    """Trains an NCF from scratch with early stopping on validation recall.

    Args:
        spec: Hyperparameters (embedding_dim, hidden_dims, learning_rate,
            epochs, patience).
        ctx: Training context (train_pairs, n_users, n_items, val_sample, gt,
            negative_ratio, batch_size, k).

    Returns:
        Tuple (model with best weights, per-epoch history, best recall).
    """
    model = NeuralCollaborativeFiltering(
        ctx["n_users"], ctx["n_items"], spec["embedding_dim"], spec["hidden_dims"]
    )
    optimizer = optim.Adam(model.parameters(), lr=spec["learning_rate"])
    loss_fn = nn.BCELoss()
    history, best_recall, best_state, patience = [], -1.0, None, 0
    for epoch in range(1, spec["epochs"] + 1):
        entry = _run_epoch(model, spec, ctx, optimizer, loss_fn, epoch)
        history.append(entry)
        best_state, best_recall, patience = _update_best(
            entry, best_state, model, best_recall, patience
        )
        if patience >= spec["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, best_recall


def _update_best(entry, best_state, model, best_recall, patience) -> tuple:
    """Tracks the best validation recall and early-stopping patience."""
    if entry["recall_at_k"] > best_recall:
        return copy.deepcopy(model.state_dict()), entry["recall_at_k"], 0
    return best_state, best_recall, patience + 1


def _combos(cfg: dict) -> list[tuple]:
    """Enumerates all (lr, dim, hidden_dims) hyperparameter combinations."""
    return [
        (lr, dim, tuple(hd))
        for lr in cfg["learning_rate_choices"]
        for dim in cfg["embedding_dim_choices"]
        for hd in cfg["hidden_dims_choices"]
    ]


def _search_spec(cfg: dict, lr, dim, hd) -> dict:
    """Builds the spec for a short random-search training trial."""
    return {
        "embedding_dim": dim,
        "hidden_dims": hd,
        "learning_rate": lr,
        "epochs": cfg["search_epochs"],
        "patience": cfg["search_early_stopping_patience"],
    }


def _search_result(lr, dim, hd, history, recall) -> dict:
    """Assembles the result row of a random-search trial."""
    return {
        "learning_rate": lr,
        "embedding_dim": dim,
        "hidden_dims": list(hd),
        "best_val_recall_at_k": recall,
        "epochs_run": len(history),
    }


def _search(ctx: dict, cfg: dict) -> list[dict]:
    """Random search of NCF hyperparameters (few epochs per trial)."""
    rng = random.Random(RANDOM_SEED)
    combos = _combos(cfg)
    results = []
    for lr, dim, hd in rng.sample(combos, k=min(cfg["n_trials"], len(combos))):
        _, history, recall = run_training(_search_spec(cfg, lr, dim, hd), ctx)
        results.append(_search_result(lr, dim, hd, history, recall))
    return results


def _build_ctx(
    data: ProcessedData, processed_dir: Path, ncf_cfg: dict, k: int, seed: int
) -> dict:
    """Builds the training context (data + validation sample)."""
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


def _persist(model, history, out_dir, best, ncf_cfg) -> dict:
    """Saves model weights and history; returns the params sub-dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_pickle(out_dir / "training_history.pkl", history)
    return {
        "embedding_dim": best["embedding_dim"],
        "hidden_dims": list(best["hidden_dims"]),
        "learning_rate": best["learning_rate"],
        "negative_ratio": ncf_cfg["negative_ratio"],
        "batch_size": ncf_cfg["batch_size"],
    }


def _final_spec(best: dict, ncf_cfg: dict) -> dict:
    """Builds the spec for the final NCF training (full epochs)."""
    return {
        "embedding_dim": best["embedding_dim"],
        "hidden_dims": tuple(best["hidden_dims"]),
        "learning_rate": best["learning_rate"],
        "epochs": ncf_cfg["epochs"],
        "patience": ncf_cfg["early_stopping_patience"],
    }


def _ncf_payload(k, params, search, val, test) -> dict:
    """Assembles the NCF metrics payload."""
    return {
        "k": k,
        "params": params,
        "search_results": search,
        "validation": val,
        "test": test,
    }


def train(
    data: ProcessedData,
    k: int,
    ncf_cfg: dict,
    out_dir: Path,
    seed: int,
    processed_dir: Path,
) -> dict:
    """Trains the NCF (random search + final training) and persists artifacts.

    Args:
        data: Processed artifacts.
        k: Top-k size.
        ncf_cfg: The ``ncf`` config block.
        out_dir: Output directory (models/neural_network).
        seed: Random seed.
        processed_dir: The ``data/processed`` directory (for train_pairs).

    Returns:
        Metrics payload with params, search_results, validation, test.
    """
    ctx = _build_ctx(data, processed_dir, ncf_cfg, k, seed)
    search = _search(ctx, ncf_cfg["search"])
    best = max(search, key=lambda r: r["best_val_recall_at_k"])
    model, history, _ = run_training(_final_spec(best, ncf_cfg), ctx)
    n_items = ctx["n_items"]
    val = _eval_sample(
        model, list(data.val_ground_truth), data.val_ground_truth, n_items, k
    )
    test = _eval_sample(
        model, list(data.test_ground_truth), data.test_ground_truth, n_items, k
    )
    params = _persist(model, history, out_dir, best, ncf_cfg)
    payload = _ncf_payload(k, params, search, val, test)
    save_metrics(out_dir, payload)
    return payload
