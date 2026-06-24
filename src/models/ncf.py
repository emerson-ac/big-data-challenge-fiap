"""Modelo Neural Collaborative Filtering (NCF) para feedback implícito."""

import numpy as np
import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """Modelo NCF: embeddings de usuário/item combinados via MLP.

    Args:
        n_users: Número de usuários no vocabulário.
        n_items: Número de itens no vocabulário.
        embedding_dim: Dimensão dos embeddings de usuário e item.
        hidden_dims: Tamanhos das camadas ocultas do MLP.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Calcula o score de afinidade usuário-produto.

        Args:
            user_ids: Tensor de índices de usuário, shape (batch,).
            item_ids: Tensor de índices de item, shape (batch,).

        Returns:
            Tensor de scores em (0, 1), shape (batch,).
        """
        embeddings = [self.user_embedding(user_ids), self.item_embedding(item_ids)]
        x = torch.cat(embeddings, dim=1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


@torch.no_grad()
def score_all_items(
    model: NeuralCollaborativeFiltering,
    user_indices: list[int],
    n_items: int,
    batch_size: int = 1000,
) -> np.ndarray:
    """Calcula a matriz densa de scores (usuarios x itens) de um modelo NCF.

    Args:
        model: Modelo NCF treinado.
        user_indices: Lista de user_idx a pontuar.
        n_items: Tamanho do catalogo de itens.
        batch_size: Numero de usuarios processados por lote.

    Returns:
        Matriz numpy (len(user_indices), n_items) com os scores.
    """
    model.eval()
    item_ids = torch.arange(n_items)
    score_rows = []
    for start in range(0, len(user_indices), batch_size):
        batch = user_indices[start : start + batch_size]
        users_rep = torch.tensor(batch).repeat_interleave(n_items)
        items_rep = item_ids.repeat(len(batch))
        scores = model(users_rep, items_rep).view(len(batch), n_items)
        score_rows.append(scores.numpy())
    return np.concatenate(score_rows, axis=0)
