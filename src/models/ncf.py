"""Neural Collaborative Filtering (NCF) model for implicit feedback."""

import numpy as np
import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """NCF model: user/item embeddings combined through an MLP.

    Args:
        n_users: Number of users in the vocabulary.
        n_items: Number of items in the vocabulary.
        embedding_dim: Dimension of the user and item embeddings.
        hidden_dims: Sizes of the MLP hidden layers.
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
        """Computes the user-item affinity score.

        Args:
            user_ids: Tensor of user indices, shape (batch,).
            item_ids: Tensor of item indices, shape (batch,).

        Returns:
            Tensor of scores in (0, 1), shape (batch,).
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
    """Computes the dense score matrix (users x items) for an NCF model.

    Args:
        model: Trained NCF model.
        user_indices: List of user_idx to score.
        n_items: Size of the item catalog.
        batch_size: Number of users processed per batch.

    Returns:
        Numpy matrix (len(user_indices), n_items) with the scores.
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
