from __future__ import annotations

import torch
from torch import nn


class LogClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        outputs, _ = self.encoder(embedded)
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)
