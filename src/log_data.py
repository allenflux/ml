from __future__ import annotations

import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class LabelEncoder:
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    @classmethod
    def build(cls, labels: list[str]) -> "LabelEncoder":
        unique_labels = sorted(set(labels))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        return cls(label_to_id=label_to_id, id_to_label=id_to_label)

    def encode(self, label: str) -> int:
        return self.label_to_id[label]

    def decode(self, index: int) -> str:
        return self.id_to_label[index]


@dataclass(frozen=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    pad_index: int = 0
    unk_index: int = 1

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1, max_size: int = 20000) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        words = [token for token, freq in counter.most_common() if freq >= min_freq]
        words = words[: max(0, max_size - 2)]
        itos = ["<pad>", "<unk>", *words]
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_length: int) -> list[int]:
        token_ids = [self.stoi.get(token, self.unk_index) for token in tokenize(text)]
        token_ids = token_ids[:max_length]
        if len(token_ids) < max_length:
            token_ids.extend([self.pad_index] * (max_length - len(token_ids)))
        return token_ids


class LogDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], vocab: Vocab, label_encoder: LabelEncoder, max_length: int) -> None:
        self.rows = rows
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        token_ids = self.vocab.encode(row["log_text"], self.max_length)
        label_id = self.label_encoder.encode(row["label"])
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)


def load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    required_columns = {"log_text", "label"}
    missing = required_columns - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV must contain columns: {sorted(required_columns)}; missing: {sorted(missing)}")
    return rows


def split_rows(rows: list[dict[str, str]], valid_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1.")

    shuffled_rows = rows[:]
    random.Random(seed).shuffle(shuffled_rows)
    split_index = int(len(shuffled_rows) * (1.0 - valid_ratio))
    split_index = min(max(split_index, 1), len(shuffled_rows) - 1)
    return shuffled_rows[:split_index], shuffled_rows[split_index:]
