from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.log_classifier import LogClassifier
from src.log_data import LabelEncoder, LogDataset, Vocab, load_rows, split_rows
from src.utils import ensure_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an error log classifier with PyTorch.")
    parser.add_argument("--data-path", type=str, default="data/logs/train.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/log_classifier")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model: LogClassifier, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for token_ids, labels in dataloader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            logits = model(token_ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    rows = load_rows(args.data_path)
    train_rows, valid_rows = split_rows(rows, valid_ratio=args.valid_ratio, seed=args.seed)

    vocab = Vocab.build([row["log_text"] for row in train_rows], min_freq=args.min_freq, max_size=args.max_vocab_size)
    label_encoder = LabelEncoder.build([row["label"] for row in rows])

    train_dataset = LogDataset(train_rows, vocab=vocab, label_encoder=label_encoder, max_length=args.max_length)
    valid_dataset = LogDataset(valid_rows, vocab=vocab, label_encoder=label_encoder, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = LogClassifier(
        vocab_size=len(vocab.itos),
        num_classes=len(label_encoder.label_to_id),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_index=vocab.pad_index,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_valid_acc = 0.0
    output_dir = ensure_dir(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for token_ids, labels in train_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        valid_loss, valid_acc = evaluate(model, valid_loader, device)
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Valid Loss: {valid_loss:.4f} "
            f"Valid Acc: {valid_acc:.4f}"
        )

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    metadata = {
        "vocab": vocab.itos,
        "label_to_id": label_encoder.label_to_id,
        "config": {
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "pad_index": vocab.pad_index,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Training finished. Best validation accuracy: {best_valid_acc:.4f}")
    print(f"Saved model to: {(output_dir / 'best_model.pt').resolve()}")
    print(f"Saved metadata to: {(output_dir / 'metadata.json').resolve()}")


if __name__ == "__main__":
    train()
