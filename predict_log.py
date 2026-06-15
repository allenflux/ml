from __future__ import annotations

import argparse
import json

import torch

from src.log_classifier import LogClassifier
from src.log_data import Vocab
from src.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an error log label.")
    parser.add_argument("--model-path", type=str, default="outputs/log_classifier/best_model.pt")
    parser.add_argument("--metadata-path", type=str, default="outputs/log_classifier/metadata.json")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    with open(args.metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    vocab_list = metadata["vocab"]
    label_to_id = metadata["label_to_id"]
    config = metadata["config"]
    vocab = Vocab(stoi={token: idx for idx, token in enumerate(vocab_list)}, itos=vocab_list)
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    model = LogClassifier(
        vocab_size=len(vocab.itos),
        num_classes=len(label_to_id),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        pad_index=config["pad_index"],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    token_ids = torch.tensor([vocab.encode(args.text, config["max_length"])], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(token_ids)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_id = int(probabilities.argmax().item())

    print(f"Predicted label: {id_to_label[predicted_id]}")
    print(f"Confidence: {probabilities[predicted_id].item():.4f}")


if __name__ == "__main__":
    main()
