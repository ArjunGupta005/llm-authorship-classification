import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataset import (
    LABELS,
    PROMPT_LIMIT,
    RESPONSE_LIMIT,
    TextDataset,
    build_vocab,
    collate_batch,
    read_jsonl,
)
from models import TextCNN, TextLSTM


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    actual = []
    predicted = []

    with torch.set_grad_enabled(training):
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(tokens, lengths)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            actual.extend(labels.detach().cpu().tolist())
            predicted.extend(
                logits.argmax(dim=1).detach().cpu().tolist()
            )

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        ),
    }


def save_curves(history, output_path):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        frame["epoch"], frame["train_loss"], label="Training"
    )
    axes[0].plot(
        frame["epoch"], frame["validation_loss"], label="Validation"
    )
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        frame["epoch"], frame["train_macro_f1"], label="Training"
    )
    axes[1].plot(
        frame["epoch"],
        frame["validation_macro_f1"],
        label="Validation",
    )
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cnn", "lstm"], required=True
    )
    parser.add_argument(
        "--mode", choices=["input", "output", "both"], required=True
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if min(
        args.epochs,
        args.patience,
        args.batch_size,
        args.threads,
    ) < 1:
        parser.error("Epochs, patience, batch size, and threads must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")

    set_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_name = f"{args.model}_{args.mode}_seed{args.seed}"
    output_dir = ROOT / "results" / run_name

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run folder already exists: {output_dir}\n"
            "Use --overwrite only if you intend to replace this run."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = ROOT / "data" / "processed" / "train.jsonl"
    validation_path = ROOT / "data" / "processed" / "validation.jsonl"

    training_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)

    for name, rows in [
        ("training", training_rows),
        ("validation", validation_rows),
    ]:
        if set(row["llm_name"] for row in rows) != set(LABELS):
            raise ValueError(f"The {name} data must contain all three families.")

    training_groups = {row["group_id"] for row in training_rows}
    validation_groups = {row["group_id"] for row in validation_rows}
    if training_groups & validation_groups:
        raise ValueError("Training and validation groups overlap.")

    vocab = build_vocab(training_rows, args.mode)

    training_dataset = TextDataset(
        training_rows, vocab, args.mode
    )
    validation_dataset = TextDataset(
        validation_rows, vocab, args.mode
    )

    generator = torch.Generator().manual_seed(args.seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    model_class = TextCNN if args.model == "cnn" else TextLSTM

    model_kwargs = {
        "vocab_size": len(vocab),
        "embedding_dim": 100,
        "num_classes": len(LABELS),
        "dropout": 0.4,
    }

    if args.model == "cnn":
        model_kwargs.update({
            "filters": 64,
            "kernel_sizes": (3, 4, 5),
        })
    else:
        model_kwargs["hidden_size"] = 64

    model = model_class(**model_kwargs).to(device)

    # Classes remain close to balanced, so use ordinary cross-entropy.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    config = {
        **vars(args),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "vocab_size": len(vocab),
        "labels": LABELS,
        "prompt_limit": PROMPT_LIMIT,
        "response_limit": RESPONSE_LIMIT,
        "model_kwargs": model_kwargs,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training_counts": dict(
            Counter(row["llm_name"] for row in training_rows)
        ),
        "validation_counts": dict(
            Counter(row["llm_name"] for row in validation_rows)
        ),
        "training_file_sha256": file_hash(train_path),
        "validation_file_sha256": file_hash(validation_path),
        "selection_metric": "validation_macro_f1",
        "gradient_clip_norm": 1.0,
        "optimizer": "Adam",
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nRun: {run_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Vocabulary: {len(vocab):,} tokens", flush=True)
    print(f"Parameters: {config['parameter_count']:,}", flush=True)
    print("The test set will not be loaded.", flush=True)

    history = []
    best_f1 = -1.0
    best_epoch = None
    best_validation = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()

        train_metrics = run_epoch(
            model,
            training_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "seconds": time.perf_counter() - epoch_started,
        })

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"F1={train_metrics['macro_f1']:.4f} | "
            f"validation loss={validation_metrics['loss']:.4f} "
            f"accuracy={validation_metrics['accuracy']:.4f} "
            f"F1={validation_metrics['macro_f1']:.4f} | "
            f"{history[-1]['seconds']:.1f}s",
            flush=True,
        )

        if validation_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_validation = validation_metrics
            epochs_without_improvement = 0

            # Save CPU tensors for portable loading.
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(state, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping.", flush=True)
            break

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation,
        "epochs_completed": len(history),
        "total_seconds": time.perf_counter() - started,
        "test_evaluated": False,
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    save_curves(history, output_dir / "training_curves.png")

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()