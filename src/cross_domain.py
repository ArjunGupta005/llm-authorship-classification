import argparse
from collections import Counter
from pathlib import Path
import json
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

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
from train import set_seed, file_hash, run_epoch, save_curves


ROOT = Path(__file__).resolve().parents[1]
HELD_OUT_DOMAIN = "writing"
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cnn", "lstm"], required=True
    )
    args = parser.parse_args()

    set_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    run_name = f"{args.model}_output_no_writing_seed{SEED}"
    output_dir = ROOT / "results" / run_name

    if output_dir.exists():
        raise FileExistsError(
            f"Run folder already exists: {output_dir}"
        )

    train_path = ROOT / "data" / "processed" / "train.jsonl"
    validation_path = ROOT / "data" / "processed" / "validation.jsonl"

    original_training = read_jsonl(train_path)
    original_validation = read_jsonl(validation_path)

    # Exclude entire groups containing writing examples in these splits.
    # The test set is not read.
    excluded_groups = {
        row["group_id"]
        for row in original_training + original_validation
        if row["domain"] == HELD_OUT_DOMAIN
    }

    training_rows = [
        row for row in original_training
        if row["group_id"] not in excluded_groups
    ]
    validation_rows = [
        row for row in original_validation
        if row["group_id"] not in excluded_groups
    ]

    for name, rows in [
        ("training", training_rows),
        ("validation", validation_rows),
    ]:
        if set(row["llm_name"] for row in rows) != set(LABELS):
            raise ValueError(f"Missing families in {name} data.")
        if any(row["domain"] == HELD_OUT_DOMAIN for row in rows):
            raise ValueError(f"Writing examples remain in {name} data.")

    if (
        {row["group_id"] for row in training_rows}
        & {row["group_id"] for row in validation_rows}
    ):
        raise ValueError("Training and validation groups overlap.")

    vocab = build_vocab(training_rows, mode="output")

    training_loader = DataLoader(
        TextDataset(training_rows, vocab, "output"),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        TextDataset(validation_rows, vocab, "output"),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

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
        model = TextCNN(**model_kwargs)
    else:
        model_kwargs["hidden_size"] = 64
        model = TextLSTM(**model_kwargs)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    output_dir.mkdir(parents=True)

    config = {
        "model": args.model,
        "mode": "output",
        "seed": SEED,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "held_out_domain": HELD_OUT_DOMAIN,
        "epochs": 15,
        "patience": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "gradient_clip_norm": 1.0,
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
        "selection_metric": "non_writing_validation_macro_f1",
        "domain_labels": "Preliminary prompt keyword labels",
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
    print("Training counts:", config["training_counts"], flush=True)
    print("Validation counts:", config["validation_counts"], flush=True)
    print("Writing excluded from training and validation.", flush=True)
    print("The test set will not be loaded.", flush=True)

    history = []
    best_f1 = -1.0
    best_epoch = None
    best_validation = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, 16):
        epoch_started = time.perf_counter()

        training_metrics = run_epoch(
            model, training_loader, criterion, device, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in training_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            "seconds": time.perf_counter() - epoch_started,
        })

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv", index=False
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train F1={training_metrics['macro_f1']:.4f} | "
            f"non-writing validation "
            f"accuracy={validation_metrics['accuracy']:.4f} "
            f"F1={validation_metrics['macro_f1']:.4f} | "
            f"{history[-1]['seconds']:.1f}s",
            flush=True,
        )

        if validation_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_validation = validation_metrics
            stale_epochs = 0

            torch.save(
                {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                output_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1

        if stale_epochs >= 3:
            print("Early stopping.", flush=True)
            break

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation,
        "epochs_completed": len(history),
        "total_seconds": time.perf_counter() - started,
        "held_out_domain": HELD_OUT_DOMAIN,
        "test_evaluated": False,
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    save_curves(history, output_dir / "training_curves.png")

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best non-writing validation macro-F1: {best_f1:.4f}")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()