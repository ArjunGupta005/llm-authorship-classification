from collections import Counter
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# Also makes identity_tokens available when loading baseline pipelines
# saved while baseline.py ran as __main__.
from baseline import identity_tokens
from analyze_style import style_features, lexical_tokens
from dataset import (
    LABELS,
    TextDataset,
    collate_batch,
    input_tokens,
    read_jsonl,
)
from models import TextCNN, TextLSTM


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINAL = RESULTS / "final_evaluation"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(actual, predicted):
    return {
        "examples": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }


def neural_predictions(run_dir, rows):
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    vocab = json.loads(
        (run_dir / "vocab.json").read_text(encoding="utf-8")
    )

    if config["labels"] != LABELS:
        raise ValueError(f"Unexpected label order in {run_dir.name}.")

    for split in ["training", "validation"]:
        filename = "train.jsonl" if split == "training" else "validation.jsonl"
        current_path = ROOT / "data" / "processed" / filename
        if file_hash(current_path) != config[f"{split}_file_sha256"]:
            raise ValueError(
                f"Processed {split} data changed after {run_dir.name} trained."
            )

    model_class = TextCNN if config["model"] == "cnn" else TextLSTM
    model = model_class(**config["model_kwargs"])

    state = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        TextDataset(rows, vocab, config["mode"]),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    predicted = []
    with torch.no_grad():
        for tokens, lengths, _ in loader:
            logits = model(tokens, lengths)
            predicted.extend(
                LABELS[index]
                for index in logits.argmax(dim=1).tolist()
            )

    return predicted


def save_result(name, rows, predicted, table):
    actual = [row["llm_name"] for row in rows]
    result = {"experiment": name, **scores(actual, predicted)}
    table.append(result)

    report = classification_report(
        actual,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    (FINAL / f"{name}_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    pd.DataFrame(
        matrix, index=LABELS, columns=LABELS
    ).to_csv(FINAL / f"{name}_confusion.csv")

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    ).plot(ax=axis, colorbar=False, cmap="Blues")
    axis.set_title(name.replace("_", " "))
    figure.tight_layout()
    figure.savefig(FINAL / f"{name}_confusion.png", dpi=180)
    plt.close(figure)

    # Save predictions without publishing conversation text.
    pd.DataFrame({
        "sample_id": [row["sample_id"] for row in rows],
        "group_id": [row["group_id"] for row in rows],
        "domain": [row["domain"] for row in rows],
        "actual": actual,
        "predicted": predicted,
    }).to_csv(FINAL / f"{name}_predictions.csv", index=False)

    print(
        f"{name}: accuracy={result['accuracy']:.4f}, "
        f"macro-F1={result['macro_f1']:.4f}",
        flush=True,
    )


def main():
    torch.set_num_threads(4)

    main_runs = [
        f"{model}_{mode}_seed42"
        for model in ["cnn", "lstm"]
        for mode in ["input", "output", "both"]
    ]
    cross_runs = [
        f"{model}_output_no_writing_seed42"
        for model in ["cnn", "lstm"]
    ]

    # Check required files before beginning evaluation.
    required = []
    for name in main_runs + cross_runs:
        required.extend([
            RESULTS / name / "config.json",
            RESULTS / name / "vocab.json",
            RESULTS / name / "best_model.pt",
        ])

    required.extend([
        RESULTS / "baselines" / f"{mode}_pipeline.joblib"
        for mode in ["input", "output", "both"]
    ])
    required.extend([
        RESULTS / "style_analysis" / "lexical_pipeline.joblib",
        RESULTS / "style_analysis" / "style_pipeline.joblib",
    ])

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment files:\n" + "\n".join(missing)
        )

    if (FINAL / "main_test_results.csv").exists():
        raise FileExistsError(
            "Final evaluation already exists. Preserve those results."
        )

    train_rows = read_jsonl(
        ROOT / "data" / "processed" / "train.jsonl"
    )
    validation_rows = read_jsonl(
        ROOT / "data" / "processed" / "validation.jsonl"
    )
    test_path = ROOT / "data" / "processed" / "test.jsonl"
    test_rows = read_jsonl(test_path)

    development_groups = {
        row["group_id"] for row in train_rows + validation_rows
    }
    test_groups = {row["group_id"] for row in test_rows}

    if development_groups & test_groups:
        raise ValueError("Test groups overlap development groups.")

    FINAL.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(test_rows)} test examples.", flush=True)
    main_results = []
    all_predictions = {}

    for name in main_runs:
        predicted = neural_predictions(RESULTS / name, test_rows)
        all_predictions[name] = predicted
        save_result(name, test_rows, predicted, main_results)

    for mode in ["input", "output", "both"]:
        name = f"tfidf_{mode}"
        pipeline = joblib.load(
            RESULTS / "baselines" / f"{mode}_pipeline.joblib"
        )
        predicted = pipeline.predict([
            input_tokens(row, mode) for row in test_rows
        ]).tolist()

        save_result(name, test_rows, predicted, main_results)

    # RQ3: Compare models on identical writing and non-writing subsets.
    cross_results = []

    for name in cross_runs:
        all_predictions[name] = neural_predictions(
            RESULTS / name, test_rows
        )

    for model in ["cnn", "lstm"]:
        for name in [
            f"{model}_output_seed42",
            f"{model}_output_no_writing_seed42",
        ]:
            for subset_name in ["writing", "non_writing"]:
                indices = [
                    index for index, row in enumerate(test_rows)
                    if (
                        row["domain"] == "writing"
                        if subset_name == "writing"
                        else row["domain"] != "writing"
                    )
                ]

                if not indices:
                    raise ValueError(f"Empty test subset: {subset_name}")

                subset_rows = [test_rows[index] for index in indices]
                predicted = [
                    all_predictions[name][index] for index in indices
                ]

                save_result(
                    f"{name}_{subset_name}",
                    subset_rows,
                    predicted,
                    cross_results,
                )

    # RQ4: Evaluate the preselected style-only and lexical models.
    style_results = []

    lexical_model = joblib.load(
        RESULTS / "style_analysis" / "lexical_pipeline.joblib"
    )
    lexical_predictions = lexical_model.predict([
        lexical_tokens(row) for row in test_rows
    ]).tolist()

    save_result(
        "tfidf_without_punctuation_or_newlines",
        test_rows,
        lexical_predictions,
        style_results,
    )

    style_model = joblib.load(
        RESULTS / "style_analysis" / "style_pipeline.joblib"
    )
    features = pd.DataFrame([
        style_features(row["llm_output"]) for row in test_rows
    ])

    save_result(
        "style_features_only",
        test_rows,
        style_model.predict(features).tolist(),
        style_results,
    )

    # Include the original TF-IDF response baseline for comparison.
    original = next(
        row for row in main_results
        if row["experiment"] == "tfidf_output"
    )
    style_results.insert(0, original.copy())

    main_frame = pd.DataFrame(main_results)
    cross_frame = pd.DataFrame(cross_results)
    style_frame = pd.DataFrame(style_results)

    cross_frame.to_csv(
        FINAL / "cross_domain_test_results.csv", index=False
    )
    style_frame.to_csv(
        FINAL / "style_test_results.csv", index=False
    )

    summary = {
        "test_file_sha256": file_hash(test_path),
        "test_examples": len(test_rows),
        "test_family_counts": dict(
            Counter(row["llm_name"] for row in test_rows)
        ),
        "writing_test_family_counts": dict(
            Counter(
                row["llm_name"] for row in test_rows
                if row["domain"] == "writing"
            )
        ),
        "uniform_random_expected_accuracy": 1 / len(LABELS),
        "majority_test_class_accuracy": (
            max(Counter(
                row["llm_name"] for row in test_rows
            ).values()) / len(test_rows)
        ),
        "notes": [
            "Checkpoints selected using validation, not test performance.",
            "Cross-domain writing results use a small, heuristic-labeled subset.",
            "Comparisons use one training seed; differences are descriptive.",
        ],
    }

    (FINAL / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Write this last to mark successful completion.
    main_frame.to_csv(
        FINAL / "main_test_results.csv", index=False
    )

    print("\nMAIN TEST RESULTS")
    print(main_frame.to_string(index=False))
    print("\nCROSS-DOMAIN TEST RESULTS")
    print(cross_frame.to_string(index=False))
    print("\nSTYLE TEST RESULTS")
    print(style_frame.to_string(index=False))
    print("\nSaved results to:", FINAL)


if __name__ == "__main__":
    main()