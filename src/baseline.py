from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from dataset import LABELS, input_tokens, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def identity_tokens(tokens):
    """Input is already tokenized using the neural models' tokenizer."""
    return tokens


def main():
    train_path = ROOT / "data" / "processed" / "train.jsonl"
    validation_path = ROOT / "data" / "processed" / "validation.jsonl"

    training_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)

    training_groups = {row["group_id"] for row in training_rows}
    validation_groups = {row["group_id"] for row in validation_rows}

    if training_groups & validation_groups:
        raise ValueError("Training and validation groups overlap.")

    output_dir = ROOT / "results" / "baselines"

    if (output_dir / "validation_results.csv").exists():
        raise FileExistsError(
            "Baseline results already exist. Preserve the previous run "
            "before rerunning."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    y_train = [row["llm_name"] for row in training_rows]
    y_validation = [row["llm_name"] for row in validation_rows]

    results = []

    for mode in ["input", "output", "both"]:
        print(f"\nTraining baseline: {mode}", flush=True)
        started = time.perf_counter()

        # Match the neural networks' token budgets and case handling.
        x_train = [
            input_tokens(row, mode) for row in training_rows
        ]
        x_validation = [
            input_tokens(row, mode) for row in validation_rows
        ]

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=identity_tokens,
                    preprocessor=identity_tokens,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, 2),
                    max_features=20_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ])

        # Vocabulary, IDF weights, and classifier fit on training only.
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_validation)

        result = {
            "model": "tfidf_logistic_regression",
            "mode": mode,
            "validation_accuracy": accuracy_score(
                y_validation, predictions
            ),
            "validation_macro_f1": f1_score(
                y_validation,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "features": len(
                pipeline.named_steps["tfidf"].get_feature_names_out()
            ),
            "seconds": time.perf_counter() - started,
        }
        results.append(result)

        # Only load this file later if it is your own saved model.
        joblib.dump(
            pipeline,
            output_dir / f"{mode}_pipeline.joblib",
        )

        print(
            f"Validation accuracy: "
            f"{result['validation_accuracy']:.4f}"
        )
        print(
            f"Validation macro-F1: "
            f"{result['validation_macro_f1']:.4f}",
            flush=True,
        )

    pd.DataFrame(results).to_csv(
        output_dir / "validation_results.csv",
        index=False,
    )

    config = {
        "model": "TF-IDF plus logistic regression",
        "ngram_range": [1, 2],
        "max_features": 20_000,
        "min_df": 2,
        "sublinear_tf": True,
        "C": 1.0,
        "max_iter": 2_000,
        "seed": 42,
        "tokenization": "Same tokenizer and budgets as neural networks",
        "test_evaluated": False,
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nBaseline validation results:")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nThe test set was not loaded.")


if __name__ == "__main__":
    main()