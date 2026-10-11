from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import identity_tokens
from dataset import LABELS, input_tokens, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def style_features(text):
    """Descriptive features of the full response."""
    words = re.findall(r"\b[^\W\d_]+\b", text.casefold())
    first_words = words[:100]
    characters = max(len(text), 1)

    return {
        "characters": len(text),
        "words": len(words),
        "mean_word_length": (
            float(np.mean([len(word) for word in words]))
            if words else 0.0
        ),
        "first_100_word_ttr": (
            len(set(first_words)) / len(first_words)
            if first_words else 0.0
        ),
        "newlines_per_1000_chars": text.count("\n") * 1000 / characters,
        "bullet_lines": len(re.findall(
            r"(?m)^\s*[-*•]\s+", text
        )),
        "numbered_lines": len(re.findall(
            r"(?m)^\s*\d+[.)]\s+", text
        )),
        "code_fences": text.count("```"),
        "question_marks_per_1000_chars": (
            text.count("?") * 1000 / characters
        ),
        "exclamations_per_1000_chars": (
            text.count("!") * 1000 / characters
        ),
        "punctuation_fraction": (
            sum(
                not character.isalnum() and not character.isspace()
                for character in text
            ) / characters
        ),
    }


def lexical_tokens(row):
    """
    Remove punctuation, underscores, and newline markers from the
    original response-only token span. Keep words/numbers and case.
    """
    return [
        token
        for token in input_tokens(row, "output")
        if token != "<NL>" and token.isalnum()
    ]


def metrics(actual, predicted):
    return {
        "validation_accuracy": accuracy_score(actual, predicted),
        "validation_macro_f1": f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    output_dir = ROOT / "results" / "style_analysis"

    if (output_dir / "validation_results.csv").exists():
        raise FileExistsError("Style-analysis results already exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    training = read_jsonl(
        ROOT / "data" / "processed" / "train.jsonl"
    )
    validation = read_jsonl(
        ROOT / "data" / "processed" / "validation.jsonl"
    )

    if (
        {row["group_id"] for row in training}
        & {row["group_id"] for row in validation}
    ):
        raise ValueError("Training and validation groups overlap.")

    train_features = pd.DataFrame([
        style_features(row["llm_output"]) for row in training
    ])
    validation_features = pd.DataFrame([
        style_features(row["llm_output"]) for row in validation
    ])

    y_train = [row["llm_name"] for row in training]
    y_validation = [row["llm_name"] for row in validation]

    # Descriptive summaries use training data only.
    descriptive = train_features.copy()
    descriptive["family"] = y_train

    means = descriptive.groupby("family").mean().reindex(LABELS)
    medians = descriptive.groupby("family").median().reindex(LABELS)

    means.to_csv(output_dir / "training_feature_means.csv")
    medians.to_csv(output_dir / "training_feature_medians.csv")

    print("\nTraining feature means:")
    print(means.round(3).to_string())

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    for axis, feature, title in zip(
        axes,
        ["words", "first_100_word_ttr", "newlines_per_1000_chars"],
        ["Response word count", "First 100 words: diversity", "Line-break density"],
    ):
        axis.bar(LABELS, means.loc[LABELS, feature].values)
        axis.set_title(title)
        axis.set_ylabel("Training-set mean")

    figure.tight_layout()
    figure.savefig(output_dir / "style_comparison.png", dpi=180)
    plt.close(figure)

    # Style-only classifier: scaling is fit on training data only.
    style_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    style_model.fit(train_features, y_train)
    style_result = {
        "experiment": "style_features_only",
        **metrics(
            y_validation,
            style_model.predict(validation_features),
        ),
    }

    joblib.dump(
        style_model, output_dir / "style_pipeline.joblib"
    )

    # Save signed associations; these are not causal explanations.
    classifier = style_model.named_steps["classifier"]
    pd.DataFrame(
        classifier.coef_,
        index=classifier.classes_,
        columns=train_features.columns,
    ).to_csv(output_dir / "standardized_style_coefficients.csv")

    lexical_model = Pipeline([
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
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])

    lexical_model.fit(
        [lexical_tokens(row) for row in training],
        y_train,
    )

    lexical_result = {
        "experiment": "tfidf_without_punctuation_or_newlines",
        **metrics(
            y_validation,
            lexical_model.predict([
                lexical_tokens(row) for row in validation
            ]),
        ),
    }

    joblib.dump(
        lexical_model, output_dir / "lexical_pipeline.joblib"
    )

    original_results = pd.read_csv(
        ROOT / "results" / "baselines" / "validation_results.csv"
    )
    original = original_results[
        original_results["mode"] == "output"
    ].iloc[0]

    original_result = {
        "experiment": "original_output_tfidf",
        "validation_accuracy": float(original["validation_accuracy"]),
        "validation_macro_f1": float(original["validation_macro_f1"]),
    }

    results = pd.DataFrame([
        original_result,
        lexical_result,
        style_result,
    ])

    results.to_csv(
        output_dir / "validation_results.csv", index=False
    )

    config = {
        "style_feature_names": list(train_features.columns),
        "style_text_scope": "Full response",
        "lexical_ablation": (
            "Remove punctuation, underscores, and newline tokens "
            "after selecting the original first 256 response tokens."
        ),
        "classifier": "Logistic regression",
        "C": 1.0,
        "seed": 42,
        "test_evaluated": False,
        "limitations": [
            "Style features can reflect task mix as well as generator style.",
            "Type-token ratio remains length-sensitive for responses under 100 words.",
            "Ablation changes the feature space and creates new word adjacencies.",
            "Style-only models use full responses; TF-IDF uses truncated responses.",
            "These analyses do not directly attribute CNN or LSTM decisions.",
        ],
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nRQ4 validation results:")
    print(results.to_string(index=False))
    print("\nThe test set was not loaded.")


if __name__ == "__main__":
    main()