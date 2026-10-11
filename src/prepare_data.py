from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from langdetect import DetectorFactory, detect, LangDetectException


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DetectorFactory.seed = SEED

SPLITS = ["train", "validation", "test"]

CODING_PATTERN = re.compile(
    r"```|"
    r"\b(python|javascript|typescript|powershell|streamlit|"
    r"netlogo|numpy|pandas|html|css|sql|debug|programming)\b|"
    r"\bc\+\+(?=\W|$)|"
    r"\b(write|create|implement|fix|explain|review)\b"
    r".{0,100}\b(code|function|script|algorithm|app)\b|"
    r"\brust\s+(enum|code|function|struct)\b",
    re.IGNORECASE | re.DOTALL,
)

SUMMARY_PATTERN = re.compile(
    r"\b(summarize|summarise|summary)\b|"
    r"\bbriefly describe\b.{0,80}\b(excerpt|text|article)\b",
    re.IGNORECASE | re.DOTALL,
)

TRANSCRIPT_PATTERN = re.compile(
    r"\b(organize|format|edit)\b.{0,80}\btranscript\b",
    re.IGNORECASE | re.DOTALL,
)

WRITING_PATTERN = re.compile(
    r"\b(poem|poetry|story|essay|fiction|screenplay|"
    r"blog post|cover letter|tweets|rewrite|rephrase|"
    r"proofread|trademark licence agreement)\b|"
    r"\b(write|compose|draft)\b.{0,80}"
    r"\b(article|letter|email|speech|agreement|paragraph)\b",
    re.IGNORECASE | re.DOTALL,
)

MATH_PATTERN = re.compile(
    r"\b(integral|derivative|algebra|geometry|probability|"
    r"variance|mathematics|polynomial|prime number|theorem|"
    r"eigenvalue|equation)\b|"
    r"\bcalculate\b.{0,50}\d",
    re.IGNORECASE | re.DOTALL,
)


def normalize(text):
    """Normalize for duplicate checks without changing saved text."""
    return " ".join(text.casefold().split())


def digest(text):
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


def assign_domain(prompt):
    """Assign preliminary domains using only the user's prompt."""
    if SUMMARY_PATTERN.search(prompt):
        return "summarization"

    if TRANSCRIPT_PATTERN.search(prompt):
        return "writing"

    # Writing code counts as coding rather than prose writing.
    if CODING_PATTERN.search(prompt):
        return "coding"

    if WRITING_PATTERN.search(prompt):
        return "writing"

    if MATH_PATTERN.search(prompt):
        return "math"

    return "general"


def detected_language(text):
    """Detect prose language; retain short/code-heavy text as uncertain."""
    prose = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    prose = re.sub(r"https?://\S+", " ", prose)

    if len(re.findall(r"[^\W\d_]", prose)) < 40:
        return "uncertain"

    try:
        return detect(prose)
    except LangDetectException:
        return "uncertain"


def duplicate_groups(df):
    """
    Connect rows sharing an exact normalized prompt, response,
    or conversation ID. Connected rows stay in the same split.
    """
    parent = list(range(len(df)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for column in ["llm_input", "llm_output", "conversation_id"]:
        seen = {}

        for index, value in enumerate(df[column]):
            key = (
                digest(value)
                if column != "conversation_id"
                else value
            )

            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    return [find(index) for index in range(len(df))]


def main():
    source = ROOT / "data" / "raw" / "llm_responses.jsonl"
    output_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source}\n"
            "Run src/collect_data.py first."
        )

    df = pd.read_json(source, lines=True)
    original_count = len(df)

    required_columns = [
        "sample_id",
        "llm_name",
        "llm_input",
        "llm_output",
        "conversation_id",
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if not df["sample_id"].is_unique:
        raise ValueError("Duplicate sample IDs found.")

    for column in required_columns:
        valid = df[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            raise ValueError(f"Missing or invalid text in {column}.")

    expected_families = {"GPT", "Claude", "Llama"}
    if set(df["llm_name"]) != expected_families:
        raise ValueError(
            "Expected the three families GPT, Claude, and Llama."
        )

    print("Checking prompt and response languages...", flush=True)

    df["prompt_language"] = df["llm_input"].map(detected_language)
    df["response_language"] = df["llm_output"].map(detected_language)

    keep = (
        df["prompt_language"].isin(["en", "uncertain"])
        & df["response_language"].isin(["en", "uncertain"])
    )

    excluded = df.loc[~keep].copy()
    exclusion_counts = (
        excluded["llm_name"]
        .value_counts()
        .reindex(sorted(expected_families), fill_value=0)
    )

    print("\nLanguage exclusions by family:")
    print(exclusion_counts.to_string())

    excluded[
        [
            "sample_id",
            "llm_name",
            "prompt_language",
            "response_language",
        ]
    ].to_csv(
        results_dir / "language_exclusions.csv",
        index=False,
    )

    df = df.loc[keep].reset_index(drop=True)

    family_counts = df["llm_name"].value_counts()
    if (
        set(family_counts.index) != expected_families
        or family_counts.min() < 10
    ):
        raise ValueError(
            "Insufficient examples for one or more families "
            "after language filtering."
        )

    df["domain"] = df["llm_input"].map(assign_domain)
    df["group_id"] = duplicate_groups(df)

    if df["group_id"].nunique() < 10:
        raise ValueError("At least ten duplicate groups are required.")

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=SEED,
    )

    fold_ids = np.full(len(df), -1, dtype=int)

    for fold, (_, indices) in enumerate(
        splitter.split(
            df,
            df["llm_name"],
            groups=df["group_id"],
        )
    ):
        fold_ids[indices] = fold

    if not (fold_ids >= 0).all():
        raise RuntimeError("Some examples were not assigned a fold.")

    df["split"] = [
        "test" if fold < 2
        else "validation" if fold == 2
        else "train"
        for fold in fold_ids
    ]

    # Confirm each split contains all three families.
    for split in SPLITS:
        subset = df[df["split"] == split]
        if set(subset["llm_name"]) != expected_families:
            raise RuntimeError(
                f"The {split} split does not contain every family."
            )

    # Confirm duplicate groups never cross split boundaries.
    if df.groupby("group_id")["split"].nunique().max() != 1:
        raise RuntimeError("Duplicate groups cross split boundaries.")

    for column in ["llm_input", "llm_output", "conversation_id"]:
        keys = (
            df[column].map(digest)
            if column != "conversation_id"
            else df[column]
        )
        checks = pd.DataFrame({
            "key": keys,
            "split": df["split"],
        })

        if checks.groupby("key")["split"].nunique().max() != 1:
            raise RuntimeError(
                f"Duplicate {column} values cross split boundaries."
            )

    for split in SPLITS:
        df[df["split"] == split].to_json(
            output_dir / f"{split}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )

    df[
        ["sample_id", "llm_name", "domain", "group_id", "split"]
    ].to_csv(
        results_dir / "split_manifest.csv",
        index=False,
    )

    split_counts = pd.crosstab(
        df["split"], df["llm_name"]
    ).reindex(SPLITS)

    domain_counts = pd.crosstab(
        df["domain"], df["llm_name"]
    )

    domain_split_counts = pd.crosstab(
        [df["split"], df["domain"]],
        df["llm_name"],
    )

    split_counts.to_csv(results_dir / "split_counts.csv")
    domain_counts.to_csv(results_dir / "domain_counts.csv")
    domain_split_counts.to_csv(
        results_dir / "domain_split_counts.csv"
    )

    # Review training examples only, sampled across domains and families.
    training = df[df["split"] == "train"]
    review_parts = []

    for _, subset in training.groupby(["domain", "llm_name"]):
        review_parts.append(
            subset.sample(
                n=min(2, len(subset)),
                random_state=SEED,
            )
        )

    review = pd.concat(review_parts).drop_duplicates("sample_id")
    review_path = output_dir / "data_review.txt"

    with review_path.open("w", encoding="utf-8") as file:
        file.write(
            "TRAINING DATA REVIEW\n"
            "Check domain labels, sensitive information, and "
            "prompt-response alignment.\n"
            "Do not correct model-generated factual or coding errors.\n"
            "Response previews are truncated; inspect the full JSONL "
            "record when needed.\n\n"
        )

        for _, row in review.iterrows():
            file.write(
                f"{'=' * 60}\n"
                f"Sample: {row['sample_id']}\n"
                f"Family: {row['llm_name']}\n"
                f"Preliminary domain: {row['domain']}\n"
                f"Prompt language: {row['prompt_language']}\n"
                f"Response language: {row['response_language']}\n\n"
                f"PROMPT:\n{row['llm_input']}\n\n"
                f"RESPONSE PREVIEW:\n{row['llm_output'][:1800]}\n\n"
            )

    summary = {
        "seed": SEED,
        "original_examples": original_count,
        "language_excluded_examples": len(excluded),
        "language_exclusions_by_family": exclusion_counts.to_dict(),
        "total_examples": len(df),
        "duplicate_groups": int(df["group_id"].nunique()),
        "split_counts": split_counts.to_dict(),
        "domain_counts": domain_counts.to_dict(),
        "domain_method": (
            "Ordered prompt keyword rules; preliminary heuristic labels."
        ),
        "language_method": (
            "langdetect on prompt and response separately after removing "
            "fenced code and URLs. English and uncertain examples retained."
        ),
        "language_limitations": (
            "Automatic detection can misclassify technical or multilingual "
            "text. Short/code-heavy examples remain uncertain."
        ),
        "leakage_checks": (
            "Exact normalized prompts, responses, and conversation IDs "
            "are grouped. Semantic near-duplicates are not guaranteed "
            "to be grouped."
        ),
        "manual_review_completed": False,
    }

    (results_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSplit counts:")
    print(split_counts.to_string())

    print("\nPreliminary domain counts:")
    print(domain_counts.to_string())

    print("\nDomain counts within each split:")
    print(domain_split_counts.to_string())

    print("\nExact-duplicate leakage checks passed.")
    print(f"Retained {len(df):,} of {original_count:,} examples.")
    print("Review file:", review_path)
    print("Preparation complete.")


if __name__ == "__main__":
    main()