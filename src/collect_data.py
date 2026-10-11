from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import re

from datasets import load_dataset
#from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "lmsys/lmsys-chat-1m"

MODEL_FAMILIES = {
    "gpt-4": "GPT",
    "claude-1": "Claude",
    "llama-2-13b-chat": "Llama",
}

TARGET_PER_FAMILY = 1_000
CANDIDATES_PER_FAMILY = 1_500
MAX_RECORDS = 500_000
SEED = 42

# Conservative filters. These do not replace a later manual review.
IDENTITY_PATTERN = re.compile(
    r"\b(chatgpt|openai|gpt[- ]?\d*|claude|anthropic|"
    r"llama|meta ai|vicuna|google|gemini|palm)\b",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)


def normalized(text):
    """Used for duplicate checks, not to alter saved text."""
    return " ".join(text.casefold().split())


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_flagged_content(value):
    """Find explicit moderation flags in nested metadata."""
    if isinstance(value, dict):
        if value.get("flagged") is True:
            return True
        return any(has_flagged_content(v) for v in value.values())
    if isinstance(value, list):
        return any(has_flagged_content(v) for v in value)
    return False


def main():
    # Pin the dataset revision so this collection can be reproduced.
    #revision = HfApi().dataset_info(DATASET_NAME).sha
    revision = "200748d9d3cddcc9d782887541057aca0b18c5da"
    print(f"Dataset revision: {revision}", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        revision=revision,
        split="train",
        streaming=True,
    )

    candidates = {family: [] for family in MODEL_FAMILIES.values()}
    rejected = Counter()
    seen_conversations = set()
    seen_pairs = set()
    scanned = 0

    for record in dataset:
        scanned += 1
        model = record.get("model")
        family = MODEL_FAMILIES.get(model)

        if family is not None:
            if len(candidates[family]) >= CANDIDATES_PER_FAMILY:
                rejected["candidate_pool_full"] += 1
            else:
                reason = None
                conversation = record.get("conversation", [])
                conversation_id = str(record.get("conversation_id", ""))

                if record.get("language") != "English":
                    reason = "not_english"
                elif record.get("redacted") is True:
                    reason = "redacted"
                elif has_flagged_content(record.get("openai_moderation", [])):
                    reason = "moderation_flagged"
                elif not conversation_id:
                    reason = "missing_conversation_id"
                elif conversation_id in seen_conversations:
                    reason = "duplicate_conversation"
                elif (
                    not isinstance(conversation, list)
                    or len(conversation) < 2
                    or not isinstance(conversation[0], dict)
                    or not isinstance(conversation[1], dict)
                    or conversation[0].get("role") != "user"
                    or conversation[1].get("role") != "assistant"
                ):
                    reason = "unexpected_first_turn"
                else:
                    prompt = conversation[0].get("content")
                    response = conversation[1].get("content")

                    if not isinstance(prompt, str) or not isinstance(response, str):
                        reason = "non_text"
                    else:
                        prompt = prompt.strip()
                        response = response.strip()
                        combined = prompt + "\n" + response
                        pair_key = text_hash(
                            normalized(prompt) + "\n" + normalized(response)
                        )

                        if not 20 <= len(prompt) <= 10_000:
                            reason = "prompt_length"
                        elif not 100 <= len(response) <= 20_000:
                            reason = "response_length"
                        elif IDENTITY_PATTERN.search(combined):
                            reason = "explicit_model_or_provider_name"
                        elif EMAIL_PATTERN.search(combined):
                            reason = "email_address"
                        elif pair_key in seen_pairs:
                            reason = "duplicate_pair"
                        else:
                            seen_pairs.add(pair_key)
                            seen_conversations.add(conversation_id)

                            candidates[family].append({
                                "sample_id": f"{model}:{conversation_id}",
                                "prompt_id": text_hash(normalized(prompt)),
                                "conversation_id": conversation_id,
                                "llm_name": family,
                                "model_version": model,
                                "llm_input": prompt,
                                "llm_output": response,
                                "domain": "unassigned",
                                "source": DATASET_NAME,
                                "source_revision": revision,
                            })

                if reason:
                    rejected[reason] += 1

        if scanned % 10_000 == 0:
            counts = {k: len(v) for k, v in candidates.items()}
            print(f"Scanned {scanned:,}: {counts}", flush=True)

        if all(
            len(rows) >= CANDIDATES_PER_FAMILY
            for rows in candidates.values()
        ):
            break

        if scanned >= MAX_RECORDS:
            break

    counts = {k: len(v) for k, v in candidates.items()}
    print("\nEligible candidate counts:", counts)

    if min(counts.values()) < TARGET_PER_FAMILY:
        raise RuntimeError(
            "Not enough eligible examples for every family. "
            "Send the candidate counts before changing the settings."
        )

    rng = random.Random(SEED)
    selected = []

    for family in sorted(candidates):
        selected.extend(
            rng.sample(candidates[family], TARGET_PER_FAMILY)
        )

    rng.shuffle(selected)

    output_path = ROOT / "data" / "raw" / "llm_responses.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": DATASET_NAME,
        "revision": revision,
        "seed": SEED,
        "records_scanned": scanned,
        "candidate_counts": counts,
        "selected_counts": dict(
            Counter(row["llm_name"] for row in selected)
        ),
        "rejection_counts": dict(rejected),
        "models": MODEL_FAMILIES,
        "sampling": (
            "Seeded random sample from capped eligible candidate pools "
            "within the scanned dataset prefix; not a uniform sample "
            "of the entire source dataset."
        ),
        "filters": {
            "language": "English",
            "first_turn_only": True,
            "prompt_characters": [20, 10_000],
            "response_characters": [100, 20_000],
            "exclude_redacted": True,
            "exclude_explicit_moderation_flags": True,
            "exclude_identity_pattern": IDENTITY_PATTERN.pattern,
            "exclude_email_addresses": True,
            "deduplicate": "conversation ID and normalized prompt-response pair",
        },
    }

    summary_path = ROOT / "results" / "collection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nCollection complete.")
    print("Selected counts:", summary["selected_counts"])
    print("Data saved to:", output_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()