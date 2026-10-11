from collections import Counter
from pathlib import Path
import json

from datasets import load_dataset


DATASET_NAME = "lmsys/lmsys-chat-1m"
INSPECTION_LIMIT = 10_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    print(f"Opening {DATASET_NAME}...", flush=True)

    # Streaming avoids downloading the entire dataset upfront.
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    model_counts = Counter()
    english_counts = Counter()
    first_record = None
    inspected = 0

    for record in dataset:
        if first_record is None:
            first_record = record

        model = str(record.get("model", "unknown"))
        language = str(record.get("language", "unknown"))

        model_counts[model] += 1

        if language.lower() in {"english", "en"}:
            english_counts[model] += 1

        inspected += 1

        if inspected % 1_000 == 0:
            print(f"Inspected {inspected:,} records...", flush=True)

        if inspected >= INSPECTION_LIMIT:
            break

    if first_record is None:
        raise RuntimeError("The dataset returned no records.")

    print("\nFields available:")
    print(list(first_record.keys()))

    print("\nLanguage label on the first record:")
    print(first_record.get("language"))

    print(f"\nModel counts within the first {inspected:,} records:")
    for model, count in model_counts.most_common():
        print(
            f"{model:35s} total={count:6,d} "
            f"English={english_counts[model]:6,d}"
        )

    conversation = first_record.get("conversation", [])
    print("\nConversation structure:")
    print("Container type:", type(conversation).__name__)

    if isinstance(conversation, list):
        print("Number of messages:", len(conversation))
        for message in conversation[:2]:
            # Inspect structure without printing user conversation content.
            if isinstance(message, dict):
                content = message.get("content", "")
                print({
                    "keys": list(message.keys()),
                    "role": message.get("role"),
                    "content_type": type(content).__name__,
                    "content_characters": (
                        len(content) if isinstance(content, str) else None
                    ),
                })

    result = {
        "dataset": DATASET_NAME,
        "records_inspected": inspected,
        "model_counts": dict(model_counts),
        "english_model_counts": dict(english_counts),
        "fields": list(first_record.keys()),
    }

    output_path = PROJECT_ROOT / "results" / "dataset_inspection.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved inspection summary to {output_path}")


if __name__ == "__main__":
    main()