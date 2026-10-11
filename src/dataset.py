from collections import Counter
import json
import re

import torch
from torch.utils.data import Dataset


LABELS = ["Claude", "GPT", "Llama"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

# Use the same prompt/response budgets across experiments.
PROMPT_LIMIT = 127
RESPONSE_LIMIT = 256

TOKEN_PATTERN = re.compile(r"\n|[^\W_]+|_|[^\w\s]", re.UNICODE)


def tokenize(text):
    """
    Preserve case, punctuation, and line breaks as potential style signals.
    Spaces and tabs are not retained.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        "<NL>" if token == "\n" else token
        for token in TOKEN_PATTERN.findall(text)
    ]


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def input_tokens(row, mode):
    prompt = tokenize(row["llm_input"])[:PROMPT_LIMIT]
    response = tokenize(row["llm_output"])[:RESPONSE_LIMIT]

    if mode == "input":
        return prompt
    if mode == "output":
        return response
    if mode == "both":
        return prompt + ["<SEP>"] + response

    raise ValueError(f"Unknown input mode: {mode}")


def build_vocab(training_rows, mode, max_vocab=20_000):
    """
    Fit only on training rows, using the text actually seen by the model.
    Validation/test words never influence vocabulary selection.
    """
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SEP>": 2,
        "<NL>": 3,
    }

    counts = Counter()
    for row in training_rows:
        counts.update(input_tokens(row, mode))

    # Deterministic ordering, including ties.
    ranked_tokens = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    for token, _ in ranked_tokens:
        if len(vocab) >= max_vocab:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


class TextDataset(Dataset):
    def __init__(self, rows, vocab, mode):
        self.rows = rows
        self.encoded = []
        self.labels = []

        for row in rows:
            tokens = input_tokens(row, mode)
            ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

            # Networks require at least one token.
            if not ids:
                ids = [vocab["<UNK>"]]

            self.encoded.append(torch.tensor(ids, dtype=torch.long))
            self.labels.append(LABEL_TO_ID[row["llm_name"]])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.encoded[index], self.labels[index]


def collate_batch(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    # CNN's largest filter has width five.
    padded_width = max(5, int(lengths.max().item()))

    tokens = torch.zeros(
        (len(sequences), padded_width),
        dtype=torch.long,
    )

    for index, sequence in enumerate(sequences):
        tokens[index, :len(sequence)] = sequence

    return (
        tokens,
        lengths,
        torch.tensor(labels, dtype=torch.long),
    )