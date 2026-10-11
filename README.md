# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 individual project  
Author and project lead: Arjun Gupta

## Overview

This project classifies responses from three LLM families using a
convolutional neural network (CNN), a bidirectional LSTM recurrent neural
network (RNN), and a TF-IDF logistic regression baseline.

Experiments investigate:
- RQ1: Can response text identify its generating LLM?
- RQ2: Does using the prompt improve classification?
- RQ3: Do learned patterns generalize to writing tasks excluded from training?
- RQ4: What linguistic and structural characteristics distinguish the models?

The findings concern the historical model versions and sampled dataset below,
rather than all versions of these LLM families.

## Dataset and provenance

Source: [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

Dataset paper: https://arxiv.org/abs/2309.11998

Dataset revision used:
`200748d9d3cddcc9d782887541057aca0b18c5da`

| Family label | Source model |
|---|---|
| GPT | gpt-4 |
| Claude | claude-1 |
| Llama | llama-2-13b-chat |

Each example contains `llm_name`, `llm_input`, and `llm_output`, with additional
identifiers and metadata used for reproducibility.

Collection scanned the first 500,000 source records. Eligible candidate pools
were capped at 1,500 examples per family; GPT had 1,285 eligible candidates.
Using seed 42, collection selected 1,000 examples per family.

Collection selected first-turn user/assistant pairs and applied language,
length, redaction, moderation, model-identity, email, and duplicate filters.
A subsequent language-detection step excluded 63 examples, retaining 2,937.

| Split | Claude | GPT | Llama | Total |
|---|---:|---:|---:|---:|
| Training | 670 | 693 | 693 | 2,056 |
| Validation | 95 | 99 | 99 | 293 |
| Test | 192 | 198 | 198 | 588 |
| Total | 957 | 990 | 990 | 2,937 |

Examples sharing an exact normalized prompt, exact normalized response, or
conversation identifier were connected into groups. A seeded
StratifiedGroupKFold procedure assigned approximately 70%/10%/20% to
training/validation/test while keeping groups together.

Domain labels were assigned using prompt keyword rules. These labels are
preliminary rather than manually verified ground truth.

Conversation text, review previews, trained checkpoints, and serialized
pipelines are excluded from this repository. Obtain source data through
Hugging Face and follow the dataset's access requirements and terms.
Automated filtering does not guarantee removal of every sensitive or
inappropriate record. The preparation summary records manual review as
incomplete.

## Models and training

Tokenization preserves case, punctuation, and newline markers. Vocabulary
construction uses training examples only.

Text budgets:
- Input only: first 127 prompt tokens.
- Output only: first 256 response tokens.
- Combined: up to 127 prompt tokens, a separator, and 256 response tokens.

CNN:
- Trainable 100-dimensional embeddings.
- Convolution widths 3, 4, and 5, with 64 filters each.
- ReLU and masked max pooling.
- Dropout 0.4 and a three-class output layer.

RNN:
- Trainable 100-dimensional embeddings.
- One bidirectional LSTM layer, hidden size 64 per direction.
- Packed sequences to handle padding.
- Concatenated final directional states.
- Dropout 0.4 and a three-class output layer.

Neural training uses Adam, learning rate 0.001, batch size 32,
cross-entropy loss, gradient clipping at 1.0, and seed 42.
Training runs for at most 15 epochs with early-stopping patience 3.
Checkpoints are selected by validation macro-F1. Reported runs used CPU.

The baseline uses training-fitted TF-IDF unigram/bigram features and
logistic regression. It shares the neural tokenizer and text budgets.

## Test results

Checkpoints and experiment settings were selected before final test evaluation.

| Model | Text mode | Accuracy | Macro-F1 |
|---|---|---:|---:|
| CNN | Input | 0.4252 | 0.4231 |
| CNN | Output | 0.9133 | 0.9137 |
| CNN | Input + output | 0.9065 | 0.9062 |
| LSTM | Input | 0.4014 | 0.4005 |
| LSTM | Output | 0.8520 | 0.8518 |
| LSTM | Input + output | 0.7228 | 0.7255 |
| TF-IDF + logistic regression | Input | 0.4150 | 0.4132 |
| TF-IDF + logistic regression | Output | 0.8793 | 0.8791 |
| TF-IDF + logistic regression | Input + output | 0.8793 | 0.8795 |

Uniform random prediction has expected accuracy 33.33%.
The test-set majority-class accuracy is 33.67%.

RQ1: Both neural architectures identify the sampled generators effectively
from responses.

RQ2: Input-only classification suggests associations between prompts and
model labels in the source dataset. User selection and task distributions
may explain these associations. Adding prompts did not improve neural
test accuracy in these experiments.

## Cross-domain experiment: RQ3

Additional output-only CNN and LSTM models excluded writing-labeled groups
from training and validation. They were evaluated on the same test subsets
as the original models.

The writing test subset contains 30 examples: 11 Claude, 5 GPT, and 14 Llama.

| Model | Training condition | Writing accuracy | Writing macro-F1 |
|---|---|---:|---:|
| CNN | Original training set | 0.8667 | 0.8456 |
| CNN | Writing excluded | 0.7667 | 0.7388 |
| LSTM | Original training set | 0.8667 | 0.8489 |
| LSTM | Writing excluded | 0.8333 | 0.7827 |

These results suggest some transfer to an excluded domain, with reduced
writing performance. The small test subset, heuristic labels, reduced
training-set size, and training variability limit the conclusion.

## Style analysis: RQ4

Training-set analysis examines response length, word length, lexical diversity,
line breaks, bullet and numbered lists, code fences, and punctuation.

A standardized style-feature logistic regression classifier is compared
with the response TF-IDF baseline and a retrained lexical ablation that
removes punctuation and newline tokens from the original response token span.

| Experiment | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Original response TF-IDF | 0.8793 | 0.8791 |
| TF-IDF without punctuation or newline tokens | 0.8724 | 0.8726 |
| Style features only | 0.6361 | 0.6340 |

Style features contain predictive information. Removing formatting-related
tokens produces a small observed decrease for TF-IDF. These experiments
do not directly attribute CNN or LSTM decisions or establish causal effects.

The style-only classifier uses full responses; TF-IDF uses truncated responses.

## Running the project

The development environment used Windows, PowerShell, and Python 3.11.9.
Exact package versions are recorded in `requirements.txt`.

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Authenticate with Hugging Face and satisfy any dataset access requirements:

```powershell
.\.venv\Scripts\hf.exe auth login
```

Before recollecting data, ensure `collect_data.py` uses the dataset revision
recorded above. Changing the revision may change the examples and results.

Run data collection and preparation:

```powershell
.\.venv\Scripts\python.exe src\collect_data.py
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\models.py
```

Train the six main neural experiments:

```powershell
.\.venv\Scripts\python.exe src\train.py --model cnn --mode input
.\.venv\Scripts\python.exe src\train.py --model cnn --mode output
.\.venv\Scripts\python.exe src\train.py --model cnn --mode both
.\.venv\Scripts\python.exe src\train.py --model lstm --mode input
.\.venv\Scripts\python.exe src\train.py --model lstm --mode output
.\.venv\Scripts\python.exe src\train.py --model lstm --mode both
```

Run the additional experiments, then evaluate:

```powershell
.\.venv\Scripts\python.exe src\baseline.py
.\.venv\Scripts\python.exe src\cross_domain.py --model cnn
.\.venv\Scripts\python.exe src\cross_domain.py --model lstm
.\.venv\Scripts\python.exe src\analyze_style.py
.\.venv\Scripts\python.exe src\evaluate.py
```

Several scripts refuse to overwrite completed results. To reproduce all
experiments, use a separate working copy and move the included `results`
directory to a backup location before running the pipeline.
The excluded model files must be regenerated before evaluation.

## Repository contents

- `src/`: data collection, preparation, models, training, and evaluation.
- `results/`: provenance, split assignments, configurations, training histories,
  plots, predictions without conversation text, and evaluation metrics.
- `requirements.txt`: development package versions.
- Project PDF: added as the final submission report.

## Limitations

- One historical model version represents each family.
- Prompts differ across generators; model labels can correlate with task mix.
- Source-prefix scanning and candidate caps introduce sampling limitations.
- Language and domain detection are imperfect.
- Exact duplicate grouping does not guarantee semantic near-duplicate removal.
- Only one training seed was used; comparisons are descriptive.
- The cross-domain writing test contains only 30 examples.
- Style associations are not causal explanations.
- Results do not establish performance on current models or new data sources.

## Assistance disclosure

LLM assistance was used for project guidance, code development, debugging,
and interpretation. The author is responsible for understanding, verifying,
and explaining the submitted work.