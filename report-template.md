# Project 1: Tokenizer

<!-- Rename this file report.md. Write in English, UTF-8 plain-text Markdown.
Replace every prompt below with your own answer. Aim for 1,000–1,500 words,
excluding tables, code, and quoted AI excerpts. Submit only report.md and src/
in project1.zip. Generated JSON/model/result directories remain local. -->

## 1. How to run

- Python version, OS, CPU, and memory:
- Third-party dependencies and exact versions (write "standard library only" if applicable):
- Installation commands, starting from the extracted ZIP:
- Benchmark command: `cd src` then `python3 student_adapter.py`
- Commands to run your tests and controlled comparison:
- Reused code, libraries, baseline, and references with attribution:

## 2. Algorithms and decisions

Describe the two candidate approaches, your choice and objective, and at least
three connections to course concepts. Define input sizes and analyze runtime
and memory, including preprocessing. Trace one small example or counterexample.

## 3. Tests and correctness

Summarize edge cases and two tests that catch plausible bugs. Explain how you
obtained expected answers independently. Link to the relevant source files.

## 4. Benchmark results

Use values from one final local run. The generated JSON files are for looking
up measurements; copy their values into these tables rather than attaching them.

- Public benchmark status: PASS or FAIL (from `summary.json`).
- If FAIL: failing stage, error, and what works so far.
- Exact benchmark command and dataset filenames:
- Seed, vocabulary budget, repeats, timeout, and model-size limit (from `config`):
- Training/evaluation data and harness SHA-256 hashes (from `sha256`):

### Overall comparison

| Measurement | Baseline | Your system |
| --- | ---: | ---: |
| Evaluation bytes (`aggregate.bytes`) | | |
| Token count (`aggregate.tokens`) | | |
| Bytes per token (`aggregate.bytes_per_token`) | | |
| Training seconds (`training.seconds`) | | |
| Training peak RSS bytes (`training.peak_rss_bytes`) | | |
| Encoding seconds (`aggregate.encode_seconds`) | | |
| Decoding seconds (`aggregate.decode_seconds`) | | |
| Evaluation peak RSS bytes (`aggregate.peak_rss_bytes`) | | |
| Vocabulary size (`training.vocab_size`) | | |
| Model bytes (`training.model_bytes`) | | |

### By text category

Copy a row for each entry in `categories` from the two result files.

| Category | Baseline tokens | Your tokens | Baseline prepare s | Your prepare s | Baseline encode s | Your encode s | Baseline decode s | Your decode s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English | | | | | | | | |
| Chinese | | | | | | | | |
| Code | | | | | | | | |
| Mixed Unicode | | | | | | | | |
| Repetitive text | | | | | | | | |

- Provisional token-efficiency points, speed points, and total /10 (from `comparison.json`):
- Interpret a useful improvement and a tradeoff. Timing points are local estimates;
  the instructor measures both systems on the same grading machine.

### Controlled comparison

State the initial prediction, the one design choice changed, what was held
fixed, the command to reproduce the comparison, and a small results table.
Explain how the result affected your choice.

## 5. Tokenization in AI

Trace text -> bytes -> tokens -> IDs -> embedding lookup -> model. Show English,
Chinese, and code examples (use escaped bytes or hex where needed). Explain the
embedding-ID relationship and the effects of sequence length and vocabulary size.

Copy the `context_budget` values from each result file:

| 128-token budget | Baseline | Your system |
| --- | ---: | ---: |
| Complete records | | |
| Bytes retained | | |
| Code points retained | | |
| Tokens used | | |

What further language-model experiment would evaluate your segmentation policy?

## 6. Three learning records and AI use

Write approximately 100–150 words per record, within the overall report length.
Each covers your question/prediction, AI suggestion or reference, submitted
trace/test/measurement, and final decision.

### Algorithm choice

### Correctness

### Experimental prediction

List AI tools/models and their roles, or state that you did not use AI. Include
up to three short authentic prompt/response excerpts tied to the records,
including a relevant follow-up. Label retrospective explanations as such.
