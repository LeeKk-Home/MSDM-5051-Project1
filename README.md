# Project 1 Tokenizer Public Benchmark

This executable prototype compares different tokenizer algorithms under a common byte-token contract. It includes an intentionally simple BPE baseline, original synthetic smoke data, contract checks, and isolated training/evaluation workers. The student assignment is in [the project brief](../project1-tokenizer-brief.md); teaching decisions are in [the instructor design](../project1-tokenizer-design.zh.md).

## Run

Start with [START_HERE.md](START_HERE.md). The executable interface is `student_adapter.py`; after implementing its four functions, run `python3 student_adapter.py` for PASS/FAIL and a baseline comparison. Run `python3 project1_check.py --adapter baseline_bpe.py` to check the installation first. Each invocation saves a fresh results directory.

The lower-level commands below expose the individual measurement and comparison steps.

From this directory, using Python 3.11+ on Linux or macOS:

```bash
python3 benchmark.py --adapter baseline_bpe.py --result baseline-result.json
python3 benchmark.py --adapter /absolute/path/to/student_adapter.py --result student-result.json
python3 compare_results.py baseline-result.json student-result.json
python3 -m unittest discover -s . -p 'test_benchmark.py' -v
```

Each result is saved beside a corresponding `.model.json`. Existing result/model files are preserved: choose a new result filename for another run. No server, browser, network, or GPU is required. Workers and their process groups are terminated on completion and failure.

To regenerate the original synthetic fixtures, run `python3 make_smoke_data.py`. To use assignment data:

```bash
python3 benchmark.py --adapter student_adapter.py \
  --train /absolute/path/to/train.jsonl \
  --evaluation /absolute/path/to/eval.jsonl \
  --result student-final-data.json
```

Every JSONL record contains string fields `id`, `category`, and `text`. IDs must be unique within a file; exact duplicate text across the training/evaluation split is rejected. The public fixtures contain five categories and a long repetitive record. They exercise the harness and are not the final grading corpus.

## Adapter

Implement `train(records, vocab_size, seed)`, `prepare(model)`, `encode(text, runtime)`, and `decode(ids, runtime)`. The baseline shows the complete interface. `train` returns a JSON object:

```json
{
  "format": "msdm5051-tokenizer-v1",
  "tokens_hex": ["00", "01", "... all base bytes through ff, then learned tokens ..."],
  "state": {"algorithm_specific_parameters": "JSON-serializable learned state"}
}
```

The example illustrates the fields; its abbreviated token list is not a valid model. Tokens are nonempty, distinct byte strings. IDs 0–255 must equal the corresponding bytes. Every additional token must occur in a training record. The default limits are 512 vocabulary entries and 1 MiB of canonical JSON. Model size includes learned state. `prepare` may construct indexes without modifying the model. Query caches belong to the returned runtime and reset on each `prepare`; module-level caches across passes are outside the benchmark contract.

Use one process and one computational thread inside adapter operations. The harness sets common BLAS/OpenMP thread variables to one. Declare all dependencies and package versions. Libraries and alternate internal languages are allowed through the adapter; all learned state must be portable through the submitted JSON. The classroom task measures this process configuration consistently.

## Measurement and validation

Training runs in a fresh process, is timed once, and produces the saved JSON. A second fresh training process verifies that the same seed produces the same canonical model. Reported training time comes from the first run. Each evaluation category runs in another fresh process loaded from the saved model.

Evaluation performs contract checks and one warm-up, then measures five passes. Each pass starts with fresh runtime state. Preparation time is recorded separately; encoding and decoding validation occurs after their timed sections. Decoding gets its own newly prepared runtime. The reported preparation median covers encoding preparations. Timing includes iteration over the workload and collecting the outputs, equally for all adapters.

A correct encoder may return any sequence of IDs whose token bytes reconstruct the input. The benchmark also checks exact decode, empty and Unicode inputs, invalid IDs/UTF-8, repeated-call determinism, record-order independence, and model mutation. The reverse-order check runs outside timing. The tests include a valid non-BPE encoder to verify this algorithm independence. The benchmark is a common acceptance suite; students also supply tests specific to their chosen algorithms.

Results include category measurements, per-record counts and output hashes, aggregate counts, context-budget capacity, all timing samples, and data/harness/adapter hashes. Aggregate encoding/decoding time is the **sum of category medians**; aggregate bytes per token uses summed counts. Peak RSS is the whole worker process high-water mark, including interpreter, validation, and runtime structures. It is not an isolated allocation measurement of the tokenizer.

The default 120-second timeout applies to each worker, including validation and repetitions. Failure writes a JSON report with the error; the repeated-training temporary model is removed. The prototype records RSS but does not enforce a memory ceiling. Before grading, instructors must calibrate and publish final time/RSS limits with the final corpus. A timeout here is a protective run setting, not a calibrated grade boundary.

For official comparison, rerun both adapters on the same quiet machine with the same dependencies and benchmark configuration. Result hashes identify the adapter file; include the full source snapshot and dependency versions to identify imported implementation files. Comparisons check dataset/harness hashes, configuration, and environment metadata. Tiny workloads produce noisy timing; the final corpus and repetition protocol need a pilot before performance grading.

## Proposed performance component

`compare_results.py` computes the provisional 10-point component described in the brief. Six points measure token reduction relative to byte-only encoding, capped at the BPE baseline. Four measure encoding speed relative to the baseline, available when token count is at most 110% of baseline. A very fast byte-only implementation therefore receives no performance points.

This script prints proposed scores and ratios. Final time/RSS eligibility is an instructor grading step after calibration; the other 90 project points cover functionality, algorithm decisions, benchmark interpretation, tests, learning records, and the understanding check. Matching the baseline is sufficient for full performance points.
