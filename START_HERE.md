# Project 1: Tokenizer — executable interface and public benchmark

Use Python 3.11+ on Linux/macOS, or Windows through WSL. Only the Python standard library is required for the supplied files. Extract the entire ZIP and keep the files together.

1. Check the supplied baseline and your environment:

   ```bash
   python3 project1_check.py --adapter baseline_bpe.py
   ```

   Expected: `PASS: public benchmark checks passed`, a performance table, a provisional score out of 10, and a `Results:` directory path. Timing and speed points vary between runs.

2. Implement `train`, `prepare`, `encode`, and `decode` in `student_adapter.py`. Each function has a signature and contract in its docstring. You may import your own modules. Keep the launcher at the bottom of the file.

3. Execute your own file:

   ```bash
   python3 student_adapter.py
   ```

   The untouched template produces `FAIL: student` with `NotImplementedError`. After implementation, the required success message is:

   ```text
   Running baseline ...
   Running student ...
   PASS: public benchmark checks passed
   Performance is scored separately; PASS does not require 10/10.
   Provisional performance points: <measured score>/10
   ... baseline and student metrics ...
   Results: <path to this run>
   ```

Each execution creates a new `benchmark-run-*` directory beside the checker, preserving earlier runs. On success it contains `summary.json`, `baseline-result.json`, `student-result.json`, the two corresponding `.model.json` files, `comparison.json`, and logs. Use the generated values to fill the benchmark tables in `report.md`; keep this generated directory locally. `summary.json` contains `"status": "passed"` or `"status": "failed"`; the process exits with 0 or 1 respectively (2 for invalid command/setup). On failure read `error` and `stage` in `summary.json`; a student result report is also saved if the harness reached its worker stage.

## What PASS means

All public checks must succeed: base-byte vocabulary and model-size contract; training-token provenance; deterministic training; fresh-process model reload; valid token IDs and exact input-byte reconstruction; decoding, Unicode/whitespace/empty cases; rejected invalid IDs/UTF-8; deterministic and record-order-independent encoding; unchanged saved model; completion of every training/evaluation worker within 120 seconds. The budget is 512 tokens, 1 MiB model, seed 5051, and five timed passes per category. Read README.md for the measurement protocol.

PASS is the public correctness/completion requirement. Compression and speed earn a separate provisional performance score using the assignment formula. A correct byte-only implementation can pass while earning zero performance points. There is no extra secret speed threshold in this local checker. Memory is reported; this version does not enforce an RSS ceiling. The attached data is the public smoke suite; assignment-sized data and final grading resource limits will be announced before release.

Do not edit `project1_check.py`, `benchmark.py`, `compare_results.py`, the baseline, or supplied JSONL data to obtain PASS. Choose any implementation meeting the interface; token IDs need not match the baseline's output. Add your own tests in separate files.

For later released data, use the same entry point:

```bash
python3 student_adapter.py --train train.jsonl --evaluation eval.jsonl
```

The reports record the actual data hashes. Run locally, obtain PASS, and submit `project1.zip` containing only `src/` (the runnable program with its required files) and `report.md` to Canvas. Uploading to Canvas does not execute tests. You may resubmit a complete ZIP during the submission period; the last complete submission received by the deadline is graded. If a failure remains at the deadline, describe the failed check and error in `report.md` for partial credit. The instructor downloads submissions, reruns the course-provided checker on a common local machine, and returns grades and feedback through Canvas. Local testing needs no network access or GPU.


## Exactly what to submit

Submit one `project1.zip` with two items: `src/` and `report.md`. Put the working Python files, tests/comparison code, and required supplied JSONL data under `src/`. Exclude generated results/models, caches, and virtual environments. After extraction and installing dependencies documented in the report, `cd src` followed by `python3 student_adapter.py` must run the benchmark.

Copy `report-template.md` to the ZIP root as `report.md` and fill it in as UTF-8 plain-text Markdown in English. Include setup instructions and attribution there. No separate README, raw JSON reports, generated models, PDF, screenshots, or full AI chat export are required. The report contains the measured results and three learning records.
