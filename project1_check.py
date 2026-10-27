"""One-command public benchmark, Python 3.11+, standard library only.

Run python3 student_adapter.py after implementing the template, or:
    python3 project1_check.py --adapter path/to/your_adapter.py
Exit 0 = PASS; exit 1 = FAIL; exit 2 = invalid command/setup.
PASS covers public contract checks and completing the benchmark within its
worker timeout. Performance is measured and scored separately, not a PASS gate.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def main(default_adapter=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=default_adapter or str(here / "student_adapter.py"))
    parser.add_argument("--train", default=str(here / "smoke-train.jsonl"))
    parser.add_argument("--evaluation", default=str(here / "smoke-eval.jsonl"))
    parser.add_argument("--results-dir", help="New directory for this run; must not already exist")
    args = parser.parse_args()
    if sys.version_info < (3, 11) or sys.platform not in ("linux", "darwin"):
        parser.error("Use Python 3.11+ on Linux/macOS, or Windows through WSL")
    from benchmark import run
    from compare_results import compare

    if args.results_dir:
        destination = Path(args.results_dir).resolve()
        try:
            destination.mkdir(exist_ok=False)
        except OSError as exc:
            parser.error(str(exc))
    else:
        destination = Path(tempfile.mkdtemp(prefix="benchmark-run-", dir=here))
    summary = {"status": "failed", "suite": "project1-local-v1", "results_dir": str(destination)}
    stage = "baseline"
    try:
        for stage, adapter in (("baseline", here / "baseline_bpe.py"), ("student", Path(args.adapter))):
            print(f"Running {stage} ...", flush=True)
            configuration = SimpleNamespace(
                adapter=str(adapter.resolve()), train=str(Path(args.train).resolve()),
                evaluation=str(Path(args.evaluation).resolve()),
                result=str(destination / f"{stage}-result.json"),
                vocab_size=512, seed=5051, repeats=5, timeout=120.0,
                max_model_bytes=1024 * 1024,
            )
            with (destination / f"{stage}.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log):
                    run(configuration)
        stage = "comparison"
        baseline = json.loads((destination / "baseline-result.json").read_text())
        student = json.loads((destination / "student-result.json").read_text())
        comparison = compare(baseline, student)
        (destination / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        summary.update(status="passed", performance_points=comparison["proposed_performance_points"])
        print("PASS: public benchmark checks passed")
        print("Performance is scored separately; PASS does not require 10/10.")
        print(f"Provisional performance points: {summary['performance_points']:.2f}/10")
        print("                    baseline          student")
        for name in ("tokens", "bytes_per_token", "encode_seconds", "decode_seconds", "peak_rss_bytes"):
            print(f"{name:20s} {baseline['aggregate'][name]:14.6g} {student['aggregate'][name]:14.6g}")
        print("Training/preparation timings, per-category results, and 128-token context capacity: see JSON reports.")
    except Exception as exc:
        summary.update(stage=stage, error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL: {stage}\n{summary['error']}")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results: {destination}")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
