"""Common tokenizer benchmark. Linux/macOS, Python 3.11+, standard library.

Each training/evaluation worker is a fresh process. Timed passes get fresh
runtime state; validation is outside encoding/decoding timing intervals.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time

FORMAT = "msdm5051-tokenizer-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not all(isinstance(row.get(key), str) for key in ("id", "category", "text")):
            raise ValueError("Each record needs string id, category and text")
        if row["id"] in seen:
            raise ValueError("Record IDs must be unique")
        row["text"].encode("utf-8", errors="strict")
        seen.add(row["id"])
    if not rows:
        raise ValueError("Dataset has no records")
    return rows


def import_adapter(path):
    spec = importlib.util.spec_from_file_location("candidate_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(module)
    for name in ("train", "prepare", "encode", "decode"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Missing adapter function: {name}")
    return module


def validate_model(model, args, training=None):
    if not isinstance(model, dict) or model.get("format") != FORMAT:
        raise ValueError("Unsupported model format")
    values = model.get("tokens_hex")
    if not isinstance(values, list) or not 256 <= len(values) <= args.vocab_size:
        raise ValueError("Invalid vocabulary size")
    tokens = [bytes.fromhex(value) for value in values]
    if tokens[:256] != [bytes([i]) for i in range(256)]:
        raise ValueError("IDs 0-255 must be the base bytes")
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Tokens must be distinct and nonempty")
    if training is not None:
        data = [text.encode("utf-8") for text in training]
        if any(not any(token in record for record in data) for token in tokens[256:]):
            raise ValueError("Additional tokens must occur in training records")
    if len(canonical(model).encode("utf-8")) > args.max_model_bytes:
        raise ValueError("Model exceeds size limit")
    return tokens


def validate_encoding(text, ids, tokens):
    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(tokens) for i in ids):
        raise ValueError("Encoder must return a list of valid integer IDs")
    if b"".join(tokens[i] for i in ids) != text.encode("utf-8"):
        raise ValueError("Token bytes do not reconstruct input")


def validate_decoding(texts, decoded):
    if any(type(value) is not str or value != text for text, value in zip(texts, decoded)):
        raise ValueError("Decode round-trip failed")


def checked_runtime(adapter, model):
    value = copy.deepcopy(model)
    runtime = adapter.prepare(value)
    if canonical(value) != canonical(model):
        raise ValueError("prepare modified the model")
    return runtime, value


def error_cases(adapter, model):
    runtime, _ = checked_runtime(adapter, model)
    for ids in ([-1], [len(model["tokens_hex"])], [True], [1.5], [255]):
        try:
            adapter.decode(ids, runtime)
        except (ValueError, TypeError, UnicodeError):
            continue
        raise ValueError(f"Decoder accepted invalid IDs/UTF-8: {ids}")
    for text in ("", "a a\t\n", "aaaaa", "é", "e\u0301", "🧬龘"):
        ids = adapter.encode(text, runtime)
        validate_encoding(text, ids, [bytes.fromhex(v) for v in model["tokens_hex"]])
        validate_decoding([text], [adapter.decode(ids, runtime)])


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def worker(args):
    adapter = import_adapter(args.adapter)
    if args.worker == "train":
        texts = [row["text"] for row in read_records(args.train)]
        start = time.perf_counter()
        model = adapter.train(texts, args.vocab_size, args.seed)
        elapsed = time.perf_counter() - start
        validate_model(model, args, texts)
        payload = canonical(model)
        Path(args.model).write_text(payload, encoding="utf-8")
        return {"seconds": elapsed, "peak_rss_bytes": peak_rss(), "model_sha256": sha(args.model), "model_bytes": len(payload.encode("utf-8")), "vocab_size": len(model["tokens_hex"])}

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    tokens = validate_model(model, args)
    original = canonical(model)
    rows = [row for row in read_records(args.evaluation) if row["category"] == args.category]
    if not rows:
        raise ValueError("Empty evaluation category")
    texts = [row["text"] for row in rows]
    error_cases(adapter, model)
    runtime, _ = checked_runtime(adapter, model)
    warm = adapter.encode(texts[0], runtime)
    adapter.decode(warm, runtime)
    encoded_reference = None
    prepare_times, encode_times, decode_times = [], [], []
    for _ in range(args.repeats):
        model_copy = copy.deepcopy(model)
        start = time.perf_counter()
        runtime = adapter.prepare(model_copy)
        prepare_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        encoded = [adapter.encode(text, runtime) for text in texts]
        encode_times.append(time.perf_counter() - start)
        for text, ids in zip(texts, encoded):
            validate_encoding(text, ids, tokens)
        if encoded_reference is not None and encoded != encoded_reference:
            raise ValueError("Encoding is not deterministic")
        encoded_reference = copy.deepcopy(encoded)
        if canonical(model_copy) != original:
            raise ValueError("Encoding modified the frozen model")
        runtime, model_copy = checked_runtime(adapter, model)
        start = time.perf_counter()
        decoded = [adapter.decode(ids, runtime) for ids in encoded]
        decode_times.append(time.perf_counter() - start)
        validate_decoding(texts, decoded)
        if canonical(model_copy) != original:
            raise ValueError("Decoding modified the frozen model")
    runtime, model_copy = checked_runtime(adapter, model)
    reversed_ids = [adapter.encode(text, runtime) for text in reversed(texts)]
    if reversed_ids != list(reversed(encoded_reference)):
        raise ValueError("Encoding depends on record order")
    if canonical(model_copy) != original:
        raise ValueError("Encoding modified the frozen model")
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    token_count = sum(map(len, encoded_reference))
    seconds = statistics.median(encode_times)
    decode_seconds = statistics.median(decode_times)
    return {
        "category": args.category, "records": len(rows), "bytes": byte_count,
        "code_points": sum(map(len, texts)), "tokens": token_count,
        "bytes_per_token": byte_count / token_count if token_count else None,
        "prepare_seconds": statistics.median(prepare_times),
        "encode_seconds": seconds, "decode_seconds": decode_seconds,
        "encode_bytes_per_second": byte_count / seconds if seconds else None,
        "decode_bytes_per_second": byte_count / decode_seconds if decode_seconds else None,
        "encode_samples_seconds": encode_times, "decode_samples_seconds": decode_times,
        "prepare_samples_seconds": prepare_times, "peak_rss_bytes": peak_rss(),
        "record_results": [{"id": row["id"], "tokens": len(ids), "ids_sha256": hashlib.sha256(canonical(ids).encode()).hexdigest()} for row, ids in zip(rows, encoded_reference)],
    }


def run_worker(args, mode, model_path, category=None):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
               "--adapter", str(Path(args.adapter).resolve()), "--train", str(Path(args.train).resolve()),
               "--evaluation", str(Path(args.evaluation).resolve()), "--model", str(model_path),
               "--vocab-size", str(args.vocab_size), "--seed", str(args.seed),
               "--repeats", str(args.repeats), "--max-model-bytes", str(args.max_model_bytes)]
    if category is not None:
        command += ["--category", category]
    env = dict(os.environ, PYTHONHASHSEED=str(args.seed), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=env, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
        if process.returncode:
            raise RuntimeError(f"{mode} worker failed:\n{stderr[-6000:]}")
        return json.loads(stdout)
    finally:
        # Only kill this worker's process group, including descendants it started.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.communicate()


def run(args):
    result_path = Path(args.result).resolve()
    if not result_path.parent.exists():
        raise ValueError("Result parent directory must already exist")
    model_path = result_path.with_suffix(".model.json")
    repeat_model = result_path.with_suffix(".repeat-model.json")
    for path in (result_path, model_path, repeat_model):
        if path.exists():
            raise ValueError(f"Refusing to overwrite {path}; choose a new --result path")
    training = read_records(args.train)
    evaluation = read_records(args.evaluation)
    if {r["text"] for r in training} & {r["text"] for r in evaluation}:
        raise ValueError("Training and evaluation contain identical records")
    report = {
        "status": "running", "benchmark_version": 1,
        "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "config": {"seed": args.seed, "vocab_size": args.vocab_size, "repeats": args.repeats, "worker_timeout_seconds": args.timeout, "max_model_bytes": args.max_model_bytes},
        "sha256": {"harness": sha(__file__), "adapter": sha(args.adapter), "training": sha(args.train), "evaluation": sha(args.evaluation)},
    }
    try:
        report["training"] = run_worker(args, "train", model_path)
        repeated = run_worker(args, "train", repeat_model)
        if repeated["model_sha256"] != report["training"]["model_sha256"]:
            raise ValueError("Training is not deterministic for a fixed seed")
        report["categories"] = [run_worker(args, "eval", model_path, category) for category in sorted({row["category"] for row in evaluation})]
        categories = report["categories"]
        total_bytes = sum(row["bytes"] for row in categories)
        total_tokens = sum(row["tokens"] for row in categories)
        total_encode = sum(row["encode_seconds"] for row in categories)
        total_decode = sum(row["decode_seconds"] for row in categories)
        # Aggregate time is explicitly the sum of category medians.
        report["aggregate"] = {
            "bytes": total_bytes, "tokens": total_tokens,
            "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
            "encode_seconds": total_encode, "decode_seconds": total_decode,
            "encode_bytes_per_second": total_bytes / total_encode if total_encode else None,
            "decode_bytes_per_second": total_bytes / total_decode if total_decode else None,
            "peak_rss_bytes": max(row["peak_rss_bytes"] for row in categories),
        }
        per_record = {row["id"]: row["tokens"] for category in categories for row in category["record_results"]}
        context = {"budget_tokens": 128, "tokens": 0, "records": 0, "bytes": 0, "code_points": 0}
        for row in evaluation:
            count = per_record[row["id"]]
            if context["tokens"] + count > 128:
                break
            context["tokens"] += count
            context["records"] += 1
            context["bytes"] += len(row["text"].encode("utf-8"))
            context["code_points"] += len(row["text"])
        report["context_budget"] = context
        if sha(model_path) != report["training"]["model_sha256"]:
            raise ValueError("Evaluation modified the saved model file")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        repeat_model.unlink(missing_ok=True)
        result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result": str(result_path), "model": str(model_path), "aggregate": report["aggregate"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", default=str(directory / "smoke-train.jsonl"))
    parser.add_argument("--evaluation", default=str(directory / "smoke-eval.jsonl"))
    parser.add_argument("--result", default=str(directory / "local-result.json"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5051)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-model-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--worker", choices=("train", "eval"), help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--category", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.vocab_size < 256 or args.repeats < 1 or args.max_model_bytes < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Invalid vocabulary budget, repeats, size limit, timeout, or seed")
    if args.worker:
        with contextlib.redirect_stdout(sys.stderr):
            result = worker(args)
        print(canonical(result))
    else:
        run(args)


if __name__ == "__main__":
    main()
