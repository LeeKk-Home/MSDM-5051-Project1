"""Meaningful contract and integration tests for the benchmark prototype."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import baseline_bpe
from compare_results import compare

HERE = Path(__file__).resolve().parent


class BenchmarkTests(unittest.TestCase):
    def test_student_command_and_baseline_command(self):
        with TemporaryDirectory() as directory:
            for name, command, expected in (
                ("starter", [str(HERE / "student_adapter.py")], 1),
                ("baseline", [str(HERE / "project1_check.py"), "--adapter", str(HERE / "baseline_bpe.py")], 0),
            ):
                destination = Path(directory) / name
                result = subprocess.run([sys.executable, *command, "--results-dir", str(destination)],
                                        capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                summary = json.loads((destination / "summary.json").read_text())
                self.assertEqual(summary["status"], "passed" if expected == 0 else "failed")
                if expected:
                    self.assertIn("NotImplementedError", summary["error"])
                    self.assertIn("FAIL:", result.stdout)
                else:
                    self.assertIn("PASS: public benchmark checks passed", result.stdout)
                    self.assertTrue((destination / "comparison.json").exists())
                    self.assertTrue((destination / "student-result.model.json").exists())
                rerun = subprocess.run([sys.executable, *command, "--results-dir", str(destination)],
                                       capture_output=True, text=True, timeout=10)
                self.assertEqual(rerun.returncode, 2)
                self.assertEqual(json.loads((destination / "summary.json").read_text()), summary)

    def run_candidate(self, directory, source=None, timeout="10"):
        adapter = HERE / "baseline_bpe.py"
        if source is not None:
            adapter = Path(directory) / "adapter.py"
            adapter.write_text("import sys\nsys.path.insert(0, " + repr(str(HERE)) + ")\nfrom baseline_bpe import *\n" + source)
        result = Path(directory) / "result.json"
        completed = subprocess.run(
            [sys.executable, str(HERE / "benchmark.py"), "--adapter", str(adapter),
             "--result", str(result), "--timeout", timeout],
            capture_output=True, text=True, timeout=30,
        )
        report = json.loads(result.read_text()) if result.exists() else None
        return completed, report

    def test_baseline_roundtrip_and_weighted_training(self):
        model = baseline_bpe.train(["ab"] * 4 + ["ac"], 257, 5051)
        self.assertEqual(model["state"]["merges"][0][:2], [97, 98])
        self.assertEqual(baseline_bpe.merge((97, 97, 97), (97, 97), 256), (256, 97))
        runtime = baseline_bpe.prepare(model)
        for text in ("", "ab ab\n", "é e\u0301", "龘🧬"):
            self.assertEqual(baseline_bpe.decode(baseline_bpe.encode(text, runtime), runtime), text)

    def test_baseline_end_to_end_and_score(self):
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(report["status"], "passed")
            self.assertLess(report["aggregate"]["tokens"], report["aggregate"]["bytes"])
            self.assertEqual(len(report["categories"]), 5)
            self.assertEqual(compare(report, report)["proposed_performance_points"], 10)
            self.assertFalse((Path(directory) / "result.repeat-model.json").exists())

    def test_valid_different_algorithm_is_accepted(self):
        # Greedy longest matching uses the baseline vocabulary but different rules.
        source = '''
def prepare(model):
    return [bytes.fromhex(t) for t in model["tokens_hex"]], {}
def encode(text, runtime):
    tokens, _ = runtime
    data = text.encode("utf-8")
    ordered = sorted(range(len(tokens)), key=lambda i: (-len(tokens[i]), i))
    result = []
    position = 0
    while position < len(data):
        token_id = next(i for i in ordered if data.startswith(tokens[i], position))
        result.append(token_id)
        position += len(tokens[token_id])
    return result
'''
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory, source)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(report["status"], "passed")

    def test_rejects_fake_roundtrip(self):
        # Even an adapter whose decode returns the original text cannot bypass bytes.
        source = '''
_saved = ""
def encode(text, runtime):
    global _saved
    _saved = text
    return [0] if text else []
def decode(ids, runtime):
    if ids in ([-1], [len(runtime[0])], [True], [1.5], [255]):
        raise ValueError("invalid")
    return _saved
'''
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory, source)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Token bytes do not reconstruct", report["error"])

    def test_rejects_model_mutation(self):
        source = '''
_old_prepare = prepare
def prepare(model):
    model["changed"] = True
    return _old_prepare(model)
'''
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory, source)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("modified", report["error"])

    def test_rejects_nondeterministic_training(self):
        source = '''
_old_train = train
def train(records, vocab_size, seed):
    import time
    model = _old_train(records, vocab_size, seed)
    model["nonce"] = time.time_ns()
    return model
'''
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory, source)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not deterministic", report["error"])

    def test_timeout_reports_failure(self):
        source = '''
def train(records, vocab_size, seed):
    import time
    time.sleep(60)
'''
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory, source, timeout="0.3")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("TimeoutExpired", report["error"])

    def test_byte_only_gets_no_speed_reward(self):
        with TemporaryDirectory() as directory:
            completed, report = self.run_candidate(directory)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            candidate = json.loads(json.dumps(report))
            candidate["aggregate"]["tokens"] = candidate["aggregate"]["bytes"]
            candidate["aggregate"]["encode_seconds"] = 1e-9
            self.assertEqual(compare(report, candidate)["proposed_performance_points"], 0)


if __name__ == "__main__":
    unittest.main()
