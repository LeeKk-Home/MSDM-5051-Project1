"""Project 1: Tokenizer — implement these four functions.

Download and extract project1-tokenizer-starter.zip first. Keep this file beside
project1_check.py. Use Python 3.11+ on Linux, macOS, or Windows through WSL.
Run: python3 student_adapter.py
The untouched template reports FAIL (NotImplementedError). Implement the four
functions, rerun, and inspect the printed PASS/FAIL and saved JSON reports.
You may put your implementation in other modules and import it here.
"""


def train(records: list[str], vocab_size: int, seed: int) -> dict:
    """Learn only from records; deterministic for a fixed seed.

    Return {"format": "msdm5051-tokenizer-v1", "tokens_hex": [...],
            "state": ... (optional JSON-compatible learned state)}.
    IDs 0..255 are bytes 00..ff. All tokens are distinct, nonempty bytes;
    learned tokens occur in training records. Respect vocab_size (default 512)
    and the 1 MiB canonical JSON limit. Persist all learned state here.
    """
    raise NotImplementedError("Implement train(records, vocab_size, seed)")


def prepare(model: dict) -> object:
    """Create fresh indexes/caches without mutating the saved model."""
    raise NotImplementedError("Implement prepare(model)")


def encode(text: str, runtime: object) -> list[int]:
    """Return valid IDs whose concatenated bytes equal text.encode('utf-8').

    Empty text -> []; preserve whitespace and normalization. Results must be
    deterministic and independent of record order. Do not update the model.
    """
    raise NotImplementedError("Implement encode(text, runtime)")


def decode(ids: list[int], runtime: object) -> str:
    """Concatenate token bytes, then decode strict UTF-8.

    Reject negative/out-of-range IDs, bool/non-integer IDs, and malformed UTF-8
    using ValueError, TypeError, or UnicodeError. Empty IDs -> empty string.
    """
    raise NotImplementedError("Implement decode(ids, runtime)")


if __name__ == "__main__":
    from project1_check import main

    raise SystemExit(main(default_adapter=__file__))
