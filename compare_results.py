"""Compare compatible runs using the proposed, capped performance rubric."""
import argparse
import json
from pathlib import Path


def compare(reference, candidate):
    if reference["status"] != "passed" or candidate["status"] != "passed":
        raise ValueError("Only passing runs can be compared")
    for key in ("training", "evaluation", "harness"):
        if reference["sha256"][key] != candidate["sha256"][key]:
            raise ValueError(f"Mismatched {key}")
    for key in ("vocab_size", "seed", "repeats", "max_model_bytes"):
        if reference["config"][key] != candidate["config"][key]:
            raise ValueError(f"Mismatched {key}")
    if reference["environment"] != candidate["environment"]:
        raise ValueError("Run both adapters in the same environment")
    ref = reference["aggregate"]
    final = candidate["aggregate"]
    if not 0 <= ref["tokens"] < ref["bytes"] or final["bytes"] != ref["bytes"]:
        raise ValueError("Reference must reduce token count below the byte baseline")
    if ref["encode_seconds"] <= 0 or final["encode_seconds"] <= 0:
        raise ValueError("Encoding times must be positive")
    efficiency = 6 * max(0, min(1, (ref["bytes"] - final["tokens"]) / (ref["bytes"] - ref["tokens"])))
    speed = 4 * min(1, ref["encode_seconds"] / final["encode_seconds"]) if final["tokens"] <= 1.10 * ref["tokens"] else 0
    return {
        "token_ratio_to_reference": final["tokens"] / ref["tokens"],
        "encoding_time_ratio_to_reference": final["encode_seconds"] / ref["encode_seconds"],
        "proposed_token_efficiency_points": efficiency,
        "proposed_encoding_speed_points": speed,
        "proposed_performance_points": efficiency + speed,
        "resource_limit_status": "Final time and RSS limits require instructor calibration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference, candidate = (json.loads(Path(path).read_text()) for path in (args.reference, args.candidate))
    print(json.dumps(compare(reference, candidate), indent=2))


if __name__ == "__main__":
    main()
