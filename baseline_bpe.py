"""Readable byte-level BPE comparator for MSDM5051 Project 1.

Standard library only. Students may reuse and adapt this baseline with attribution.
"""
from collections import Counter


def merge(sequence, pair, output_id):
    result = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            result.append(output_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return tuple(result)


def train(records, vocab_size, seed):
    if vocab_size < 256:
        raise ValueError("The vocabulary budget must include all 256 bytes")
    tokens = [bytes([i]) for i in range(256)]
    ids = {token: i for i, token in enumerate(tokens)}
    sequences = Counter(tuple(text.encode("utf-8")) for text in records)
    rules = []
    while len(tokens) < vocab_size:
        counts = Counter()
        for sequence, frequency in sequences.items():
            for pair in zip(sequence, sequence[1:]):
                counts[pair] += frequency
        if not counts:
            break
        pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
        if counts[pair] < 2:
            break
        value = tokens[pair[0]] + tokens[pair[1]]
        if value not in ids:
            ids[value] = len(tokens)
            tokens.append(value)
        output_id = ids[value]
        rules.append([pair[0], pair[1], output_id])
        updated = Counter()
        for sequence, frequency in sequences.items():
            updated[merge(sequence, pair, output_id)] += frequency
        sequences = updated
    return {
        "format": "msdm5051-tokenizer-v1",
        "tokens_hex": [token.hex() for token in tokens],
        "state": {"merges": rules},
    }


def prepare(model):
    tokens = [bytes.fromhex(value) for value in model["tokens_hex"]]
    ranks = {}
    for rank, (left, right, output_id) in enumerate(model["state"]["merges"]):
        pair = (left, right)
        if pair in ranks or tokens[left] + tokens[right] != tokens[output_id]:
            raise ValueError("Invalid merge rule")
        ranks[pair] = (rank, output_id)
    return tokens, ranks


def encode(text, runtime):
    _, ranks = runtime
    sequence = tuple(text.encode("utf-8"))
    while len(sequence) > 1:
        candidates = (
            pair for pair in zip(sequence, sequence[1:]) if pair in ranks
        )
        pair = min(candidates, key=lambda value: ranks[value][0], default=None)
        if pair is None:
            break
        sequence = merge(sequence, pair, ranks[pair][1])
    return list(sequence)


def decode(ids, runtime):
    tokens, _ = runtime
    if any(type(i) is not int or not 0 <= i < len(tokens) for i in ids):
        raise ValueError("Invalid token ID")
    return b"".join(tokens[i] for i in ids).decode("utf-8", errors="strict")
