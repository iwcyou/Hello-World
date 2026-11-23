"""Evaluate JSONL runs against dev or teacher-held test judgments."""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def keyed(path, key):
    rows = {}
    for row in read_jsonl(path):
        identifier = row[key]
        if not isinstance(identifier, str) or identifier in rows:
            raise ValueError(f"{path}: invalid or repeated {key}: {identifier}")
        rows[identifier] = row
    return rows


def score(corpus, qrels, run):
    if not qrels or run.keys() != qrels.keys():
        raise ValueError("Run must include exactly one row for each judged query.")
    cutoffs = (1, 2, 5, 10, 50)
    counts = {k: [0, 0, 0] for k in cutoffs}
    mrr = 0.0
    for qid, judgment in qrels.items():
        gold = set(judgment["relevant_chunk_ids"])
        ranked = run[qid]["chunk_ids"]
        if (len(judgment["relevant_chunk_ids"]) != 2 or len(gold) != 2 or not gold <= corpus.keys()
                or {corpus[chunk_id]["field"] for chunk_id in gold} != {"caption", "context"}):
            raise ValueError(f"{qid}: invalid judgments")
        if (not isinstance(ranked, list) or len(ranked) != 50
                or not all(isinstance(doc, str) for doc in ranked)
                or len(set(ranked)) != len(ranked) or not set(ranked) <= corpus.keys()):
            raise ValueError(f"{qid}: expected 50 distinct known chunk_ids")
        for k in cutoffs:
            hits = len(gold.intersection(ranked[:k]))
            counts[k][hits] += 1
        mrr += next((1 / rank for rank, doc in enumerate(ranked[:10], 1) if doc in gold), 0)
    n = len(qrels)
    metrics = {"queries": n, "MRR@10": mrr / n}
    for k, (zero, one, both) in counts.items():
        metrics[f"Recall@{k}"] = (one + 2 * both) / (2 * n)
        metrics[f"ZeroHit@{k}"] = zero / n
        metrics[f"OneHit@{k}"] = one / n
        metrics[f"BothHit@{k}"] = both / n
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("qrels", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = score(keyed(args.corpus, "chunk_id"), keyed(args.qrels, "query_id"), keyed(args.run, "query_id"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
