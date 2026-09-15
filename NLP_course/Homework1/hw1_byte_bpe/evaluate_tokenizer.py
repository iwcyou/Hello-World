"""Evaluate round-trip correctness and compression on the held-out split."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from byte_bpe_tokenizer import ByteBPETokenizer
from train_tokenizer import read_document, split_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("../gene_related_papers_1000"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("outputs/tokenizer"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer.from_pretrained(args.tokenizer_dir)
    _, validation_files = split_files(args.corpus_dir, args.train_ratio)
    started = time.perf_counter()
    total_bytes = total_chars = total_tokens = exact = 0
    failures: list[str] = []
    for path in validation_files:
        text = read_document(path)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        if decoded == text:
            exact += 1
        else:
            failures.append(path.name)
    elapsed = time.perf_counter() - started
    result = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "validation_documents": len(validation_files),
        "exact_roundtrip_documents": exact,
        "roundtrip_accuracy": exact / len(validation_files) if validation_files else 0.0,
        "failed_files": failures,
        "characters": total_chars,
        "utf8_bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "tokens_per_character": total_tokens / total_chars if total_chars else 0.0,
        "byte_sequence_reduction": 1 - total_tokens / total_bytes if total_bytes else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "documents_per_second": len(validation_files) / elapsed if elapsed else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Round-trip validation failed")


if __name__ == "__main__":
    main()
