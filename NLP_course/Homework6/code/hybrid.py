"""Build BM25 rankings and fuse them with dense retrieval."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from rag import ROOT, TOP_K, chunk_text, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "temp/student")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    corpus = read_jsonl(args.data / "corpus.jsonl")
    queries = read_jsonl(args.data / "query.jsonl")
    dense = {row["query_id"]: row["chunk_ids"] for row in read_jsonl(args.out / "retrieve.jsonl")}
    start = perf_counter()
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    docs = vectorizer.fit_transform(map(chunk_text, corpus))
    lengths = np.asarray(docs.sum(axis=1)).ravel()
    # BM25 k1=1.2, b=0.75, positive log1p IDF.
    length_penalty = 1.2 * (0.25 + 0.75 * lengths / lengths.mean())
    postings = docs.tocsc()
    df = np.diff(postings.indptr)
    idf = np.log1p((len(corpus) - df + 0.5) / (df + 0.5))
    built = perf_counter()
    sparse_runs, hybrid_runs = [], []
    for row, query_tokens in zip(queries, vectorizer.transform(q["query"] for q in queries)):
        qid = row["query_id"]
        scores = np.zeros(len(corpus))
        for token in query_tokens.indices:
            left, right = postings.indptr[token:token + 2]
            ids = postings.indices[left:right]
            counts = postings.data[left:right]
            scores[ids] += idf[token] * counts * 2.2 / (counts + length_penalty[ids])
        top = np.argsort(-scores, kind="stable")[:TOP_K]
        sparse = [corpus[i]["chunk_id"] for i in top]
        fusion = {}
        for ranking in [dense[qid], sparse]:
            for rank, chunk_id in enumerate(ranking, 1):
                fusion[chunk_id] = fusion.get(chunk_id, 0) + 1 / (60 + rank)
        merged = sorted(fusion, key=lambda chunk_id: (-fusion[chunk_id], chunk_id))[:TOP_K]
        sparse_runs.append({"query_id": qid, "chunk_ids": sparse})
        hybrid_runs.append({"query_id": qid, "chunk_ids": merged})
    write_jsonl(args.out / "bm25.jsonl", sparse_runs)
    write_jsonl(args.out / "hybrid.jsonl", hybrid_runs)
    write_json(args.out / "hybrid_timing.json", {"build_seconds": built - start,
               "query_and_fusion_seconds": perf_counter() - built, "queries": len(queries)})


if __name__ == "__main__":
    main()
