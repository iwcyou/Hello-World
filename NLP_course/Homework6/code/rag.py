"""Build a FAISS index, rerank its results and generate grounded examples."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from time import perf_counter

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RETRIEVER = ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
RERANKER = ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")
GENERATOR = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 50
EVIDENCE_K = 3
DENSE_LENGTH = 512
PAIR_LENGTH = 384
EVIDENCE_LENGTH = 420
MAX_NEW_TOKENS = 160


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def chunk_text(chunk):
    return chunk["text"]


@dataclass
class Runtime:
    device: torch.device
    batch_size: int

    @property
    def dtype(self):
        if self.device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def tick(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    def model(self, factory, spec, local_dir=None):
        repo, revision = spec
        source = str(local_dir) if local_dir else repo
        options = {} if local_dir else {"revision": revision}
        tokenizer = AutoTokenizer.from_pretrained(source, **options)
        model = factory.from_pretrained(source, dtype=self.dtype, **options)
        return tokenizer, model.to(self.device).eval()


@torch.inference_mode()
def embed(texts, tokenizer, model, runtime, max_length):
    batches = []
    for start in range(0, len(texts), runtime.batch_size):
        batch = tokenizer(texts[start:start + runtime.batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(runtime.device)
        cls = model(**batch).last_hidden_state[:, 0].float()
        batches.append(torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)


def build(data, work, out, runtime):
    path = data / "corpus.jsonl"
    corpus = read_jsonl(path)
    chunk_ids = [row["chunk_id"] for row in corpus]
    if len(corpus) < TOP_K or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Corpus must contain at least 50 distinct chunks.")
    tokenizer, model = runtime.model(AutoModel, RETRIEVER)
    start = runtime.tick()
    vectors = embed([chunk_text(row) for row in corpus], tokenizer, model, runtime, DENSE_LENGTH)
    indexed = runtime.tick()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(work / "corpus.faiss"))
    write_json(work / "chunk_ids.json", chunk_ids)
    write_json(work / "index_config.json", {"corpus_sha256": sha256(path), "retriever": list(RETRIEVER),
                                           "max_tokens": DENSE_LENGTH})
    write_json(out / "index_metrics.json", {"chunks": index.ntotal, "dimensions": index.d,
               "embedding_seconds": indexed - start, "build_seconds": runtime.tick() - start,
               "faiss_bytes": (work / "corpus.faiss").stat().st_size,
               "mapping_bytes": (work / "chunk_ids.json").stat().st_size,
               "device": torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else "CPU",
               "precision": str(runtime.dtype), "batch_size": runtime.batch_size})


@torch.inference_mode()
def rerank_pairs(queries, pairs, tokenizer, model, runtime):
    logits = []
    for start in range(0, len(pairs), runtime.batch_size):
        batch = pairs[start:start + runtime.batch_size]
        features = tokenizer([queries[q]["query"] for q, _ in batch], [text for _, text in batch],
                             padding=True, truncation="longest_first", max_length=PAIR_LENGTH,
                             return_tensors="pt").to(runtime.device)
        logits.append(model(**features).logits.flatten().float().cpu().numpy())
    return np.concatenate(logits)


def retrieve(data, work, out, runtime):
    path = data / "corpus.jsonl"
    config = json.loads((work / "index_config.json").read_text())
    if config != {"corpus_sha256": sha256(path), "retriever": list(RETRIEVER), "max_tokens": DENSE_LENGTH}:
        raise ValueError("Index configuration does not match this corpus and retriever.")
    corpus = {row["chunk_id"]: chunk_text(row) for row in read_jsonl(path)}
    queries = read_jsonl(data / "query.jsonl")
    chunk_ids = json.loads((work / "chunk_ids.json").read_text())
    index = faiss.read_index(str(work / "corpus.faiss"))
    if len(chunk_ids) != index.ntotal or set(chunk_ids) != corpus.keys():
        raise ValueError("Index-to-chunk mapping is inconsistent.")
    tok, model = runtime.model(AutoModel, RETRIEVER)
    start = runtime.tick()
    vectors = embed([QUERY_PREFIX + row["query"] for row in queries], tok, model, runtime, DENSE_LENGTH)
    encoded = runtime.tick()
    _, hits = index.search(vectors, TOP_K)
    retrieved = runtime.tick()
    candidates = [[chunk_ids[int(i)] for i in indices] for indices in hits]
    write_run(out / "retrieve.jsonl", queries, candidates)
    write_json(out / "retrieve_timing.json", {"queries": len(queries),
               "query_embedding_seconds": encoded - start, "faiss_search_seconds": retrieved - encoded,
               "retrieve_seconds": retrieved - start})


def write_run(path, queries, ranking):
    write_jsonl(path, ({"query_id": row["query_id"], "chunk_ids": ids}
                       for row, ids in zip(queries, ranking)))


def rerank(data, out, runtime, stage):
    corpus = {row["chunk_id"]: chunk_text(row) for row in read_jsonl(data / "corpus.jsonl")}
    queries = read_jsonl(data / "query.jsonl")
    rows = read_jsonl(out / f"{stage}.jsonl")
    before = {row["query_id"]: row["chunk_ids"] for row in rows}
    if len(before) != len(rows) or before.keys() != {row["query_id"] for row in queries}:
        raise ValueError("Candidate run must cover each query exactly once.")
    candidates = [before[row["query_id"]] for row in queries]
    if any(len(ids) != TOP_K or len(set(ids)) != TOP_K or not set(ids) <= corpus.keys()
           for ids in candidates):
        raise ValueError("Candidate run must contain 50 distinct known chunks per query.")
    tok, model = runtime.model(AutoModelForSequenceClassification, RERANKER)
    start_rerank = runtime.tick()
    pairs = [(i, corpus[chunk_id]) for i, ids in enumerate(candidates) for chunk_id in ids]
    logits = rerank_pairs(queries, pairs, tok, model, runtime).reshape(len(queries), TOP_K)
    orders = np.argsort(-logits, axis=1, kind="stable")
    reranked = [[ids[int(j)] for j in order] for ids, order in zip(candidates, orders)]
    rerank_time = runtime.tick() - start_rerank
    write_run(out / f"{stage}_rerank.jsonl", queries, reranked)
    write_json(out / f"{stage}_rerank_timing.json", {"queries": len(queries),
               "rerank_seconds": rerank_time, "pair_count": len(pairs)})


@torch.inference_mode()
def generate(data, out, runtime, local_generator, ranking_stage, evidence_k):
    corpus = {row["chunk_id"]: row for row in read_jsonl(data / "corpus.jsonl")}
    queries = read_jsonl(data / "query.jsonl")
    queries.sort(key=lambda row: row["query_id"])
    rankings = {row["query_id"]: row["chunk_ids"] for row in read_jsonl(out / f"{ranking_stage}.jsonl")}
    sample = random.Random(42).sample(queries, 10)
    tok, model = runtime.model(AutoModelForCausalLM, GENERATOR, local_generator)
    results = []
    started = runtime.tick()
    for row in sample:
        ids = rankings[row["query_id"]][:evidence_k]
        evidence = []
        for position, chunk_id in enumerate(ids, 1):
            text = chunk_text(corpus[chunk_id])
            token_ids = tok.encode(text, add_special_tokens=False)
            evidence.append({"ref": f"E{position}", "chunk_id": chunk_id,
                             "text": tok.decode(token_ids[:EVIDENCE_LENGTH]),
                             "truncated": len(token_ids) > EVIDENCE_LENGTH})
        passages = "\n\n".join(f"[{entry['ref']}] {entry['text']}" for entry in evidence)
        messages = [
            {"role": "system", "content": "You check descriptions of scientific figures. Use only the supplied evidence. "
             "Write one or two short English conclusions. Cite each conclusion using its bracketed evidence ID "
             "and quote the supporting words exactly. "
             "If the evidence is insufficient, say what is missing. Do not treat the claim itself as evidence."},
            {"role": "user", "content": f"Figure description to check:\n{row['query']}\n\nEvidence:\n{passages}"},
        ]
        inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                         return_dict=True, return_tensors="pt").to(runtime.device)
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        answer = tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        refs = {e["ref"]: e["chunk_id"] for e in evidence}
        cited = re.findall(r"\[(E\d+)\]", answer)
        def resolve(match):
            ref = match.group(1)
            return f"[{refs[ref]}]" if ref in refs else match.group()
        results.append({**row, "answer_raw": answer, "answer": re.sub(r"\[(E\d+)\]", resolve, answer),
                        "citation_chunk_ids": [refs[c] for c in cited if c in refs],
                        "invalid_citations": [c for c in cited if c not in refs], "evidence": evidence})
        print(row["query_id"], answer[:120], flush=True)
    write_jsonl(out / f"{ranking_stage}_generation.jsonl", results)
    write_json(out / f"{ranking_stage}_generation_timing.json", {"cases": len(results), "seconds": runtime.tick() - started,
               "generator": str(local_generator) if local_generator else list(GENERATOR),
               "seed": 42, "top_k": evidence_k, "ranking_stage": ranking_stage, "evidence_token_limit": EVIDENCE_LENGTH,
               "max_new_tokens": MAX_NEW_TOKENS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "retrieve", "rerank", "generate"])
    parser.add_argument("--data", type=Path, default=ROOT / "temp/student")
    parser.add_argument("--work", type=Path, default=ROOT / "temp/rag")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--generator-dir", type=Path)
    parser.add_argument("--candidate-stage", choices=["retrieve", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--ranking-stage", choices=["hybrid", "hybrid_rerank", "retrieve_rerank", "bm25_rerank"],
                        default="hybrid_rerank")
    parser.add_argument("--evidence-k", type=int, choices=[1, 2, 3, 4, 5], default=EVIDENCE_K)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    runtime = Runtime(torch.device(args.device), args.batch_size)
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    if args.stage == "build":
        build(args.data, args.work, args.out, runtime)
    elif args.stage == "generate":
        generate(args.data, args.out, runtime, args.generator_dir, args.ranking_stage, args.evidence_k)
    elif args.stage == "retrieve":
        retrieve(args.data, args.work, args.out, runtime)
    else:
        rerank(args.data, args.out, runtime, args.candidate_stage)
    if runtime.device.type == "cuda":
        memory = {"peak_allocated_mib": torch.cuda.max_memory_allocated(runtime.device) / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(runtime.device) / 2**20}
        write_json(args.out / f"{args.stage}_memory.json", memory)
        print(args.stage, memory, flush=True)


if __name__ == "__main__":
    main()
