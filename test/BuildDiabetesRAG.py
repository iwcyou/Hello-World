# build_rag.py
"""
将文件夹中的 PDF 指南处理为 RAG 文档库 (JSONL) + 可选 FAISS 向量索引
Usage:
    python build_rag.py --pdf_dir /path/to/pdf_folder --out_dir ./out \
        --chunk_size 500 --chunk_overlap 100 --embed_model openai
Options:
    --pdf_dir       : 要处理的 PDF 文件夹（递归）
    --out_dir       : 输出目录（默认 ./out）
    --chunk_size    : 每个 chunk 的最大 token/word 估计 (默认 500)
    --chunk_overlap : 重叠量 (默认 100)
    --embed_model   : 'openai' 或 'sentence-transformers' 或 'none'
    --faiss         : 如果设置则构建 FAISS 向量索引
Notes:
    - 如果使用 openai 模式，请提前设置环境变量 OPENAI_API_KEY。
    - 安装依赖: pip install -r requirements.txt
    - requirements.txt 建议包含:
        - fitz       (PyMuPDF)
        - nltk
        - sentence-transformers
        - faiss-cpu
        - openai
        - tqdm
        - regex
"""

import os
import sys
import json
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz            # PyMuPDF
import nltk
import re
from tqdm import tqdm

# Optional imports (lazy)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

try:
    import faiss
except Exception:
    faiss = None

# Ensure required NLTK resources exist (compat with newer NLTK requiring punkt_tab)
def _ensure_nltk_resources():
    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),  # some NLTK versions require this
    ]
    for res_path, name in required:
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                nltk.download(name)
            except Exception as e:
                print(f"[WARN] Failed to download NLTK resource '{name}': {e}")

_ensure_nltk_resources()

# ---------------------------
# Utilities: text extraction
# ---------------------------
def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """
    返回 list of (page_number, text)
    """
    doc = fitz.open(path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text")
        # basic clean
        text = re.sub(r'\s+\n', '\n', text)
        text = text.strip()
        pages.append((i+1, text))
    doc.close()
    return pages

# ---------------------------
# Chunking (sentence + sliding window)
# ---------------------------
def chunk_text_sentences(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    基于句子分割，然后用近似词/token数量做滑窗chunk。
    chunk_size, overlap 单位为词(token/word 近似)。
    """
    sents = nltk.sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0

    def join_cur():
        return " ".join(cur).strip()

    for sent in sents:
        wcount = len(sent.split())
        if cur_len + wcount <= chunk_size:
            cur.append(sent)
            cur_len += wcount
        else:
            # flush current chunk
            if cur:
                chunks.append(join_cur())
            # start new chunk: with overlap from previous
            # we attempt to carry last `overlap` words
            carry = []
            carry_count = 0
            # get words from tail of cur
            tail_words = " ".join(cur).split() if cur else []
            if overlap > 0 and tail_words:
                # take up to overlap words
                take = tail_words[-overlap:]
                carry = [" ".join(take)]
                carry_count = len(take)
            cur = carry + [sent]
            cur_len = carry_count + wcount
    if cur:
        chunks.append(join_cur())
    # final clean: remove too short chunks
    chunks = [c for c in chunks if len(c.split()) > 20]
    return chunks

# ---------------------------
# Embedding helpers
# ---------------------------
class Embedder:
    def __init__(self, mode: str = "none", openai_model: str = "text-embedding-3-large", local_model_name: str = "all-MiniLM-L6-v2"):
        self.mode = mode
        if mode == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            self.model = openai_model
            # OpenAI key expected in env OPENAI_API_KEY
        elif mode == "sentence-transformers":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed")
            self.model = SentenceTransformer(local_model_name)
        else:
            self.model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.mode == "none":
            raise RuntimeError("Embedder mode is 'none' -- no embeddings will be produced.")
        if self.mode == "openai":
            # batch call naive
            out = []
            for t in texts:
                resp = openai.Embedding.create(model=self.model, input=t)
                vec = resp['data'][0]['embedding']
                out.append(vec)
            return out
        elif self.mode == "sentence-transformers":
            return self.model.encode(texts, show_progress_bar=False).tolist()
        else:
            raise RuntimeError("Unknown embedder mode")

# ---------------------------
# Main build pipeline
# ---------------------------
def build_rag_from_pdfs(pdf_dir: str, out_dir: str, chunk_size: int, chunk_overlap: int,
                        embed_model: str, build_faiss: bool):
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "rag_docs.jsonl"
    meta_path = out_dir / "metadata.jsonl"   # one-to-one with rag_docs
    embeddings_path = out_dir / "embeddings.npy"

    docs = []
    metas = []
    embeddings = []

    # init embedder if requested
    embedder = None
    if embed_model and embed_model != "none":
        embedder = Embedder(mode=embed_model)

    # walk pdfs
    pdf_paths = sorted([p for p in pdf_dir.rglob("*.pdf")])
    if not pdf_paths:
        print(f"No pdf found in {pdf_dir}")
        return

    for pdf in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            pages = extract_text_from_pdf(str(pdf))
        except Exception as e:
            print(f"Failed to extract {pdf}: {e}")
            continue

        # naive title from filename
        title = pdf.stem
        # accumulate page text to form doc-level segmentation or process page-by-page
        # We'll create chunks per page (concat small pages)
        page_texts = []
        buffer_text = ""
        buffer_start = None
        for (pgnum, txt) in pages:
            if not txt.strip():
                continue
            # if page is long, process immediately
            if len(txt.split()) > chunk_size * 2:
                # flush buffer
                if buffer_text:
                    page_texts.append((buffer_start, pgnum-1, buffer_text))
                    buffer_text = ""
                    buffer_start = None
                page_texts.append((pgnum, pgnum, txt))
            else:
                if buffer_text == "":
                    buffer_start = pgnum
                buffer_text += "\n" + txt
                # if buffer exceeds chunk_size*1.5 do flush
                if len(buffer_text.split()) > chunk_size * 1.5:
                    page_texts.append((buffer_start, pgnum, buffer_text))
                    buffer_text = ""
                    buffer_start = None
        if buffer_text:
            page_texts.append((buffer_start, pages[-1][0], buffer_text))

        # chunk each page_text
        for (pstart, pend, text) in page_texts:
            # normalize whitespace
            text = re.sub(r'\n{2,}', '\n', text).strip()
            chunks = chunk_text_sentences(text, chunk_size=chunk_size, overlap=chunk_overlap)
            for i, chunk in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                metadata = {
                    "source_pdf": str(pdf),
                    "title": title,
                    "page_start": pstart,
                    "page_end": pend,
                    "chunk_index": i,
                    "chunk_word_count": len(chunk.split())
                }
                docs.append({"id": doc_id, "text": chunk, "metadata": metadata})
                metas.append(metadata)

    # write docs to jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Wrote {len(docs)} chunks to {jsonl_path}")

    # embeddings + faiss
    if embedder is not None:
        batch = []
        batch_ids = []
        all_vecs = []
        B = 64
        for i in tqdm(range(0, len(docs), B), desc="Embedding docs"):
            texts = [d["text"] for d in docs[i:i+B]]
            vecs = embedder.embed(texts)
            all_vecs.extend(vecs)
        import numpy as np
        arr = np.array(all_vecs).astype('float32')
        np.save(out_dir / "embeddings.npy", arr)
        print(f"Saved embeddings to {out_dir/'embeddings.npy'}")
        if build_faiss:
            if faiss is None:
                print("faiss not installed; skipping FAISS index build")
            else:
                dim = arr.shape[1]
                index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors or IP if normalized
                # normalize for cosine
                faiss.normalize_L2(arr)
                index.add(arr)
                faiss.write_index(index, str(out_dir / "faiss_index.bin"))
                print(f"FAISS index written to {out_dir/'faiss_index.bin'}")

    print("Done.")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="/data/kunfeng/ADA-Diabetes_Guidelines")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--embed_model", type=str, default="sentence-transformers",
                        choices=["none", "openai", "sentence-transformers"])
    parser.add_argument("--faiss", action="store_true", help="Build FAISS index after embedding")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_rag_from_pdfs(args.pdf_dir, args.out_dir, args.chunk_size, args.chunk_overlap,
                        args.embed_model, args.faiss)
