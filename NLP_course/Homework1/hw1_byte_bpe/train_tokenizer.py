"""Train a byte-level BPE tokenizer on the provided JSON corpus."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from byte_bpe_tokenizer import ByteBPETokenizer, pretokenize


# 只抽取论文的自然语言字段，不把图像和参考文献元数据序列化进训练文本。
TEXT_FIELDS = ("article_title", "article_abstract", "article_text")


def split_files(corpus_dir: Path, train_ratio: float = 0.9) -> tuple[list[Path], list[Path]]:
    # 文件名字典序排序使每次运行获得相同的数据划分。
    files = sorted(corpus_dir.glob("*.json"))
    cut = int(len(files) * train_ratio)
    return files[:cut], files[cut:]


def read_document(path: Path) -> str:
    item = json.loads(path.read_text(encoding="utf-8"))
    # 用换行分隔字段；缺失值或 null 按空字符串处理。
    return "\n".join(str(item.get(field) or "") for field in TEXT_FIELDS)


def build_word_counts(paths: Iterable[Path]) -> tuple[Counter[bytes], dict[str, int]]:
    # 相同预切分片段只保存一次，并用 Counter 记录其语料频次。
    # 后续按频次加权即可得到与逐个扫描所有片段相同的统计结果。
    counts: Counter[bytes] = Counter()
    chars = utf8_bytes = pieces = 0
    for path in paths:
        text = read_document(path)
        chars += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        current = pretokenize(text)
        pieces += len(current)
        # BPE 的初始表示是 UTF-8 字节，而不是 Unicode 字符。
        counts.update(piece.encode("utf-8") for piece in current if piece)
    return counts, {
        "characters": chars,
        "utf8_bytes": utf8_bytes,
        "pretoken_count": pieces,
        "unique_pretokens": len(counts),
    }


def merge_pair(sequence: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """从左到右合并 pair 的所有不重叠出现。"""
    output: list[int] = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            output.append(new_id)
            i += 2
        else:
            output.append(sequence[i])
            i += 1
    return tuple(output)


def adjacent_counts(sequence: tuple[int, ...]) -> Counter[tuple[int, int]]:
    """统计一个符号序列中每种相邻 pair 的出现次数。"""
    return Counter(zip(sequence, sequence[1:]))


def learn_merges(
    word_counts: Counter[bytes], num_merges: int, min_frequency: int = 2
) -> tuple[list[tuple[int, int]], list[int]]:
    """Learn exact corpus-frequency merges with incremental pair updates."""
    # sequences 中每项是一种不同预切分片段的当前符号表示；frequencies 是其权重。
    sequences = [tuple(word) for word in word_counts]
    frequencies = list(word_counts.values())

    # pair_frequency: 每个 pair 在整个语料中的加权频次。
    # pair_words: 倒排索引，记录哪些不同片段可能包含该 pair。
    pair_frequency: Counter[tuple[int, int]] = Counter()
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for word_id, sequence in enumerate(sequences):
        for pair, occurrences in adjacent_counts(sequence).items():
            pair_frequency[pair] += occurrences * frequencies[word_id]
            pair_words[pair].add(word_id)

    # heapq 是最小堆，使用负频次即可高效取出当前最高频 pair。
    # 元组的第二项 pair 同时提供确定性的并列频次排序。
    heap = [(-frequency, pair) for pair, frequency in pair_frequency.items()]
    heapq.heapify(heap)
    merges: list[tuple[int, int]] = []
    merge_frequencies: list[int] = []

    for merge_index in range(num_merges):
        # 频次更新时采用“向堆中追加新值”的懒更新策略；弹出时跳过旧值。
        while heap:
            neg_frequency, pair = heapq.heappop(heap)
            if -neg_frequency == pair_frequency.get(pair, 0):
                break
        else:
            break
        frequency = -neg_frequency
        if frequency < min_frequency:
            break

        # 0..255 已被原始字节占用，新合并符号从 256 开始顺序编号。
        new_id = 256 + merge_index
        # 只有包含目标 pair 的片段会变化，无需每轮扫描所有不同片段。
        affected = list(pair_words.get(pair, ()))
        changed_pairs: set[tuple[int, int]] = set()
        for word_id in affected:
            old = sequences[word_id]
            if pair not in zip(old, old[1:]):
                continue
            weight = frequencies[word_id]
            old_pairs = adjacent_counts(old)
            new = merge_pair(old, pair, new_id)
            new_pairs = adjacent_counts(new)
            sequences[word_id] = new
            # 先扣除该片段旧表示贡献的 pair 频次，再加入新表示的贡献。
            for old_pair, count in old_pairs.items():
                pair_frequency[old_pair] -= count * weight
                changed_pairs.add(old_pair)
            for new_pair, count in new_pairs.items():
                pair_frequency[new_pair] += count * weight
                pair_words[new_pair].add(word_id)
                changed_pairs.add(new_pair)

        # merges 的顺序就是编码阶段必须遵循的规则优先级。
        merges.append(pair)
        merge_frequencies.append(frequency)
        for changed in changed_pairs:
            heapq.heappush(heap, (-pair_frequency[changed], changed))
        if (merge_index + 1) % 100 == 0 or merge_index == num_merges - 1:
            print(f"learned {merge_index + 1}/{num_merges} merges; last frequency={frequency}")
    return merges, merge_frequencies


def file_manifest(paths: Iterable[Path]) -> str:
    """对有序文件名生成摘要，用于确认两次训练采用了同一划分。"""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("../gene_related_papers_1000"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tokenizer"))
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()
    if args.vocab_size < 260:
        raise ValueError("vocab-size must include 4 special tokens and 256 byte tokens")

    started = time.perf_counter()
    train_files, validation_files = split_files(args.corpus_dir, args.train_ratio)
    word_counts, corpus_stats = build_word_counts(train_files)
    # 最终词表 = 4 个特殊 token + 256 个原始字节 + 学到的合并 token。
    num_merges = args.vocab_size - 4 - 256
    merges, frequencies = learn_merges(word_counts, num_merges, args.min_frequency)
    tokenizer = ByteBPETokenizer(merges=merges)
    tokenizer.save_pretrained(args.output_dir)
    # 把配置、数据规模和耗时一起保存，方便实验复现与报告核查。
    stats = {
        "corpus_dir": str(args.corpus_dir),
        "text_fields": list(TEXT_FIELDS),
        "split_method": "lexicographic filename order",
        "train_ratio": args.train_ratio,
        "train_files": len(train_files),
        "validation_files": len(validation_files),
        "train_filename_manifest_sha256": file_manifest(train_files),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "learned_merges": len(merges),
        "min_frequency": args.min_frequency,
        "last_merge_frequency": frequencies[-1] if frequencies else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **corpus_stats,
    }
    (args.output_dir / "training_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
