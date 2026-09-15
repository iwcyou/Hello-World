"""A small, dependency-free byte-level BPE tokenizer.

The implementation is intentionally readable for an NLP programming assignment.
Text is pre-tokenized with Python's Unicode-aware ``re`` module, converted to
UTF-8 bytes, and then transformed with learned BPE merge rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence


# 依次匹配：连续空白、Unicode 字母、数字、下划线、其他标点/符号。
# 各分支覆盖所有字符，因此预切分不会删除内容，也不会让 BPE 跨片段合并。
_PRETOKEN_PATTERN = re.compile(r"\s+|[^\W\d_]+|\d+|_+|[^\w\s]+", re.UNICODE)


def pretokenize(text: str) -> list[str]:
    """Split text without dropping any character."""
    pieces = _PRETOKEN_PATTERN.findall(text)
    # 主动检查可逆性，防止修改正则后静默丢失字符。
    if "".join(pieces) != text:
        raise ValueError("Pre-tokenization was not lossless")
    return pieces


class ByteBPETokenizer:
    """Byte-level BPE tokenizer with a GPT-style training interface.

    Internal BPE symbols 0..255 are raw bytes. Every merge appends one new
    symbol. Public token IDs reserve the first IDs for special tokens.
    """

    VERSION = 1

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] | None = None,
        special_tokens: Sequence[str] = ("<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"),
    ) -> None:
        # 对外 token ID 的开头留给特殊 token；BPE 内部符号仍从 0 开始编号。
        self.special_tokens = list(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        self.special_to_id = {token: i for i, token in enumerate(self.special_tokens)}
        self.offset = len(self.special_tokens)
        self.merges = [tuple(pair) for pair in (merges or [])]
        # rank 越小表示该合并规则越早学到，编码时优先级越高。
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # 内部符号 0..255 分别代表一个原始字节。
        self.symbol_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for rank, (left, right) in enumerate(self.merges):
            expected_max = 256 + rank
            # 一条规则只能引用原始字节或此前已经产生的合并符号。
            if not (0 <= left < expected_max and 0 <= right < expected_max):
                raise ValueError(f"Invalid merge {rank}: {(left, right)}")
            # 保存每个合并符号实际代表的完整字节串，解码时可直接查表拼接。
            self.symbol_bytes.append(self.symbol_bytes[left] + self.symbol_bytes[right])

    @property
    def vocab_size(self) -> int:
        return self.offset + len(self.symbol_bytes)

    @property
    def pad_token_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def unk_token_id(self) -> int:
        return self.special_to_id["<|unk|>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def _apply_bpe(self, byte_values: bytes) -> list[int]:
        # 输入先由单字节符号构成，再按照训练阶段固定的规则逐轮合并。
        symbols = list(byte_values)
        while len(symbols) > 1:
            # 找出当前序列中所有可用规则，选择 rank 最小（最早学到）的规则。
            candidates = (
                (self.merge_ranks[(symbols[i], symbols[i + 1])], i)
                for i in range(len(symbols) - 1)
                if (symbols[i], symbols[i + 1]) in self.merge_ranks
            )
            best = min(candidates, default=None)
            if best is None:
                break
            rank, _ = best
            pair = self.merges[rank]
            # 第 rank 条规则产生的内部符号 ID 固定为 256 + rank。
            merged_symbol = 256 + rank
            updated: list[int] = []
            i = 0
            # 从左到右替换本轮 pair 的所有不重叠出现。
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                    updated.append(merged_symbol)
                    i += 2
                else:
                    updated.append(symbols[i])
                    i += 1
            symbols = updated
        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        for piece in pretokenize(text):
            # 每个预切分片段独立编码，offset 将内部符号转换为公开 token ID。
            ids.extend(self.offset + symbol for symbol in self._apply_bpe(piece.encode("utf-8")))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        # 先恢复原始字节流，最后只执行一次严格 UTF-8 解码。
        data = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError(f"Token ID out of range: {token_id}")
            if token_id < self.offset:
                # 训练文本通常跳过 pad/bos/eos；关闭 skip 后则输出其文本形式。
                if skip_special_tokens:
                    continue
                data.extend(self.special_tokens[token_id].encode("utf-8"))
                continue
            symbol = token_id - self.offset
            if not 0 <= symbol < len(self.symbol_bytes):
                raise ValueError(f"Token ID out of range: {token_id}")
            data.extend(self.symbol_bytes[symbol])
        return bytes(data).decode("utf-8", errors="strict")

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        """Encode one string or a batch; optionally return PyTorch tensors."""
        is_batch = not isinstance(text, str)
        texts = list(text) if is_batch else [text]
        batches = [self.encode(x, add_bos=add_bos, add_eos=add_eos) for x in texts]
        if padding:
            # 右侧补齐到批次最大长度；真实 token 的 mask 为 1，pad 为 0。
            width = max((len(x) for x in batches), default=0)
            masks = [[1] * len(x) + [0] * (width - len(x)) for x in batches]
            batches = [x + [self.pad_token_id] * (width - len(x)) for x in batches]
        else:
            masks = [[1] * len(x) for x in batches]
        input_ids: object = batches if is_batch else batches[0]
        attention_mask: object = masks if is_batch else masks[0]
        if return_tensors is not None:
            if return_tensors != "pt":
                raise ValueError("Only return_tensors='pt' is supported")
            if not padding and is_batch and len({len(x) for x in batches}) > 1:
                raise ValueError("Set padding=True for a variable-length tensor batch")
            try:
                # PyTorch 仅在用户请求张量输出时导入，核心 tokenizer 无此依赖。
                import torch
            except ImportError as exc:
                raise ImportError("PyTorch is required for return_tensors='pt'") from exc
            input_ids = torch.tensor(batches if is_batch else [batches[0]], dtype=torch.long)
            attention_mask = torch.tensor(masks if is_batch else [masks[0]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def vocab_records(self) -> list[dict[str, object]]:
        # 用十六进制保存字节可避免控制字符或非完整 UTF-8 字节破坏 JSON。
        records = [
            {"id": i, "kind": "special", "token": token, "bytes_hex": None}
            for i, token in enumerate(self.special_tokens)
        ]
        records.extend(
            {
                "id": self.offset + symbol,
                "kind": "byte" if symbol < 256 else "merge",
                "token": f"<0x{value.hex().upper()}>",
                "bytes_hex": value.hex(),
            }
            for symbol, value in enumerate(self.symbol_bytes)
        )
        return records

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # tokenizer.json 是重新加载所需的权威配置。
        config = {
            "version": self.VERSION,
            "model_type": "byte_level_bpe",
            "pretokenizer": _PRETOKEN_PATTERN.pattern,
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
        }
        (directory / "tokenizer.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # vocab.json 面向查看/提交；真正恢复模型时以 merges 和特殊 token 为准。
        vocab = {record["token"]: record["id"] for record in self.vocab_records()}
        (directory / "vocab.json").write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (directory / "merges.txt").open("w", encoding="utf-8") as handle:
            handle.write("#version: byte-bpe-v1\n")
            for rank, (left, right) in enumerate(self.merges):
                handle.write(f"{rank}\t{left}\t{right}\t{256 + rank}\n")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "ByteBPETokenizer":
        # 只需特殊 token 和有序 merges 即可重建全部符号及 ID 映射。
        config = json.loads((Path(directory) / "tokenizer.json").read_text(encoding="utf-8"))
        if config.get("version") != cls.VERSION:
            raise ValueError(f"Unsupported tokenizer version: {config.get('version')}")
        return cls(
            merges=[tuple(pair) for pair in config["merges"]],
            special_tokens=config["special_tokens"],
        )
