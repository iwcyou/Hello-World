"""Minimal demonstration of the interface expected by a later GPT model."""

from pathlib import Path

from byte_bpe_tokenizer import ByteBPETokenizer


EXAMPLE_TEXT = "The proto-oncogene c-Myc is vital for vascular development and promotes tumor angiogenesis, but the mechanisms by which it controls blood vessel growth remain unclear. In the present work we investigated the effects of c-Myc knockdown in endothelial cell functions essential for angiogenesis to define its role in the vasculature. We provide the first evidence that reduction in c-Myc expression in endothelial cells leads to a pro-inflammatory senescent phenotype, features typically observed during vascular aging and pathologies associated with endothelial dysfunction. c-Myc knockdown in human umbilical vein endothelial cells using lentivirus expressing specific anti-c-Myc shRNA reduced proliferation and tube formation. These functional defects were associated with morphological changes, increase in senescence-associated-β-galactosidase activity, upregulation of cell cycle inhibitors and accumulation of c-Myc-deficient cells in G1-phase, indicating that c-Myc knockdown in endothelial cells induces senescence. Gene expression analysis of c-Myc-deficient endothelial cells showed that senescent phenotype was accompanied by significant upregulation of growth factors, adhesion molecules, extracellular-matrix components and remodeling proteins, and a cluster of pro-inflammatory mediators, which include Angptl4, Cxcl12, Mdk, Tgfb2 and Tnfsf15. At the peak of expression of these cytokines, transcription factors known to be involved in growth control (E2f1, Id1 and Myb) were downregulated, while those involved in inflammatory responses (RelB, Stat1, Stat2 and Stat4) were upregulated. Our results demonstrate a novel role for c-Myc in the prevention of vascular pro-inflammatory phenotype, supporting an important physiological function as a central regulator of inflammation and endothelial dysfunction."


def main() -> None:
    import torch
    from torch import nn

    script_dir = Path(__file__).resolve().parent
    tokenizer_dir = script_dir / "outputs" / "tokenizer"
    tokenizer = ByteBPETokenizer.from_pretrained(tokenizer_dir)
    texts = [EXAMPLE_TEXT]
    batch = tokenizer(
        texts,
        add_bos=True,
        add_eos=True,
        padding=True,
        return_tensors="pt",
    )
    embedding = nn.Embedding(tokenizer.vocab_size, 64, padding_idx=tokenizer.pad_token_id)
    hidden = embedding(batch["input_ids"])

    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("embedding shape:", tuple(hidden.shape))
    print("tokens:", [tokenizer.special_tokens[token_id] if token_id < tokenizer.offset else tokenizer.symbol_bytes[token_id - tokenizer.offset].decode("utf-8", errors="backslashreplace") for token_id in batch["input_ids"][0].tolist()])
    decoded_texts = [tokenizer.decode(token_ids.tolist()) for token_ids in batch["input_ids"]]
    successful_map_backs = sum(decoded == original for decoded, original in zip(decoded_texts, texts))
    map_back_success_rate = successful_map_backs / len(texts) if texts else 0.0
    print("roundtrip:", decoded_texts[0])
    print("map_back_result:", f"{successful_map_backs}/{len(texts)}")
    print("map_back_success_rate:", f"{map_back_success_rate:.2%}")


if __name__ == "__main__":
    main()
