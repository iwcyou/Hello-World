"""Disease Ontology inference helpers for single-term and batch predictions.

By default the script runs batch inference on the evaluation JSON (or a
user-provided dataset) and stores predictions/metrics under ``output/``. Pass
``--single-term`` to reproduce the legacy interactive demo instead, and use
``--base-only`` to compare the unfine-tuned base model against the LoRA adapter.
"""

# extra metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "/data/kunfeng/models/Llama-3.1-8B-Instruct"
LORA_ADAPTER = Path("llama-do-definition-lora")
DEFAULT_DATASET = Path(
    "/home/kfeng/python_code/Hello-World/P_DiseaseOntology/do_terms_test_24.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model: Any | None = None
tokenizer: Any | None = None


def load_inference_components(
    base_model: str = MODEL, adapter_path: Optional[Path] = LORA_ADAPTER
) -> None:
    """Load tokenizer/model, optionally applying a LoRA adapter."""

    global model, tokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=quant_config,
    )

    if adapter_path is not None:
        base = PeftModel.from_pretrained(base, str(adapter_path))

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base.config.pad_token_id = tok.pad_token_id  # type: ignore[attr-defined]
    base.config.eos_token_id = tok.eos_token_id  # type: ignore[attr-defined]

    model = base
    tokenizer = tok


def _ensure_components_loaded() -> None:
    if model is None or tokenizer is None:
        raise RuntimeError(
            "Model/tokenizer not initialized. Call load_inference_components() first."
        )


def _format_list(values: List[str]) -> str:
    if not values:
        return "N/A"
    return ", ".join(values)


def _clean_generated_text(raw: str) -> str:
    if "Definition:" in raw:
        return raw.split("Definition:", 1)[-1].strip()
    return raw.strip()


def _ensure_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def generate_definition(entry: Dict[str, str], max_new_tokens: int = 128) -> str:
    global model, tokenizer
    _ensure_components_loaded()
    assert model is not None and tokenizer is not None
    synonyms = _ensure_str_list(entry.get("synonyms"))
    parents = _ensure_str_list(entry.get("parents"))
    prompt = f"""Generate a precise disease ontology definition.

ID: {entry['id']}
Name: {entry['name']}
Synonyms: {_format_list(synonyms)}
Parents: {_format_list(parents)}

Definition:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return _clean_generated_text(decoded)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(set(pred_tokens) & set(ref_tokens))
    precision = overlap / len(set(pred_tokens))
    recall = overlap / len(set(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(results: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Compute multiple metrics for definition generation:
    - Exact Match
    - Token-level F1
    - ROUGE-L
    - BLEU
    - BERTScore (F1)
    - SentenceTransformer Cosine Similarity
    """

    # ---- prepare scorer models ----
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    references = []
    predictions = []
    for item in results:
        references.append((item.get("reference_definition") or "").strip())
        predictions.append((item.get("predicted_definition") or "").strip())

    # ---- BERTScore ----
    try:
        bert_P, bert_R, bert_F = bert_score(
            predictions, references, lang="en", model_type="allenai/scibert_scivocab_uncased"
        )
        bert_f1_scores = bert_F.tolist()
    except Exception:
        bert_f1_scores = [0.0] * len(results)

    # ---- loop metrics ----
    exact_matches = 0
    token_f1_scores = []
    rouge_scores = []
    bleu_scores = []
    cosine_scores = []

    smoothie = SmoothingFunction().method4

    for ref, pred in zip(references, predictions):
        ref_l = ref.lower()
        pred_l = pred.lower()

        # exact match
        if ref_l == pred_l and ref_l.strip():
            exact_matches += 1

        # token F1
        token_f1_scores.append(_token_f1(pred_l, ref_l))

        # rouge-L
        rouge_val = rouge.score(ref_l, pred_l)["rougeL"].fmeasure
        rouge_scores.append(rouge_val)

        # BLEU (sentence-level)
        try:
            bleu = sentence_bleu(
                [ref_l.split()],
                pred_l.split(),
                smoothing_function=smoothie,
            )
        except ZeroDivisionError:
            bleu = 0.0
        bleu_scores.append(bleu)

        # cosine similarity (sentence transformer)
        try:
            emb1 = st_model.encode(ref_l, convert_to_tensor=True)
            emb2 = st_model.encode(pred_l, convert_to_tensor=True)
            cos = util.cos_sim(emb1, emb2).item()
        except Exception:
            cos = 0.0
        cosine_scores.append(cos)

    n = len(results)

    return {
        "count": n,
        "exact_match_ratio": exact_matches / n,
        "avg_token_f1": sum(token_f1_scores) / n,
        "avg_rougeL": sum(rouge_scores) / n,
        "avg_bleu": sum(bleu_scores) / n,
        "avg_bert_f1": sum(bert_f1_scores) / n,
        "avg_cosine_similarity": sum(cosine_scores) / n,
    }


def run_batch_inference(
    dataset_path: Path,
    output_dir: Path,
    max_records: Optional[int] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        records = payload.get("data") or payload.get("items") or []
    else:
        records = payload

    if not isinstance(records, list):
        raise ValueError("Dataset must be a list of term objects.")

    if max_records:
        records = records[:max_records]

    predictions = []
    for entry in records:
        prediction = generate_definition(entry, max_new_tokens=max_new_tokens)
        predictions.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "reference_definition": entry.get("definition", ""),
                "predicted_definition": prediction,
            }
        )

    metrics = compute_metrics(predictions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"predictions_{dataset_path.stem}_{timestamp}.json"
    metrics_path = output_dir / f"metrics_{dataset_path.stem}_{timestamp}.json"

    with open(prediction_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "predictions_path": prediction_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def _demo_single_term():
    example = {
        "id": "DOID_0040007",
        "name": "abacavir allergy",
        "synonyms": ["ABC allergy"],
        "parents": ["DOID_0060500"],
    }
    definition = generate_definition(example)
    print("Generated definition:\n", definition)


def main():
    parser = argparse.ArgumentParser(description="Disease Ontology inference helper")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a JSON file with DO terms for batch inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store prediction and metric files.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of terms to process.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per definition.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=MODEL,
        help="Base model path or HF hub ID to load for generation.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=LORA_ADAPTER,
        help="Path to a LoRA adapter. Ignored when --base-only is set.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Skip loading the LoRA adapter and use only the base model.",
    )
    parser.add_argument(
        "--single-term",
        action="store_true",
        help="Run the legacy single-term demo instead of batch inference.",
    )
    args = parser.parse_args()

    adapter_path = None if args.base_only else args.adapter
    load_inference_components(base_model=args.base_model, adapter_path=adapter_path)

    if args.single_term:
        _demo_single_term()
    else:
        if not args.dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        results = run_batch_inference(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_records=args.max_records,
            max_new_tokens=args.max_new_tokens,
        )
        print("Batch inference completed.")
        print(f"Predictions saved to: {results['predictions_path']}")
        print(f"Metrics saved to: {results['metrics_path']}")
        print("Metrics:", json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    main()
