import argparse
import json
import os
from collections import defaultdict
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


VALID_VERDICTS = {"ALLOW", "DENY", "ASK_CLARIFY"}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer: Any, sample: dict[str, Any]) -> str:
    msgs = sample.get("messages", [])
    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_first_json(text: str) -> tuple[bool, Any]:
    text = text.strip()
    if not text:
        return False, None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return True, obj
        except json.JSONDecodeError:
            continue
    return False, None


def extract_strict_json(text: str) -> tuple[bool, Any]:
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False, None
    try:
        return True, json.loads(t)
    except json.JSONDecodeError:
        return False, None


def schema_valid(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    required = {"verdict", "confidence", "violations", "notes"}
    if not required.issubset(obj.keys()):
        return False
    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in VALID_VERDICTS:
        return False
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
        return False
    notes = obj.get("notes")
    if not isinstance(notes, str):
        return False
    violations = obj.get("violations")
    if not isinstance(violations, list):
        return False
    for v in violations:
        if not isinstance(v, dict):
            return False
        if "rule_id" not in v or "evidence" not in v:
            return False
        if not isinstance(v["rule_id"], str) or not isinstance(v["evidence"], str):
            return False
    return True


def parse_obj(obj: Any) -> tuple[str | None, list[str]]:
    if not isinstance(obj, dict):
        return None, []
    verdict = obj.get("verdict")
    if isinstance(verdict, str):
        verdict = verdict.strip().upper()
    else:
        verdict = None
    rules: list[str] = []
    violations = obj.get("violations", [])
    if isinstance(violations, list):
        for v in violations:
            if isinstance(v, dict):
                rid = v.get("rule_id")
                if isinstance(rid, str) and rid.strip():
                    rules.append(rid.strip())
            elif isinstance(v, str) and v.strip():
                rules.append(v.strip())
    return verdict, sorted(set(rules))


def generate_predictions(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    for idx, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        gen_ids = out[0][enc["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        parser_ok, parser_obj = extract_first_json(text)
        strict_ok, strict_obj = extract_strict_json(text)
        verdict, pred_rules = parse_obj(parser_obj if parser_ok else None)
        s_valid = schema_valid(strict_obj) if strict_ok else False

        out_rows.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "expected_verdict": row.get("expected_verdict"),
                "expected_rules": sorted(set(row.get("expected_rules", []))),
                "raw_generation": text,
                "parser_json_valid": parser_ok,
                "strict_json_valid": strict_ok,
                "schema_valid": s_valid,
                "pred_verdict": verdict,
                "pred_rules": pred_rules,
            }
        )
        if (idx + 1) % 50 == 0:
            print(f"[gen] {idx + 1}/{len(rows)}")
    return out_rows


def _safe(n: int) -> float:
    return float(n) if n > 0 else 1.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    parser_ok = sum(1 for r in rows if r["parser_json_valid"])
    strict_ok = sum(1 for r in rows if r["strict_json_valid"])
    schema_ok = sum(1 for r in rows if r["schema_valid"])
    verdict_ok = sum(1 for r in rows if r["pred_verdict"] == r["expected_verdict"])
    rules_ok = sum(1 for r in rows if sorted(r["pred_rules"]) == sorted(r["expected_rules"]))

    pred_deny = {r["id"] for r in rows if r["pred_verdict"] == "DENY"}
    gold_deny = {r["id"] for r in rows if r["expected_verdict"] == "DENY"}
    tp = len(pred_deny & gold_deny)
    fn = len(gold_deny - pred_deny)

    return {
        "samples": n,
        "parser_json_valid_rate": parser_ok / _safe(n),
        "strict_json_valid_rate": strict_ok / _safe(n),
        "schema_valid_rate": schema_ok / _safe(n),
        "verdict_accuracy": verdict_ok / _safe(n),
        "rules_exact_match": rules_ok / _safe(n),
        "deny_recall": tp / _safe(len(gold_deny)),
        "unsafe_false_negative_rate_deny": fn / _safe(len(gold_deny)),
    }


def by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get("source", "unknown"))].append(r)
    return {k: summarize(v) for k, v in sorted(buckets.items(), key=lambda x: x[0])}


def markdown_compare(base_m: dict[str, float], adapter_m: dict[str, float]) -> str:
    keys = [
        "samples",
        "parser_json_valid_rate",
        "strict_json_valid_rate",
        "schema_valid_rate",
        "verdict_accuracy",
        "rules_exact_match",
        "deny_recall",
        "unsafe_false_negative_rate_deny",
    ]
    lines = [
        "| metric | base_model | adapter_model | delta(adapter-base) |",
        "|---|---:|---:|---:|",
    ]
    for k in keys:
        b, a = base_m[k], adapter_m[k]
        if k == "samples":
            lines.append(f"| {k} | {int(b)} | {int(a)} | {int(a-b)} |")
        else:
            lines.append(f"| {k} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")
    return "\n".join(lines)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_source_csv(path: str, base_src: dict[str, dict[str, float]], adapter_src: dict[str, dict[str, float]]) -> None:
    cols = [
        "samples",
        "parser_json_valid_rate",
        "strict_json_valid_rate",
        "schema_valid_rate",
        "verdict_accuracy",
        "rules_exact_match",
        "deny_recall",
        "unsafe_false_negative_rate_deny",
    ]
    sources = sorted(set(base_src.keys()) | set(adapter_src.keys()))
    with open(path, "w", encoding="utf-8") as f:
        headers = ["source"]
        for c in cols:
            headers.extend([f"base_{c}", f"adapter_{c}", f"delta_{c}"])
        f.write(",".join(headers) + "\n")
        for src in sources:
            b = base_src.get(src, {})
            a = adapter_src.get(src, {})
            row = [f'"{src}"']
            for c in cols:
                bv = float(b.get(c, 0.0))
                av = float(a.get(c, 0.0))
                dv = av - bv
                row.extend([f'"{bv:.6f}"', f'"{av:.6f}"', f'"{dv:+.6f}"'])
            f.write(",".join(row) + "\n")


def get_quant_config() -> BitsAndBytesConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(model_id: str, local_files_only: bool) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_quant_config(),
        device_map="auto",
        local_files_only=local_files_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/llama32_3b_judge_qlora/adapter_final")
    parser.add_argument("--test-file", default="combined_judge_dataset/test.jsonl")
    parser.add_argument("--out-dir", default="outputs/llama32_3b_judge_qlora/eval_large")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_jsonl(args.test_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No test rows found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[info] loaded test rows: {len(rows)}")
    print("[info] loading base model ...")
    base_model = load_base_model(args.model_id, args.local_files_only)
    base_model.eval()

    print("[info] generating base predictions ...")
    base_preds = generate_predictions(base_model, tokenizer, rows, args.max_new_tokens)
    base_metrics = summarize(base_preds)
    base_by_source = by_source(base_preds)

    print("[info] loading adapter onto base model ...")
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapter_model.eval()

    print("[info] generating adapter predictions ...")
    adapter_preds = generate_predictions(adapter_model, tokenizer, rows, args.max_new_tokens)
    adapter_metrics = summarize(adapter_preds)
    adapter_by_source = by_source(adapter_preds)

    summary_md = markdown_compare(base_metrics, adapter_metrics)

    write_json(os.path.join(args.out_dir, "metrics_base.json"), base_metrics)
    write_json(os.path.join(args.out_dir, "metrics_adapter.json"), adapter_metrics)
    write_json(os.path.join(args.out_dir, "metrics_base_by_source.json"), base_by_source)
    write_json(os.path.join(args.out_dir, "metrics_adapter_by_source.json"), adapter_by_source)
    write_json(os.path.join(args.out_dir, "predictions_base.json"), base_preds)
    write_json(os.path.join(args.out_dir, "predictions_adapter.json"), adapter_preds)
    write_source_csv(os.path.join(args.out_dir, "comparison_by_source.csv"), base_by_source, adapter_by_source)
    with open(os.path.join(args.out_dir, "comparison_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md + "\n")

    print("\n=== OVERALL COMPARISON ===")
    print(summary_md)
    print(f"\n[done] wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
