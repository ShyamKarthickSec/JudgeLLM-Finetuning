import argparse
import json
import os
import re
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


VERDICTS = {"ALLOW", "DENY", "ASK_CLARIFY"}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


def first_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def parse_prediction(text: str) -> tuple[bool, str | None, list[str], Any]:
    jtxt = first_json_object(text)
    if not jtxt:
        return False, None, [], None
    try:
        obj = json.loads(jtxt)
    except json.JSONDecodeError:
        return False, None, [], None
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
    return True, verdict, sorted(set(rules)), obj


def generate_outputs(
    model: Any,
    tokenizer: Any,
    data: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    for row in data:
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
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        is_json, verdict, pred_rules, pred_obj = parse_prediction(gen_text)
        outputs.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "expected_verdict": row.get("expected_verdict"),
                "expected_rules": sorted(set(row.get("expected_rules", []))),
                "raw_generation": gen_text,
                "json_valid": is_json,
                "pred_verdict": verdict,
                "pred_rules": pred_rules,
                "pred_obj": pred_obj,
            }
        )
    return outputs


def safe_denominator(n: int) -> float:
    return float(n) if n > 0 else 1.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    json_valid = sum(1 for r in rows if r["json_valid"])
    verdict_exact = sum(1 for r in rows if r["pred_verdict"] == r["expected_verdict"])
    rules_exact = sum(1 for r in rows if sorted(r["pred_rules"]) == sorted(r["expected_rules"]))

    pred_deny = {r["id"] for r in rows if r["pred_verdict"] == "DENY"}
    gold_deny = {r["id"] for r in rows if r["expected_verdict"] == "DENY"}
    tp = len(pred_deny & gold_deny)
    fn = len(gold_deny - pred_deny)
    unsafe_false_negative_rate = fn / safe_denominator(len(gold_deny))

    return {
        "samples": total,
        "json_valid_rate": json_valid / safe_denominator(total),
        "verdict_accuracy": verdict_exact / safe_denominator(total),
        "rules_exact_match": rules_exact / safe_denominator(total),
        "unsafe_false_negative_rate_deny": unsafe_false_negative_rate,
        "deny_recall": tp / safe_denominator(len(gold_deny)),
    }


def markdown_summary(base_m: dict[str, float], adapter_m: dict[str, float]) -> str:
    keys = [
        "samples",
        "json_valid_rate",
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
        b = base_m[k]
        a = adapter_m[k]
        d = a - b
        if k == "samples":
            lines.append(f"| {k} | {int(b)} | {int(a)} | {int(d)} |")
        else:
            lines.append(f"| {k} | {b:.4f} | {a:.4f} | {d:+.4f} |")
    return "\n".join(lines)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    headers = [
        "id",
        "source",
        "expected_verdict",
        "expected_rules",
        "base_json_valid",
        "base_pred_verdict",
        "base_pred_rules",
        "adapter_json_valid",
        "adapter_pred_verdict",
        "adapter_pred_rules",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = []
            for h in headers:
                val = r.get(h, "")
                if isinstance(val, list):
                    val = "|".join(str(x) for x in val)
                val = str(val).replace('"', '""')
                vals.append(f'"{val}"')
            f.write(",".join(vals) + "\n")


def make_comparison_rows(base_rows: list[dict[str, Any]], adapter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["id"]: r for r in adapter_rows}
    out = []
    for b in base_rows:
        a = by_id.get(b["id"], {})
        out.append(
            {
                "id": b["id"],
                "source": b["source"],
                "expected_verdict": b["expected_verdict"],
                "expected_rules": b["expected_rules"],
                "base_json_valid": b["json_valid"],
                "base_pred_verdict": b["pred_verdict"],
                "base_pred_rules": b["pred_rules"],
                "adapter_json_valid": a.get("json_valid"),
                "adapter_pred_verdict": a.get("pred_verdict"),
                "adapter_pred_rules": a.get("pred_rules", []),
            }
        )
    return out


def get_quant_config() -> BitsAndBytesConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(model_id: str) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_quant_config(),
        device_map="auto",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--adapter-dir",
        default="outputs/llama32_3b_judge_qlora/adapter_final",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--test-file",
        default="judge_sft_llama4scout_1k/judge_sft_llama4scout_1k/test.jsonl",
        help="Path to test jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/llama32_3b_judge_qlora/eval",
        help="Directory to write eval artifacts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full test set")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test_rows = load_jsonl(args.test_file)
    if args.max_samples > 0:
        test_rows = test_rows[: args.max_samples]
    if not test_rows:
        raise ValueError("No test rows found")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[info] loaded {len(test_rows)} test samples")
    print("[info] loading base model ...")
    base_model = load_base_model(args.model_id)
    base_model.eval()

    print("[info] running base model on test set ...")
    base_rows = generate_outputs(base_model, tokenizer, test_rows, args.max_new_tokens)
    base_metrics = summarize(base_rows)

    print("[info] loading adapter on base model ...")
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapter_model.eval()

    print("[info] running adapter model on test set ...")
    adapter_rows = generate_outputs(adapter_model, tokenizer, test_rows, args.max_new_tokens)
    adapter_metrics = summarize(adapter_rows)

    summary_md = markdown_summary(base_metrics, adapter_metrics)
    comparison_rows = make_comparison_rows(base_rows, adapter_rows)

    write_json(os.path.join(args.out_dir, "metrics_base.json"), base_metrics)
    write_json(os.path.join(args.out_dir, "metrics_adapter.json"), adapter_metrics)
    write_json(os.path.join(args.out_dir, "predictions_base.json"), base_rows)
    write_json(os.path.join(args.out_dir, "predictions_adapter.json"), adapter_rows)
    write_csv(os.path.join(args.out_dir, "comparison_table.csv"), comparison_rows)
    with open(os.path.join(args.out_dir, "comparison_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md + "\n")

    print("\n=== COMPARISON SUMMARY ===")
    print(summary_md)
    print(f"\n[done] artifacts written to: {args.out_dir}")


if __name__ == "__main__":
    main()
