#!/usr/bin/env python3
"""Cross-validate LLM Judge with a second model (Qwen2.5-72B).

Samples 50 abstractive QA pairs, judges with Qwen, compares against DeepSeek-V3 judge.
Reports agreement rate, Cohen's kappa, and per-system accuracy with both judges.
"""

import json
import random
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

ANSWER_JUDGE_PROMPT = """你的任务是判断系统生成的答案是否正确。

## 问题
{question}

## 标准答案
{gold_answer}

## 系统生成的答案
{generated_answer}

## 判断标准
- 只要系统答案涉及与标准答案相同的主题，即判定为正确
- 对于时间相关问题，只要指代相同的日期或时间段，即使格式不同也应判定为正确
- 如果系统答案比标准答案长但包含正确核心信息，应判定为正确
- 如果系统答案与标准答案无关或矛盾，判定为错误

请先提供一句推理，然后输出 CORRECT 或 WRONG。
以 JSON 格式返回，key 为 "label"。"""


def parse_judge_response(response: str) -> bool:
    """Parse judge response."""
    if not response:
        return False
    import re
    try:
        json_match = re.search(r'\{[^}]*"label"\s*:\s*"(CORRECT|WRONG)"[^}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("label", "WRONG") == "CORRECT"
    except json.JSONDecodeError:
        pass
    if "CORRECT" in response.upper() and "WRONG" not in response.upper()[:50]:
        return True
    elif "WRONG" in response.upper():
        return False
    if "正确" in response[:50]:
        return True
    return False


def main():
    random.seed(42)

    base_dir = Path(__file__).parent.parent
    result_dir = base_dir / "data" / "eval_results"

    # Load QA metadata
    qa_data = json.loads((base_dir / "data" / "validated" / "bitpqa_test_zh.json").read_text(encoding="utf-8"))

    # Find abstractive QA pairs
    abstractive_qas = []
    for qa in qa_data["qa_pairs"]:
        atype = qa.get("answer_type", {})
        if isinstance(atype, dict):
            atype = atype.get("value", "")
        if atype == "abstractive":
            abstractive_qas.append(qa)

    print(f"Total abstractive QA pairs: {len(abstractive_qas)}")

    # Sample 50
    sample = random.sample(abstractive_qas, min(50, len(abstractive_qas)))
    print(f"Sampled {len(sample)} QA pairs for cross-validation")

    # Load DeepSeek judge results from evaluation files
    system_files = {
        "FAISS Vector Store": "eval_faiss_vector_store_1774518673.json",
        "BM25": "eval_bm25_1774522334.json",
        "Naive RAG": "eval_naive_rag_1774521546.json",
    }

    # Build judge_results: qa_id -> {system_name -> ds_judge_verdict}
    ds_judge = {}
    generated_answers = {}

    for sys_name, fname in system_files.items():
        fpath = result_dir / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for sr in data["scenario_results"]:
            for qr in sr["qa_results"]:
                qid = qr["qa_id"]
                if qid not in ds_judge:
                    ds_judge[qid] = {}
                if qid not in generated_answers:
                    generated_answers[qid] = {}
                ds_judge[qid][sys_name] = qr["is_correct"]
                generated_answers[qid][sys_name] = qr["answer"]["raw_response"]

    # Find QA IDs in our sample that have results
    sample_ids = {qa["qa_id"] for qa in sample}
    available_ids = sample_ids & set(ds_judge.keys())
    print(f"Available for cross-validation: {len(available_ids)}")

    if len(available_ids) < 20:
        print("ERROR: Not enough QA pairs with results for meaningful cross-validation")
        return

    # Initialize Qwen client
    client = OpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-xjakkyzbynwbvxvetpspqwnsrzfqblgkhkrfrrdmcgtqmwwz",
        timeout=60,
    )
    qwen_model = "Qwen/Qwen2.5-72B-Instruct"

    # Judge with Qwen
    print(f"\nJudging {len(available_ids)} QA pairs with {qwen_model}...")

    qwen_judge = {}
    qa_lookup = {qa["qa_id"]: qa for qa in qa_data["qa_pairs"]}

    # Use FAISS results for cross-validation (most representative)
    target_system = "FAISS Vector Store"
    judged = 0
    for qid in sorted(available_ids):
        if target_system not in ds_judge.get(qid, {}):
            continue
        if target_system not in generated_answers.get(qid, {}):
            continue

        qa = qa_lookup[qid]
        gen_answer = generated_answers[qid][target_system]
        gold = qa["answer_zh"]
        question = qa["question_zh"]

        prompt = ANSWER_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold,
            generated_answer=gen_answer,
        )

        try:
            resp = client.chat.completions.create(
                model=qwen_model,
                messages=[
                    {"role": "system", "content": "你是一个公正严格的评判者。你的任务是判断一个系统生成的答案是否正确。请严格按照格式回复。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            qwen_verdict = parse_judge_response(raw)
            qwen_judge[qid] = {
                "qwen_correct": qwen_verdict,
                "ds_correct": ds_judge[qid][target_system],
                "qwen_raw": raw[:200],
                "generated_answer": gen_answer[:100],
                "gold_answer": gold[:100],
            }
            judged += 1
            if judged % 10 == 0:
                print(f"  Judged {judged}/{len(available_ids)}...")
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            print(f"  Error judging {qid}: {e}")
            time.sleep(2)

    print(f"\nJudged {judged} QA pairs total")

    # Compute agreement
    agree = sum(1 for v in qwen_judge.values() if v["qwen_correct"] == v["ds_correct"])
    total = len(qwen_judge)
    agreement = agree / total if total > 0 else 0

    # Compute Cohen's kappa
    both_correct = sum(1 for v in qwen_judge.values() if v["qwen_correct"] and v["ds_correct"])
    both_wrong = sum(1 for v in qwen_judge.values() if not v["qwen_correct"] and not v["ds_correct"])
    qwen_correct_ds_wrong = sum(1 for v in qwen_judge.values() if v["qwen_correct"] and not v["ds_correct"])
    ds_correct_qwen_wrong = sum(1 for v in qwen_judge.values() if not v["qwen_correct"] and v["ds_correct"])

    p_obs = agreement
    p_qwen_correct = (both_correct + qwen_correct_ds_wrong) / total
    p_qwen_wrong = (ds_correct_qwen_wrong + both_wrong) / total
    p_ds_correct = (both_correct + ds_correct_qwen_wrong) / total
    p_ds_wrong = (qwen_correct_ds_wrong + both_wrong) / total
    p_exp = p_qwen_correct * p_ds_correct + p_qwen_wrong * p_ds_wrong
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 0

    # Qwen accuracy
    qwen_acc = sum(1 for v in qwen_judge.values() if v["qwen_correct"]) / total
    ds_acc = sum(1 for v in qwen_judge.values() if v["ds_correct"]) / total

    print(f"\n{'='*70}")
    print(f"LLM JUDGE CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  Judge 1: DeepSeek-V3 (original)")
    print(f"  Judge 2: Qwen2.5-72B-Instruct")
    print(f"  System evaluated: {target_system}")
    print(f"  Sample size: {total} abstractive QA pairs")
    print(f"\n  Agreement rate: {agree}/{total} = {agreement*100:.1f}%")
    print(f"  Cohen's kappa: {kappa:.3f}", end="")
    if kappa > 0.8:
        print(" (almost perfect)")
    elif kappa > 0.6:
        print(" (substantial)")
    elif kappa > 0.4:
        print(" (moderate)")
    elif kappa > 0.2:
        print(" (fair)")
    else:
        print(" (slight/poor)")
    print(f"\n  DeepSeek-V3 accuracy: {ds_acc*100:.1f}%")
    print(f"  Qwen2.5-72B accuracy: {qwen_acc*100:.1f}%")
    print(f"\n  Confusion matrix:")
    print(f"                    Qwen CORRECT  Qwen WRONG")
    print(f"    DeepSeek CORRECT    {both_correct:>5}        {ds_correct_qwen_wrong:>5}")
    print(f"    DeepSeek WRONG      {qwen_correct_ds_wrong:>5}        {both_wrong:>5}")

    # Show disagreements
    disagreements = [(qid, v) for qid, v in qwen_judge.items() if v["qwen_correct"] != v["ds_correct"]]
    print(f"\n  Disagreements ({len(disagreements)}):")
    for qid, v in disagreements[:10]:
        ds_label = "CORRECT" if v["ds_correct"] else "WRONG"
        qw_label = "CORRECT" if v["qwen_correct"] else "WRONG"
        print(f"\n    [{qid}]")
        print(f"    Gold: {v['gold_answer']}")
        print(f"    Generated: {v['generated_answer']}")
        print(f"    DeepSeek: {ds_label} | Qwen: {qw_label}")

    # Save results
    results = {
        "description": "LLM Judge cross-validation: DeepSeek-V3 vs Qwen2.5-72B",
        "sample_size": total,
        "system": target_system,
        "agreement_rate": round(agreement, 4),
        "cohens_kappa": round(kappa, 4),
        "deepseek_accuracy": round(ds_acc, 4),
        "qwen_accuracy": round(qwen_acc, 4),
        "confusion_matrix": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "ds_correct_qwen_wrong": ds_correct_qwen_wrong,
            "qwen_correct_ds_wrong": qwen_correct_ds_wrong,
        },
        "disagreements": [
            {
                "qa_id": qid,
                "gold_answer": v["gold_answer"],
                "generated_answer": v["generated_answer"],
                "deepseek_verdict": "CORRECT" if v["ds_correct"] else "WRONG",
                "qwen_verdict": "CORRECT" if v["qwen_correct"] else "WRONG",
            }
            for qid, v in disagreements
        ],
    }

    out_path = result_dir / "judge_crossvalidation.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
