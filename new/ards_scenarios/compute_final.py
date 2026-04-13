#!/usr/bin/env python3
"""Compute PA and SEED-Bench accuracy, then generate summary markdown."""
import json, os, sys

OUTPUT_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/eval_outputs"
SCENARIOS_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"

all_results = {}

# ---- ScienceQA Clean (already computed) ----
f = os.path.join(OUTPUT_DIR, "scienceqa/clean/llava-v1.5-7b_result.json")
if os.path.exists(f):
    all_results['sqa_clean'] = json.load(open(f))
    print(f"SQA Clean: {all_results['sqa_clean']['acc']:.2f}%")

# ---- ScienceQA SA (already computed) ----
f = os.path.join(OUTPUT_DIR, "scienceqa/sa_attack/llava-v1.5-7b_result.json")
if os.path.exists(f):
    all_results['sqa_sa'] = json.load(open(f))
    print(f"SQA SA: {all_results['sqa_sa']['acc']:.2f}%")

# ---- ScienceQA PA ----
f = os.path.join(OUTPUT_DIR, "scienceqa/pa_attack/llava-v1.5-7b.jsonl")
if os.path.exists(f):
    lines = [json.loads(l) for l in open(f)]
    print(f"SQA PA lines: {len(lines)}")
    if lines:
        k0 = list(lines[0].keys())
        print(f"  keys: {k0}")
        # Check format
        if 'attack_results' in lines[0]:
            total = correct = 0
            for line in lines:
                attacks = line.get('attack_results', [])
                if not attacks:
                    continue
                total += 1
                if all(a.get('text','') == a.get('label','') for a in attacks):
                    correct += 1
            acc = correct / total * 100 if total > 0 else 0
            all_results['sqa_pa'] = {'acc': acc, 'correct': correct, 'count': total}
            print(f"SQA PA: {acc:.2f}% ({correct}/{total})")
        else:
            # Try alternative format
            has_text = 'text' in lines[0]
            has_label = any(k in lines[0] for k in ['label', 'answer', 'gt'])
            print(f"  has_text={has_text}, has_label={has_label}")
            print(f"  sample: {json.dumps(lines[0], indent=2)[:500]}")

# ---- SEED-Bench Clean ----
f = os.path.join(OUTPUT_DIR, "seed_bench/clean/llava-v1.5-7b/merge.jsonl")
anno_f = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/seed_bench/SEED-Bench.json"
if os.path.exists(f) and os.path.exists(anno_f):
    preds = [json.loads(l) for l in open(f)]
    anno = json.load(open(anno_f))
    qid2ans = {}
    qid2type = {}
    type_names = anno.get('question_type', {})
    for q in anno['questions']:
        if q['data_type'] == 'image':
            qid2ans[int(q['question_id'])] = q['answer']
            qid2type[int(q['question_id'])] = q['question_type_id']

    total = correct = 0
    type_c = {}
    type_t = {}
    for p in preds:
        qid = p['question_id']
        if isinstance(qid, str):
            try:
                qid = int(qid)
            except ValueError:
                continue
        if qid not in qid2ans:
            continue
        gt = qid2ans[qid]
        pred_text = p['text'].strip()
        answer = pred_text[0] if pred_text and pred_text[0] in 'ABCD' else pred_text
        total += 1
        qt = qid2type[qid]
        type_t[qt] = type_t.get(qt, 0) + 1
        if answer == gt:
            correct += 1
            type_c[qt] = type_c.get(qt, 0) + 1
    
    seed_acc = correct / total * 100 if total > 0 else 0
    per_type = {}
    for qt in sorted(type_t.keys()):
        tc = type_c.get(qt, 0)
        tt = type_t[qt]
        name = type_names.get(str(qt), f"Type {qt}")
        per_type[str(qt)] = {'name': name, 'acc': tc/tt*100, 'correct': tc, 'count': tt}
    
    all_results['seed_clean'] = {'acc': seed_acc, 'correct': correct, 'count': total, 'per_type': per_type}
    print(f"SEED Clean: {seed_acc:.2f}% ({correct}/{total})")
    for qt in sorted(per_type.keys(), key=int):
        info = per_type[qt]
        print(f"  [{qt}] {info['name']}: {info['acc']:.1f}% ({info['correct']}/{info['count']})")

# Save results JSON
json.dump(all_results, open(os.path.join(OUTPUT_DIR, "all_results.json"), 'w'), indent=2)
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'all_results.json')}")

# ---- Generate Summary Markdown ----
md = []
md.append("# Reusable Baselines Summary\n")
md.append("**Model**: `liuhaotian/llava-v1.5-7b` (pre-trained, zero-shot / full-data baseline)")
md.append("**Environment**: `/LOCAL2/zhuoyun/PAC_robust/ards_venv`")
md.append("**ARDS repo**: `/LOCAL2/zhuoyun/PAC_robust/ARDS`\n")
md.append("---\n")
md.append("## Results Table\n")
md.append("| Task | Eval Type | Accuracy | Samples | Status |")
md.append("|------|-----------|----------|---------|--------|")

if 'sqa_clean' in all_results:
    r = all_results['sqa_clean']
    md.append(f"| ScienceQA | Clean | {r['acc']:.2f}% | {r['count']} | Done |")
if 'sqa_sa' in all_results:
    r = all_results['sqa_sa']
    md.append(f"| ScienceQA | SA (Symbol Attack QWERT) | {r['acc']:.2f}% | {r['count']} | Done |")
if 'sqa_pa' in all_results:
    r = all_results['sqa_pa']
    md.append(f"| ScienceQA | PA (Position Attack) | {r['acc']:.2f}% | {r['count']} | Done |")
if 'seed_clean' in all_results:
    r = all_results['seed_clean']
    md.append(f"| SEED-Bench (image) | Clean | {r['acc']:.2f}% | {r['count']} | Done |")
md.append("")

if 'seed_clean' in all_results and 'per_type' in all_results['seed_clean']:
    md.append("\n## SEED-Bench Per-Type Breakdown\n")
    md.append("| Type | Category | Accuracy | Samples |")
    md.append("|------|----------|----------|---------|")
    for qt in sorted(all_results['seed_clean']['per_type'].keys(), key=int):
        info = all_results['seed_clean']['per_type'][qt]
        md.append(f"| {qt} | {info['name']} | {info['acc']:.1f}% | {info['count']} |")
    md.append("")

md.append("\n---\n")
md.append("## Output Paths\n")
md.append("| Item | Path |")
md.append("|------|------|")
md.append("| ARDS venv | `/LOCAL2/zhuoyun/PAC_robust/ards_venv` |")
md.append("| Model weights | `/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b` |")
md.append("| SQA Clean answers | `eval_outputs/scienceqa/clean/llava-v1.5-7b.jsonl` |")
md.append("| SQA SA answers | `eval_outputs/scienceqa/sa_attack/llava-v1.5-7b.jsonl` |")
md.append("| SQA PA answers | `eval_outputs/scienceqa/pa_attack/llava-v1.5-7b.jsonl` |")
md.append("| SEED Clean answers | `eval_outputs/seed_bench/clean/llava-v1.5-7b/merge.jsonl` |")
md.append("| All results JSON | `eval_outputs/all_results.json` |")

md.append("\n---\n")
md.append("## Fixes Applied\n")
md.append("1. `model_vqa_science_option_attack.py`: Added missing `--eval-img` argument")
md.append("2. ScienceQA SA: Generated `llava_test_CQM-A_convertedABCDE-QWERT.json` (ABCDE->QWERT)")
md.append("3. SEED-Bench: Fixed image paths (no `.jpg` extension)")
md.append("4. Environment: Created dedicated venv (`transformers==4.37.2` required by ARDS)")

md.append("\n---\n")
md.append("## Next Steps for Plugin\n")
md.append("- All baselines can be used as comparison targets for LoRA / Plugin-LoRA")
md.append("- LoRA fine-tune: use `scripts/v1_5/finetune_task_lora.sh`")
md.append("- Plugin-LoRA: insert into `llava/train/llava_trainer.py` `compute_loss()`")
md.append("- Gamma sweep: gamma in {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}")

summary_path = os.path.join(SCENARIOS_DIR, "reusable_baselines_summary.md")
with open(summary_path, 'w') as f:
    f.write('\n'.join(md))
print(f"Written: {summary_path}")
