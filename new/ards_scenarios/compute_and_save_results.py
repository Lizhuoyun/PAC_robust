#!/usr/bin/env python3
"""Compute all eval results and write summary files."""
import json, os

OUTPUT_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/eval_outputs"
SCENARIOS_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"

results = {}

# ScienceQA Clean (already computed)
path = os.path.join(OUTPUT_DIR, "scienceqa/clean/llava-v1.5-7b_result.json")
if os.path.exists(path):
    results['sqa_clean'] = json.load(open(path))

# ScienceQA SA (already computed)
path = os.path.join(OUTPUT_DIR, "scienceqa/sa_attack/llava-v1.5-7b_result.json")
if os.path.exists(path):
    results['sqa_sa'] = json.load(open(path))

# ScienceQA PA
pa_file = os.path.join(OUTPUT_DIR, "scienceqa/pa_attack/llava-v1.5-7b.jsonl")
if os.path.exists(pa_file):
    lines = [json.loads(l) for l in open(pa_file)]
    total = correct = img_total = img_correct = 0
    for line in lines:
        attacks = line.get('attack_results', [])
        if not attacks:
            continue
        is_img = '<image>' in line.get('prompt', '')
        total += 1
        if all(a['text'] == a['label'] for a in attacks):
            correct += 1
            if is_img:
                img_correct += 1
        if is_img:
            img_total += 1
    if total > 0:
        results['sqa_pa'] = {
            'acc': correct / total * 100, 'correct': correct, 'count': total,
            'img_acc': img_correct / img_total * 100 if img_total > 0 else None,
            'img_correct': img_correct, 'img_count': img_total
        }
        json.dump(results['sqa_pa'],
                  open(os.path.join(OUTPUT_DIR, "scienceqa/pa_attack/llava-v1.5-7b_result.json"), 'w'), indent=2)

# SEED-Bench
seed_merge = os.path.join(OUTPUT_DIR, "seed_bench/clean/llava-v1.5-7b/merge.jsonl")
seed_anno = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/seed_bench/SEED-Bench.json"
if os.path.exists(seed_merge) and os.path.exists(seed_anno):
    preds = [json.loads(l) for l in open(seed_merge)]
    anno = json.load(open(seed_anno))
    qid2ans = {int(q['question_id']): q['answer'] for q in anno['questions'] if q['data_type'] == 'image'}
    qid2type = {int(q['question_id']): q['question_type_id'] for q in anno['questions'] if q['data_type'] == 'image'}
    type_names = anno.get('question_type', {})

    total = correct = 0
    type_correct = {}
    type_total = {}
    for p in preds:
        qid = p['question_id']
        if qid not in qid2ans:
            continue
        gt = qid2ans[qid]
        pred_text = p['text'].strip()
        answer = pred_text[0] if pred_text and pred_text[0] in 'ABCD' else pred_text
        total += 1
        qt = qid2type[qid]
        type_total[qt] = type_total.get(qt, 0) + 1
        if answer == gt:
            correct += 1
            type_correct[qt] = type_correct.get(qt, 0) + 1

    type_results = {}
    for qt in sorted(type_total.keys()):
        tc = type_correct.get(qt, 0)
        tt = type_total[qt]
        type_results[str(qt)] = {'acc': tc / tt * 100, 'correct': tc, 'count': tt,
                                  'name': type_names.get(str(qt), f"Type {qt}")}

    results['seed_clean'] = {
        'acc': correct / total * 100, 'correct': correct, 'count': total,
        'per_type': type_results
    }
    json.dump(results['seed_clean'],
              open(os.path.join(OUTPUT_DIR, "seed_bench/clean/llava-v1.5-7b/result.json"), 'w'), indent=2)

# Save all results
json.dump(results, open(os.path.join(OUTPUT_DIR, "all_results.json"), 'w'), indent=2)

# Write summary markdown
md_lines = []
md_lines.append("# Reusable Baselines Summary\n")
md_lines.append(f"**Model**: `liuhaotian/llava-v1.5-7b` (full-data baseline)\n")
md_lines.append(f"**Environment**: `/LOCAL2/zhuoyun/PAC_robust/ards_venv` (Python 3.10, torch 2.1.2, transformers 4.37.2)\n")
md_lines.append(f"**ARDS repo**: `/LOCAL2/zhuoyun/PAC_robust/ARDS`\n")
md_lines.append("")
md_lines.append("---\n")
md_lines.append("## Results Table\n")
md_lines.append("| Task | Eval Type | Accuracy | IMG-Accuracy | Samples | Status |")
md_lines.append("|------|-----------|----------|-------------|---------|--------|")

if 'sqa_clean' in results:
    r = results['sqa_clean']
    md_lines.append(f"| ScienceQA | Clean | {r['acc']:.2f}% | - | {r['count']} | ✅ |")

if 'sqa_sa' in results:
    r = results['sqa_sa']
    img_str = f"{r.get('img_acc', 0):.2f}%" if r.get('img_acc') else "-"
    md_lines.append(f"| ScienceQA | SA (Symbol Attack, QWERT) | {r['acc']:.2f}% | {img_str} | {r['count']} | ✅ |")

if 'sqa_pa' in results:
    r = results['sqa_pa']
    img_str = f"{r.get('img_acc', 0):.2f}%" if r.get('img_acc') else "-"
    md_lines.append(f"| ScienceQA | PA (Position Attack) | {r['acc']:.2f}% | {img_str} | {r['count']} | ✅ |")

if 'seed_clean' in results:
    r = results['seed_clean']
    md_lines.append(f"| SEED-Bench (image) | Clean | {r['acc']:.2f}% | - | {r['count']} | ✅ |")

md_lines.append("")
md_lines.append("---\n")
md_lines.append("## ScienceQA Details\n")
md_lines.append("- **Clean**: standard eval with ABCDE options")
md_lines.append("- **SA (Symbol Attack)**: options replaced with QWERT (tests option-letter bias)")
md_lines.append("- **PA (Position Attack)**: all permutations of option positions (tests position bias)")
md_lines.append("  - A sample counts as 'correct' only if ALL permutations are answered correctly\n")

if 'seed_clean' in results and 'per_type' in results['seed_clean']:
    md_lines.append("## SEED-Bench Per-Type Breakdown\n")
    md_lines.append("| Type ID | Category | Accuracy | Samples |")
    md_lines.append("|---------|----------|----------|---------|")
    for qt, info in sorted(results['seed_clean']['per_type'].items(), key=lambda x: int(x[0])):
        md_lines.append(f"| {qt} | {info['name']} | {info['acc']:.1f}% | {info['count']} |")
    md_lines.append("")

md_lines.append("---\n")
md_lines.append("## Output Paths\n")
md_lines.append("| Item | Path |")
md_lines.append("|------|------|")
md_lines.append("| ARDS venv | `/LOCAL2/zhuoyun/PAC_robust/ards_venv` |")
md_lines.append("| Model weights | `/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b` |")
md_lines.append("| SQA Clean answers | `eval_outputs/scienceqa/clean/llava-v1.5-7b.jsonl` |")
md_lines.append("| SQA SA answers | `eval_outputs/scienceqa/sa_attack/llava-v1.5-7b.jsonl` |")
md_lines.append("| SQA PA answers | `eval_outputs/scienceqa/pa_attack/llava-v1.5-7b.jsonl` |")
md_lines.append("| SEED Clean answers | `eval_outputs/seed_bench/clean/llava-v1.5-7b/merge.jsonl` |")
md_lines.append("| All results JSON | `eval_outputs/all_results.json` |")
md_lines.append("")
md_lines.append("---\n")
md_lines.append("## Fixes Applied\n")
md_lines.append("1. **`model_vqa_science_option_attack.py` line 275**: Added missing `--eval-img` argument (仓库 bug)")
md_lines.append("2. **ScienceQA SA**: Generated converted question file `llava_test_CQM-A_convertedABCDE-QWERT.json` (ABCDE→QWERT)")
md_lines.append("3. **SEED-Bench image paths**: Images extracted without `.jpg` extension; fixed question file paths")
md_lines.append("4. **Environment**: Created dedicated venv at `/LOCAL2/zhuoyun/PAC_robust/ards_venv` (transformers 4.37.2 required)\n")
md_lines.append("---\n")
md_lines.append("## Next Steps for Plugin\n")
md_lines.append("All baselines above can be directly used as comparison targets for:")
md_lines.append("1. **普通 LoRA 微调**: use `scripts/v1_5/finetune_task_lora.sh`")
md_lines.append("2. **Plugin-LoRA**: insert into `llava/train/llava_trainer.py` `compute_loss()`")
md_lines.append("3. **Gamma sweep**: γ ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}")

summary_path = os.path.join(SCENARIOS_DIR, "reusable_baselines_summary.md")
with open(summary_path, 'w') as f:
    f.write('\n'.join(md_lines))

print(f"Written: {summary_path}")
print(f"Written: {os.path.join(OUTPUT_DIR, 'all_results.json')}")

# Print summary to stdout
for k, v in results.items():
    if isinstance(v, dict) and 'acc' in v:
        print(f"  {k}: {v['acc']:.2f}% ({v.get('correct','?')}/{v.get('count','?')})")
