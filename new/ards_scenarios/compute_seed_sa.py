#!/usr/bin/env python3
"""Compute SEED SA accuracy and write results to a .py file for reading."""
import json, os

OUTPUT_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/eval_outputs"
SCENARIOS_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios"

seed_sa_merge = os.path.join(OUTPUT_DIR, "seed_bench/sa_attack/llava-v1.5-7b/merge.jsonl")
seed_anno = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/seed_bench/SEED-Bench_convertedABCD-QWER.json"

output_lines = []

if os.path.exists(seed_sa_merge) and os.path.exists(seed_anno):
    preds = [json.loads(l) for l in open(seed_sa_merge)]
    anno = json.load(open(seed_anno))
    
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
            try: qid = int(qid)
            except: continue
        if qid not in qid2ans:
            continue
        gt = qid2ans[qid]
        pred_text = p['text'].strip()
        answer = pred_text[0] if pred_text and pred_text[0] in 'QWER' else pred_text
        total += 1
        qt = qid2type[qid]
        type_t[qt] = type_t.get(qt, 0) + 1
        if answer == gt:
            correct += 1
            type_c[qt] = type_c.get(qt, 0) + 1
    
    acc = correct / total * 100 if total > 0 else 0
    output_lines.append(f"SEED_SA_ACC = {acc}")
    output_lines.append(f"SEED_SA_CORRECT = {correct}")
    output_lines.append(f"SEED_SA_TOTAL = {total}")
    
    per_type = {}
    for qt in sorted(type_t.keys()):
        tc = type_c.get(qt, 0)
        tt = type_t[qt]
        name = type_names.get(str(qt), f"Type {qt}")
        per_type[qt] = (tc/tt*100, tc, tt, name)
        output_lines.append(f"SEED_SA_TYPE_{qt} = ({tc/tt*100:.2f}, {tc}, {tt}, '{name}')")
    
    result = {'acc': acc, 'correct': correct, 'count': total}
    json.dump(result, open(os.path.join(OUTPUT_DIR, "seed_bench/sa_attack/llava-v1.5-7b/result.json"), 'w'), indent=2)
    
    # Update all_results
    all_path = os.path.join(OUTPUT_DIR, "all_results.json")
    if os.path.exists(all_path):
        all_results = json.load(open(all_path))
    else:
        all_results = {}
    all_results['seed_sa'] = result
    json.dump(all_results, open(all_path, 'w'), indent=2)
else:
    output_lines.append(f"SEED_SA_MERGE_EXISTS = {os.path.exists(seed_sa_merge)}")
    output_lines.append(f"SEED_SA_ANNO_EXISTS = {os.path.exists(seed_anno)}")

# Write to .py file for reading
with open(os.path.join(SCENARIOS_DIR, "seed_sa_results.py"), 'w') as f:
    f.write('\n'.join(output_lines) + '\n')

# Also update the summary markdown
all_path = os.path.join(OUTPUT_DIR, "all_results.json")
if os.path.exists(all_path):
    all_results = json.load(open(all_path))
    
    md = []
    md.append("# Reusable Baselines Summary\n")
    md.append("**Model**: `liuhaotian/llava-v1.5-7b` (pre-trained, zero-shot / full-data baseline)")
    md.append("**Environment**: `/LOCAL2/zhuoyun/PAC_robust/ards_venv`")
    md.append("**ARDS repo**: `/LOCAL2/zhuoyun/PAC_robust/ARDS`\n")
    md.append("---\n")
    md.append("## Results Table\n")
    md.append("| Task | Eval Type | Accuracy | Samples | Status |")
    md.append("|------|-----------|----------|---------|--------|")
    
    for key, label_task, label_type in [
        ('sqa_clean', 'ScienceQA', 'Clean'),
        ('sqa_sa', 'ScienceQA', 'SA (Symbol Attack QWERT)'),
        ('sqa_pa', 'ScienceQA', 'PA (Position Attack)'),
        ('seed_clean', 'SEED-Bench (image)', 'Clean'),
        ('seed_sa', 'SEED-Bench (image)', 'SA (Symbol Attack QWER)'),
    ]:
        if key in all_results:
            r = all_results[key]
            md.append(f"| {label_task} | {label_type} | {r['acc']:.2f}% | {r['count']} | Done |")
    
    md.append("")
    md.append("\n---\n")
    md.append("## Attack Details\n")
    md.append("- **Clean**: Standard evaluation with original ABCDE/ABCD options")
    md.append("- **SA (Symbol Attack)**: Option labels replaced (ScienceQA: ABCDE->QWERT, SEED: ABCD->QWER)")
    md.append("- **PA (Position Attack)**: All permutations of option positions (sample correct only if ALL perms correct)")
    
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
    md.append("| SEED SA answers | `eval_outputs/seed_bench/sa_attack/llava-v1.5-7b/merge.jsonl` |")
    md.append("| All results JSON | `eval_outputs/all_results.json` |")
    
    md.append("\n---\n")
    md.append("## Fixes Applied\n")
    md.append("1. `model_vqa_science_option_attack.py`: Added missing `--eval-img` argument")
    md.append("2. ScienceQA SA: Generated `llava_test_CQM-A_convertedABCDE-QWERT.json` (ABCDE->QWERT)")
    md.append("3. SEED-Bench: Fixed image paths (no `.jpg` extension), generated QWER-converted files")
    md.append("4. Environment: Created dedicated venv (`transformers==4.37.2` required by ARDS)")
    
    md.append("\n---\n")
    md.append("## Next Steps for Plugin\n")
    md.append("- All baselines can be used as comparison targets for LoRA / Plugin-LoRA")
    md.append("- LoRA fine-tune: use `scripts/v1_5/finetune_task_lora.sh`")
    md.append("- Plugin-LoRA: insert into `llava/train/llava_trainer.py` `compute_loss()`")
    md.append("- See `next_step_for_plugin.md` for detailed integration points")
    
    with open(os.path.join(SCENARIOS_DIR, "reusable_baselines_summary.md"), 'w') as f:
        f.write('\n'.join(md))
