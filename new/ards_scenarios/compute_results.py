#!/usr/bin/env python3
import json, os, sys

OUTPUT_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/eval_outputs"

results = {}

# ScienceQA Clean
path = os.path.join(OUTPUT_DIR, "scienceqa/clean/llava-v1.5-7b_result.json")
if os.path.exists(path):
    r = json.load(open(path))
    results['sqa_clean'] = r
    print(f"SQA Clean: {r['acc']:.2f}% ({r['correct']}/{r['count']})")
else:
    print("SQA Clean: MISSING")

# ScienceQA SA
path = os.path.join(OUTPUT_DIR, "scienceqa/sa_attack/llava-v1.5-7b_result.json")
if os.path.exists(path):
    r = json.load(open(path))
    results['sqa_sa'] = r
    print(f"SQA SA: {r['acc']:.2f}% ({r['correct']}/{r['count']})")
else:
    print("SQA SA: MISSING")

# ScienceQA PA
pa_file = os.path.join(OUTPUT_DIR, "scienceqa/pa_attack/llava-v1.5-7b.jsonl")
if os.path.exists(pa_file):
    lines = [json.loads(l) for l in open(pa_file)]
    total = correct = 0
    img_total = img_correct = 0
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
        results['sqa_pa'] = {'acc': correct/total*100, 'correct': correct, 'count': total,
                             'img_acc': img_correct/img_total*100 if img_total > 0 else None}
        print(f"SQA PA: {correct/total*100:.2f}% ({correct}/{total})")
        if img_total > 0:
            print(f"SQA PA IMG: {img_correct/img_total*100:.2f}% ({img_correct}/{img_total})")
else:
    print("SQA PA: MISSING")

# SEED-Bench
seed_merge = os.path.join(OUTPUT_DIR, "seed_bench/clean/llava-v1.5-7b/merge.jsonl")
seed_anno = "/LOCAL2/zhuoyun/PAC_robust/ARDS/playground/data/eval/seed_bench/SEED-Bench.json"
if os.path.exists(seed_merge) and os.path.exists(seed_anno):
    preds = [json.loads(l) for l in open(seed_merge)]
    anno = json.load(open(seed_anno))
    qid2ans = {int(q['question_id']): q['answer'] for q in anno['questions'] if q['data_type'] == 'image'}
    total = correct = 0
    for p in preds:
        qid = p['question_id']
        if qid not in qid2ans:
            continue
        gt = qid2ans[qid]
        pred_text = p['text'].strip()
        answer = pred_text[0] if pred_text and pred_text[0] in 'ABCD' else pred_text
        total += 1
        if answer == gt:
            correct += 1
    results['seed_clean'] = {'acc': correct/total*100, 'correct': correct, 'count': total}
    print(f"SEED Clean: {correct/total*100:.2f}% ({correct}/{total})")
else:
    print("SEED Clean: MISSING")

# Save results
json.dump(results, open(os.path.join(OUTPUT_DIR, "all_results.json"), 'w'), indent=2)
print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'all_results.json')}")
