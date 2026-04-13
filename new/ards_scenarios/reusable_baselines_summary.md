# Reusable Baselines Summary

**Model**: `liuhaotian/llava-v1.5-7b` (pre-trained, zero-shot / full-data baseline)
**Environment**: `/LOCAL2/zhuoyun/PAC_robust/ards_venv`
**ARDS repo**: `/LOCAL2/zhuoyun/PAC_robust/ARDS`

---

## Results Table

| Task | Eval Type | Accuracy | Samples | Status |
|------|-----------|----------|---------|--------|
| ScienceQA | Clean | 70.22% | 4241 | Done |
| ScienceQA | SA (Symbol Attack, ABCDE→QWERT) | 49.61% | 4241 | Done |
| ScienceQA | PA (Position Attack, all perms) | 46.26% | 4241 | Done |
| SEED-Bench (image) | Clean | 66.20% | 14233 | Done |
| SEED-Bench (image) | SA (Symbol Attack, ABCD→QWER) | 58.13% | 14233 | Done |

---

## Key Observations

1. **ScienceQA robustness gap**: Clean 70.22% → SA 49.61% → PA 46.26%
   - SA drops 20.6pp, PA drops 23.96pp
   - Strong option-symbol and position bias
2. **SEED-Bench robustness gap**: Clean 66.20% → SA 58.13%
   - SA drops 8.07pp (less than ScienceQA)
   - More reliance on image understanding than option symbols
3. Both tasks are **discrete multiple-choice**, ideal for Plugin-LoRA evaluation

---

## SEED-Bench Per-Type (Clean vs SA)

| Type | Category | Clean | SA | Delta | Samples |
|------|----------|-------|-----|-------|---------|
| 1 | Scene Understanding | 74.0% | 69.0% | -5.0pp | 3158 |
| 2 | Instance Identity | 68.9% | 60.7% | -8.2pp | 1831 |
| 3 | Instance Attributes | 67.0% | 58.0% | -9.0pp | 4648 |
| 4 | Instance Location | 59.9% | 49.7% | -10.2pp | 978 |
| 5 | Instance Counting | 58.6% | 48.8% | -9.8pp | 2447 |
| 6 | Spatial Relation | 51.4% | 42.0% | -9.4pp | 657 |
| 7 | Instance Interaction | 69.1% | 55.7% | -13.4pp | 97 |
| 8 | Visual Reasoning | 76.7% | 72.5% | -4.2pp | 331 |
| 9 | Text Understanding | 37.2% | 41.9% | +4.7pp | 86 |

---

## Attack Details

- **Clean**: Standard evaluation with original ABCDE/ABCD options
- **SA (Symbol Attack)**: Option labels replaced (ScienceQA: ABCDE→QWERT, SEED: ABCD→QWER)
- **PA (Position Attack)**: All permutations of option positions tested; correct only if ALL perms correct

---

## Output Paths

| Item | Path |
|------|------|
| ARDS venv | `/LOCAL2/zhuoyun/PAC_robust/ards_venv` |
| Model weights | `/LOCAL2/zhuoyun/hf_cache/llava-v1.5-7b` |
| SQA Clean | `eval_outputs/scienceqa/clean/llava-v1.5-7b.jsonl` |
| SQA SA | `eval_outputs/scienceqa/sa_attack/llava-v1.5-7b.jsonl` |
| SQA PA | `eval_outputs/scienceqa/pa_attack/llava-v1.5-7b.jsonl` |
| SEED Clean | `eval_outputs/seed_bench/clean/llava-v1.5-7b/merge.jsonl` |
| SEED SA | `eval_outputs/seed_bench/sa_attack/llava-v1.5-7b/merge.jsonl` |
| All results | `eval_outputs/all_results.json` |

---

## Fixes Applied

1. `model_vqa_science_option_attack.py`: Added missing `--eval-img` argument
2. ScienceQA SA: Generated `llava_test_CQM-A_convertedABCDE-QWERT.json`
3. SEED-Bench: Fixed image paths (no `.jpg` extension), generated QWER-converted files
4. Environment: Created `/LOCAL2/zhuoyun/PAC_robust/ards_venv` (transformers==4.37.2)

---

## Next Steps for Plugin

- All baselines can be used as comparison targets for LoRA / Plugin-LoRA
- **LoRA fine-tune**: `scripts/v1_5/finetune_task_lora.sh`
- **Plugin-LoRA**: insert into `llava/train/llava_trainer.py` `compute_loss()`
- **Gamma sweep**: gamma in {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}
- See `next_step_for_plugin.md` for detailed integration points
