# PAC_robust

Reproducible research codebase for PAC-Bayes + spectral regularization for worst-class robust risk in LLM adaptation.

## Setup

```bash
pip install -r requirements.txt
```

Ensure NLTK data is available (WordNet + punkt + averaged_perceptron_tagger). If you already have it, point `NLTK_DATA` to the directory.

## Train / Eval (Classification)

```bash
python -m experiments.train_classification --config configs/classification/arc/erm.yaml
python -m experiments.eval_classification --config configs/classification/arc/erm.yaml --ckpt results/arc/erm/seed42
```

## Train / Eval (Generation)

```bash
python -m experiments.train_generation --config configs/generation/gsm8k/nll.yaml
python -m experiments.eval_generation --config configs/generation/gsm8k/nll.yaml --ckpt results/gsm8k/nll/seed42
```

## Aggregation and Plots

```bash
python -m src.analysis.aggregate_runs --input results --out results/summary
python -m src.analysis.plots --matrix results/arc/erm/seed42/matrix.json --out results/arc/erm/seed42/heatmap.png
```

## Reproducing Tables

- Table 1/2: run ARC/CSQA configs in `configs/classification/*` and aggregate metrics.
- Table 3: run GSM8K/StrategyQA configs in `configs/generation/*` and aggregate metrics.
- Table 5: use wall-clock logging (see `logging` in config) to compute overhead.

## Repo Structure

- `src/data`: datasets, prompts, perturbations
- `src/models`: LoRA integration, verbalizer extraction
- `src/losses`: base objectives, R3F, SMART, spectral regularization
- `src/train`: trainers and utilities
- `src/eval`: evaluation and WCR computation
- `src/analysis`: aggregation and plots
- `experiments`: CLI entrypoints
- `configs`: experiment configs
- `tests`: minimal unit tests

Outputs are stored under `results/<exp>/<method>/<seed>/`.

## Baseline Hyperparameters (Reproducibility)

Each run writes `config_resolved.yaml` (after preset resolution and CLI overrides) and logs the resolved baseline hyperparameters to `metrics.json`. This is required for fair comparison across ERM/Augment/R3F/SMART and +Spectral variants.

Markers are fixed and deterministic:
- classification: `Answer: ` (with trailing space)
- generation: `Final answer: ` (with trailing space)

## Prompt Immutability and Field-Level Perturbations

Only data fields are perturbed (question/context/choice texts). The prompt template, markers, instruction text, and option letters are immutable. Perturbed field caches live under `cache/perturb_fields/<dataset>/<split>/<hash>/data.jsonl` and are reused across runs for deterministic experiments.

## Optional Wandb

Wandb is optional and defaults to offline. Local JSONL/JSON outputs are always written.

```bash
export WANDB_MODE=offline
# later: wandb sync ./wandb
```
