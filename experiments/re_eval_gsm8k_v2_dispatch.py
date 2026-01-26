"""
Re-evaluate GSM8K checkpoints with the updated answer-extraction logic, without
overwriting existing eval_metrics.json files.

This script dispatches eval jobs onto multiple GPUs (by spawning subprocesses
with per-process CUDA_VISIBLE_DEVICES).

Outputs:
  - eval_metrics_v2.json
  - eval_metrics_v2.jsonl
inside each checkpoint directory.
"""

import os
import subprocess
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional


@dataclass
class Job:
    model_name_or_path: str
    ckpt_dir: str
    config_path: str = "configs/generation/gsm8k/nll.yaml"

    @property
    def out_json(self) -> str:
        return os.path.join(self.ckpt_dir, "eval_metrics_v2.json")

    @property
    def out_jsonl(self) -> str:
        return os.path.join(self.ckpt_dir, "eval_metrics_v2.jsonl")


def _infer_model_name(root: str) -> Optional[str]:
    # root examples:
    #   results/gsm8k/qwen25_7b_instruct_gsm8k_suite_v1
    #   results/gsm8k/llama31_8b_instruct_gsm8k_suite_v1
    #   results/gsm8k/mistral7b_instruct_v03_gsm8k_suite_v1
    base = os.path.basename(root)
    if base.startswith("qwen25_7b_instruct"):
        return "Qwen/Qwen2.5-7B-Instruct"
    if base.startswith("llama31_8b_instruct"):
        return "meta-llama/Llama-3.1-8B-Instruct"
    if base.startswith("mistral7b_instruct_v03"):
        return "mistralai/Mistral-7B-Instruct-v0.3"
    return None


def _collect_jobs(results_root: str = "results/gsm8k") -> List[Job]:
    jobs: List[Job] = []
    for suite_root in sorted(glob(os.path.join(results_root, "*_suite_v1"))):
        model = _infer_model_name(suite_root)
        if not model:
            continue
        for method in sorted(os.listdir(suite_root)):
            if method.startswith("_"):
                continue
            ckpt_dir = os.path.join(suite_root, method, "seed42")
            if not os.path.isdir(ckpt_dir):
                continue
            if not os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
                continue
            jobs.append(Job(model_name_or_path=model, ckpt_dir=ckpt_dir))
    return jobs


def _run_job(job: Job, gpu_id: int, python_bin: str, extra_env: Dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(extra_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        python_bin,
        "-m",
        "src.eval.eval_generation",
        "--config",
        job.config_path,
        "--ckpt",
        job.ckpt_dir,
        "--override",
        f"model_name_or_path={job.model_name_or_path}",
        "--override",
        "model.torch_dtype=bf16",
        "--override",
        'lora.target_modules=["q_proj","k_proj","v_proj","o_proj"]',
        "--override",
        f"logging.save_dir={job.ckpt_dir}",
        "--override",
        f"logging.metrics_path={job.out_jsonl}",
        "--override",
        f"logging.final_metrics_path={job.out_json}",
    ]
    print(f"[gpu{gpu_id}] eval {job.model_name_or_path} ckpt={job.ckpt_dir}")
    return subprocess.call(cmd, env=env)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,2", help="Comma-separated GPU indices to use, e.g. '1,2'")
    parser.add_argument("--python", default="/LOCAL2/zhuoyun/Robustfairnessgpu3/venv/bin/python")
    parser.add_argument("--hf_home", default="/LOCAL2/zhuoyun/hf_cache")
    parser.add_argument("--datasets_cache", default="/LOCAL2/zhuoyun/hf_cache/hf_datasets_cache")
    parser.add_argument("--nltk_data", default="/LOCAL2/zhuoyun/nltk_data")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    extra_env = {
        "HF_HOME": args.hf_home,
        "HUGGINGFACE_HUB_CACHE": os.path.join(args.hf_home, "hub"),
        "HF_DATASETS_CACHE": args.datasets_cache,
        "TOKENIZERS_PARALLELISM": "false",
        "NLTK_DATA": args.nltk_data,
    }

    jobs = _collect_jobs()
    # Only run jobs that haven't produced v2 metrics yet.
    jobs = [j for j in jobs if not os.path.exists(j.out_json)]
    print(f"[dispatch] total_jobs={len(jobs)} gpus={gpus}")
    if not jobs:
        return

    # Parallel workers: one process per GPU, each consuming jobs from a shared queue.
    from multiprocessing import Process, Queue

    q: "Queue[Job]" = Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu_id: int) -> None:
        while True:
            try:
                j = q.get_nowait()
            except Exception:
                return
            rc = _run_job(j, gpu_id=gpu_id, python_bin=args.python, extra_env=extra_env)
            if rc != 0:
                print(f"[warn] job failed rc={rc} ckpt={j.ckpt_dir}")

    procs: List[Process] = []
    for gpu_id in gpus:
        p = Process(target=worker, args=(gpu_id,))
        p.daemon = True
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()

