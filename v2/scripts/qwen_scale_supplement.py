#!/usr/bin/env python3
"""
Supplementary Qwen scale study launcher.

Goal
====
Extend the existing qwen05b/qwen7b results with intermediate scales
(`qwen15b`, `qwen3b`) so we can compare method gains versus model size.

Phases
======
- `main`: q25-only head-to-head comparison for all core methods
- `alpha_spot`: lightweight alpha sensitivity check for plugin methods only

Design choices
==============
- Default to q25 for the scale comparison. This isolates model-size effects
  from gamma-selection effects because q25 is the established main setting.
- Keep `max_per_gpu=1` by default. This is slower but robust on A100-40GB.
- Reuse existing result CSV naming so completed jobs are auto-skipped.
"""
import argparse
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROJECT = "/LOCAL2/zhuoyun/PAC_robust/v2"
PYTHON = "/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python"
RUN_SCRIPT = os.path.join(PROJECT, "scripts", "run_experiment.py")
LOG_DIR = os.path.join(PROJECT, "logs", "qwen_scale_supplement")
os.makedirs(LOG_DIR, exist_ok=True)

import sys
sys.path.insert(0, PROJECT)
from config import MODEL_REGISTRY

MAIN_METHODS = [
    "base_clean", "base_aug", "plugin",
    "r3f", "r3f_plugin",
    "smart", "smart_plugin",
    "awp", "awp_plugin",
]
ALPHA_SPOT_METHODS = ["plugin", "smart_plugin"]
DEFAULT_MODELS = ["qwen15b", "qwen3b"]
DEFAULT_TASKS = ["arc", "boolq"]
DEFAULT_SEEDS = [42, 123]
DEFAULT_GAMMA_Q = 0.25
DEFAULT_ALPHA_VALUES = [0.05, 0.50]
DEFAULT_TIMEOUT = 21600

MODEL_PROFILES: Dict[str, Dict[str, int]] = {
    "qwen15b": {"batch_size": 8, "grad_accum": 4},
    "qwen3b": {"batch_size": 4, "grad_accum": 8},
}


def _fmt_tag_float(x: float) -> str:
    return f"{x:.4g}".replace("-", "m").replace(".", "p")


@dataclass
class Job:
    name: str
    model_tag: str
    task: str
    method: str
    seed: int
    gamma_q: float = DEFAULT_GAMMA_Q
    alpha: Optional[float] = None
    deps: List[str] = field(default_factory=list)
    phase: str = "main"

    @property
    def result_csv(self) -> str:
        tag = self._result_tag()
        return os.path.join(
            PROJECT, "results", f"{self.model_tag}_{self.task}", f"results_{tag}.csv")

    @property
    def log_path(self) -> str:
        return os.path.join(LOG_DIR, f"{self.name}.log")

    def _result_tag(self) -> str:
        if self.method.endswith("_plugin") or self.method == "plugin":
            tag = f"{self.method}_q{int(self.gamma_q*100):02d}_s{self.seed}"
        else:
            tag = f"{self.method}_s{self.seed}"
        if self.alpha is not None:
            tag += f"_a{_fmt_tag_float(self.alpha)}"
        return tag

    def cmd(self, gpu_id: int) -> List[str]:
        model_path = MODEL_REGISTRY[self.model_tag]
        profile = MODEL_PROFILES.get(self.model_tag, {})
        cmd = [
            PYTHON, RUN_SCRIPT,
            "--method", self.method,
            "--seed", str(self.seed),
            "--gamma_q", str(self.gamma_q),
            "--device", f"cuda:{gpu_id}",
            "--task", self.task,
            "--model_path", model_path,
            "--model_tag", self.model_tag,
        ]
        if "batch_size" in profile:
            cmd += ["--batch_size", str(profile["batch_size"])]
        if "grad_accum" in profile:
            cmd += ["--grad_accum", str(profile["grad_accum"])]
        if self.alpha is not None:
            cmd += ["--alpha", str(self.alpha)]
        return cmd


class GPUPool:
    def __init__(self, gpus: List[int], max_per_gpu: int):
        self.gpus = gpus
        self.max_per_gpu = max_per_gpu
        self._usage = {g: 0 for g in gpus}
        self._lock = threading.Lock()

    def acquire(self) -> int:
        while True:
            with self._lock:
                for gpu_id in self.gpus:
                    if self._usage[gpu_id] < self.max_per_gpu:
                        self._usage[gpu_id] += 1
                        return gpu_id
            time.sleep(2)

    def release(self, gpu_id: int):
        with self._lock:
            self._usage[gpu_id] = max(0, self._usage[gpu_id] - 1)


def _ensure_models_exist(models: List[str]):
    missing = [m for m in models if m not in MODEL_REGISTRY]
    if not missing:
        return
    hints = {
        "qwen15b": "QWEN15B_SNAPSHOT or HF cache models--Qwen--Qwen2.5-1.5B-Instruct",
        "qwen3b": "QWEN3B_SNAPSHOT or HF cache models--Qwen--Qwen2.5-3B-Instruct",
    }
    lines = ["Missing model snapshots for supplementary scale study:"]
    for tag in missing:
        lines.append(f"  - {tag}: set {hints.get(tag, 'a valid snapshot path')}")
    raise SystemExit("\n".join(lines))


def build_jobs(models: List[str], tasks: List[str], seeds: List[int],
               phase: str, alpha_values: List[float],
               alpha_methods: List[str]) -> List[Job]:
    _ensure_models_exist(models)

    jobs: Dict[str, Job] = {}

    def add(job: Job):
        jobs.setdefault(job.name, job)

    for model_tag in models:
        for task in tasks:
            for seed in seeds:
                base_aug_name = f"{model_tag}_{task}_base_aug_s{seed}"

                if phase in ("main", "both"):
                    for method in MAIN_METHODS:
                        if method == "base_aug":
                            name = base_aug_name
                            deps = []
                        elif method.endswith("_plugin") or method == "plugin":
                            name = f"{model_tag}_{task}_{method}_q25_s{seed}"
                            deps = [base_aug_name]
                        else:
                            name = f"{model_tag}_{task}_{method}_s{seed}"
                            deps = []
                        add(Job(
                            name=name,
                            model_tag=model_tag,
                            task=task,
                            method=method,
                            seed=seed,
                            deps=deps,
                            phase="main",
                        ))

                if phase in ("alpha_spot", "both"):
                    add(Job(
                        name=base_aug_name,
                        model_tag=model_tag,
                        task=task,
                        method="base_aug",
                        seed=seed,
                        deps=[],
                        phase="alpha_spot",
                    ))
                    for method in alpha_methods:
                        for alpha in alpha_values:
                            add(Job(
                                name=(f"{model_tag}_{task}_{method}_q25_s{seed}"
                                      f"_a{_fmt_tag_float(alpha)}"),
                                model_tag=model_tag,
                                task=task,
                                method=method,
                                seed=seed,
                                alpha=alpha,
                                deps=[base_aug_name],
                                phase="alpha_spot",
                            ))

    def _priority(job: Job):
        if job.method == "base_aug":
            return (0, job.model_tag, job.task, job.seed, job.method)
        if job.phase == "main" and not (job.method.endswith("_plugin") or job.method == "plugin"):
            return (1, job.model_tag, job.task, job.seed, job.method)
        if job.phase == "main":
            return (2, job.model_tag, job.task, job.seed, job.method, job.alpha or -1)
        return (3, job.model_tag, job.task, job.seed, job.method, job.alpha or -1)

    return sorted(jobs.values(), key=_priority)


def run_jobs(jobs: List[Job], gpus: List[int], max_per_gpu: int,
             timeout: int) -> bool:
    pool = GPUPool(gpus, max_per_gpu)
    done = set()
    failed = set()
    pending = []
    skipped = 0
    lock = threading.Lock()
    threads: List[threading.Thread] = []

    for job in jobs:
        if os.path.exists(job.result_csv):
            done.add(job.name)
            skipped += 1
        else:
            pending.append(job)

    print(f"Total jobs: {len(jobs)}")
    print(f"Skipped existing: {skipped}")
    print(f"Jobs to run: {len(pending)}")
    if not pending:
        return True

    def worker(job: Job, gpu_id: int):
        ok = False
        try:
            with open(job.log_path, "w") as log_f:
                cmd = job.cmd(gpu_id)
                log_f.write(f"CMD: {' '.join(cmd)}\n\n")
                log_f.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=PROJECT,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"  [GPU {gpu_id}] TIMEOUT {job.name}", flush=True)
        except Exception as exc:
            print(f"  [GPU {gpu_id}] ERROR {job.name}: {exc}", flush=True)
        finally:
            with lock:
                if ok:
                    done.add(job.name)
                    print(f"  [GPU {gpu_id}] OK   {job.name}", flush=True)
                else:
                    failed.add(job.name)
                    print(f"  [GPU {gpu_id}] FAIL {job.name}", flush=True)
            pool.release(gpu_id)

    start = time.time()
    while pending or any(t.is_alive() for t in threads):
        launchable = []
        with lock:
            for job in pending:
                if any(dep in failed for dep in job.deps):
                    failed.add(job.name)
                    print(f"  [SKIP] {job.name} (dep failed)", flush=True)
                elif all(dep in done for dep in job.deps):
                    launchable.append(job)
            pending = [job for job in pending if job.name not in failed and job not in launchable]

        for job in launchable:
            gpu_id = pool.acquire()
            print(f"  [GPU {gpu_id}] START {job.name}", flush=True)
            t = threading.Thread(target=worker, args=(job, gpu_id), daemon=True)
            threads.append(t)
            t.start()

        time.sleep(1)

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print(f"\n{'='*68}")
    print(f"Finished in {elapsed/60:.1f} min")
    print(f"Succeeded: {len(done)}/{len(jobs)}")
    print(f"Failed:    {len(failed)}")
    if failed:
        print("Failed jobs:")
        for name in sorted(failed):
            print(f"  - {name}")
    print(f"{'='*68}\n")
    return not failed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="qwen15b,qwen3b")
    p.add_argument("--tasks", default="arc,boolq")
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--phase", choices=["main", "alpha_spot", "both"],
                   default="main")
    p.add_argument("--alpha_values", default="0.05,0.50")
    p.add_argument("--alpha_methods", default="plugin,smart_plugin")
    p.add_argument("--max_per_gpu", type=int, default=1)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    return p.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    alpha_values = [float(a) for a in args.alpha_values.split(",") if a.strip()]
    alpha_methods = [m.strip() for m in args.alpha_methods.split(",") if m.strip()]

    jobs = build_jobs(models, tasks, seeds, args.phase, alpha_values, alpha_methods)
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print(f"Phase: {args.phase}")
    if args.phase in ("alpha_spot", "both"):
        print(f"Alpha spot values: {alpha_values}")
        print(f"Alpha methods: {alpha_methods}")
    ok = run_jobs(jobs, gpus=gpus, max_per_gpu=args.max_per_gpu,
                  timeout=args.timeout)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
