#!/usr/bin/env python3
"""
7B Alpha Sweep Scheduler
========================
Runs alpha sweep for Qwen-7B and Mistral-7B across all plugin methods,
using GPU 0 and GPU 1 only, max 1 concurrent job per GPU.

Total: 4 methods x 3 alpha x 2 tasks x 2 models x 2 seeds = 96 jobs
"""
import os, sys, time, subprocess, threading, queue, json
from dataclasses import dataclass, field
from typing import List

PYTHON = "/LOCAL2/zhuoyun/Robustness_fairness/venv/bin/python"
PROJECT = "/LOCAL2/zhuoyun/PAC_robust/v2"
RUN_SCRIPT = os.path.join(PROJECT, "scripts", "run_experiment.py")
LOG_DIR = os.path.join(PROJECT, "logs", "alpha_sweep_7b")
os.makedirs(LOG_DIR, exist_ok=True)

MODELS = {
    "qwen7b": (
        "/LOCAL2/zhuoyun/hf_cache/models/"
        "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
        "a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "mistral7b": (
        "/LOCAL2/zhuoyun/hf_cache/models/"
        "models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/"
        "3ad372fc79158a2148299e3318516c786aeded6c"
    ),
}

TASKS = ["arc", "boolq"]
METHODS = ["plugin", "r3f_plugin", "smart_plugin", "awp_plugin"]
ALPHAS = [0.3, 1.0, 3.0]
SEEDS = [42, 123]
GAMMA_Q = 0.25
GPUS = [0, 1]
MAX_PER_GPU = 1


@dataclass
class Job:
    model_tag: str
    task: str
    method: str
    alpha: float
    seed: int
    priority: int = 0

    @property
    def tag(self):
        a_str = f"{self.alpha:.4g}".replace("-", "m").replace(".", "p")
        return (f"{self.model_tag}_{self.task}_{self.method}"
                f"_q{int(GAMMA_Q*100):02d}_s{self.seed}_a{a_str}")

    @property
    def result_csv(self):
        a_str = f"{self.alpha:.4g}".replace("-", "m").replace(".", "p")
        result_dir = os.path.join(
            PROJECT, "results", f"{self.model_tag}_{self.task}")
        fname = f"results_{self.method}_q{int(GAMMA_Q*100):02d}_s{self.seed}_a{a_str}.csv"
        return os.path.join(result_dir, fname)

    @property
    def log_path(self):
        return os.path.join(LOG_DIR, f"{self.tag}.log")

    def cmd(self, gpu_id: int) -> List[str]:
        return [
            PYTHON, RUN_SCRIPT,
            "--method", self.method,
            "--seed", str(self.seed),
            "--gamma_q", str(GAMMA_Q),
            "--device", f"cuda:{gpu_id}",
            "--task", self.task,
            "--model_path", MODELS[self.model_tag],
            "--model_tag", self.model_tag,
            "--alpha", str(self.alpha),
        ]


def build_jobs() -> List[Job]:
    """Build job list sorted by priority (lower = higher priority)."""
    jobs = []
    for model_tag in MODELS:
        for task in TASKS:
            for method in METHODS:
                for alpha in ALPHAS:
                    for seed in SEEDS:
                        pri = _priority(model_tag, task, method, alpha)
                        jobs.append(Job(model_tag, task, method, alpha, seed, pri))

    jobs.sort(key=lambda j: (j.priority, j.model_tag, j.task, j.method, j.alpha, j.seed))
    return jobs


def _priority(model_tag, task, method, alpha) -> int:
    """Lower number = higher priority, matching the plan."""
    if task == "arc":
        if model_tag == "mistral7b" and method == "plugin":
            return 0
        if model_tag == "qwen7b" and method == "smart_plugin":
            return 1
        if model_tag == "qwen7b" and method == "plugin":
            return 2
        if model_tag == "mistral7b" and method in ("smart_plugin", "r3f_plugin", "awp_plugin"):
            return 3
        if model_tag == "qwen7b" and method in ("r3f_plugin", "awp_plugin"):
            return 4
    if task == "boolq":
        if method == "plugin":
            return 5
        if method == "smart_plugin":
            return 6
        return 7
    return 8


def gpu_worker(gpu_id: int, job_queue: queue.Queue, stats: dict, lock: threading.Lock):
    """Worker thread for a single GPU. Pulls jobs from shared queue."""
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return

        with lock:
            stats["running"] += 1
            running_now = stats["running"]
            total = stats["total"]
            done = stats["done"]
            skipped = stats["skipped"]
        print(f"  [GPU {gpu_id}] START  {job.tag}  "
              f"(running={running_now}, done={done}, skipped={skipped}/{total})")

        try:
            with open(job.log_path, "w") as log_f:
                log_f.write(f"CMD: {' '.join(job.cmd(gpu_id))}\n\n")
                log_f.flush()
                proc = subprocess.run(
                    job.cmd(gpu_id),
                    stdout=log_f, stderr=subprocess.STDOUT,
                    cwd=PROJECT, timeout=21600,
                )
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
            print(f"  [GPU {gpu_id}] TIMEOUT {job.tag}")
        except Exception as e:
            ok = False
            print(f"  [GPU {gpu_id}] ERROR {job.tag}: {e}")

        with lock:
            stats["running"] -= 1
            stats["done"] += 1
            if ok:
                stats["succeeded"] += 1
            else:
                stats["failed"] += 1
                stats["failed_tags"].append(job.tag)
            done = stats["done"]
            total = stats["total"]
        status = "OK" if ok else "FAIL"
        print(f"  [GPU {gpu_id}] {status}   {job.tag}  "
              f"(done={done}/{total})")
        job_queue.task_done()


def main():
    all_jobs = build_jobs()
    print(f"Total jobs generated: {len(all_jobs)}")

    pending = []
    skipped = 0
    for j in all_jobs:
        if os.path.exists(j.result_csv):
            skipped += 1
        else:
            pending.append(j)

    print(f"Skipping {skipped} already-completed jobs")
    print(f"Jobs to run: {len(pending)}")

    if not pending:
        print("Nothing to do!")
        return

    job_queue = queue.Queue()
    for j in pending:
        job_queue.put(j)

    lock = threading.Lock()
    stats = {
        "total": len(pending) + skipped,
        "done": skipped,
        "skipped": skipped,
        "running": 0,
        "succeeded": skipped,
        "failed": 0,
        "failed_tags": [],
    }

    t_start = time.time()
    threads = []
    for gpu_id in GPUS:
        for slot in range(MAX_PER_GPU):
            t = threading.Thread(target=gpu_worker,
                                 args=(gpu_id, job_queue, stats, lock),
                                 daemon=True)
            t.start()
            threads.append(t)
            time.sleep(2)

    for t in threads:
        t.join()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Alpha sweep 7B complete!")
    print(f"  Total jobs:    {stats['total']}")
    print(f"  Skipped:       {stats['skipped']}")
    print(f"  Succeeded:     {stats['succeeded']}")
    print(f"  Failed:        {stats['failed']}")
    print(f"  Elapsed:       {elapsed/60:.1f} min")
    if stats["failed_tags"]:
        print(f"\n  Failed jobs:")
        for tag in stats["failed_tags"]:
            print(f"    - {tag}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
