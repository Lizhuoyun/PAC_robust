#!/usr/bin/env python3
"""
Master parallel launcher for all (model, task, method, seed, gamma_q) combos.

Distributes experiments across GPUs, respecting:
  - 7B models need ~8-10 GB per job → 1 job/GPU for 7B, 2 for 0.5B
  - plugin methods depend on base_aug checkpoint (per model+task+seed)
  - gamma calibration is serialised per (model+task+seed) via file lock

Usage:
  python scripts/run_all_combos.py
  python scripts/run_all_combos.py --models qwen05b --tasks boolq
  python scripts/run_all_combos.py --models qwen7b,mistral7b --tasks arc,boolq
"""
import argparse, os, sys, subprocess, time, threading
from collections import defaultdict
from dataclasses import dataclass, field

PYTHON     = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, BASE_DIR)
from config import MODEL_REGISTRY, TASK_PRESETS


@dataclass
class Job:
    name: str
    cmd:  list
    deps: list = field(default_factory=list)
    gpu:  int  = -1
    slots_needed: int = 1
    done: bool   = False
    failed: bool = False


def _run_exp_cmd(method, seed, gamma_q, model_tag, model_path, task,
                 device="cuda:0", batch_size=None, grad_accum=None):
    cmd = [
        PYTHON, os.path.join(SCRIPT_DIR, "run_experiment.py"),
        "--method",     method,
        "--seed",       str(seed),
        "--gamma_q",    str(gamma_q),
        "--device",     device,
        "--task",       task,
        "--model_path", model_path,
        "--model_tag",  model_tag,
    ]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if grad_accum is not None:
        cmd += ["--grad_accum", str(grad_accum)]
    return cmd


def build_jobs(models, tasks, seeds, gamma_qs, methods):
    jobs = []

    for model_tag in models:
        model_path = MODEL_REGISTRY[model_tag]
        is_big = "7b" in model_tag.lower()

        for task in tasks:
            prefix = f"{model_tag}_{task}"
            bs = 4 if is_big else 8
            ga = 8 if is_big else 4

            # Wave A: independent methods
            for seed in seeds:
                for method in methods:
                    if method.endswith("_plugin") or method == "plugin":
                        continue
                    j = Job(
                        name=f"{prefix}_{method}_s{seed}",
                        cmd=_run_exp_cmd(method, seed, 0.25,
                                         model_tag, model_path, task,
                                         batch_size=bs, grad_accum=ga),
                        deps=[],
                        slots_needed=2 if is_big else 1,
                    )
                    jobs.append(j)

            # Wave B: plugin methods (depend on base_aug)
            for seed in seeds:
                aug_dep = f"{prefix}_base_aug_s{seed}"
                for method in methods:
                    if not (method.endswith("_plugin") or method == "plugin"):
                        continue
                    for q in gamma_qs:
                        j = Job(
                            name=f"{prefix}_{method}_q{int(q*100):02d}_s{seed}",
                            cmd=_run_exp_cmd(method, seed, q,
                                             model_tag, model_path, task,
                                             batch_size=bs, grad_accum=ga),
                            deps=[aug_dep],
                            slots_needed=2 if is_big else 1,
                        )
                        jobs.append(j)

    return jobs


class GPUPool:
    def __init__(self, gpus, slots_per_gpu):
        self.gpus   = gpus
        self.max_slots = slots_per_gpu
        self._lock  = threading.Lock()
        self._usage = defaultdict(int)

    def acquire(self, slots_needed=1):
        while True:
            with self._lock:
                for g in self.gpus:
                    if self._usage[g] + slots_needed <= self.max_slots:
                        self._usage[g] += slots_needed
                        return g, "cuda:0"
            time.sleep(3)

    def release(self, gpu_id, slots_needed=1):
        with self._lock:
            self._usage[gpu_id] = max(0, self._usage[gpu_id] - slots_needed)


def _result_csv_path(job_name, base_dir):
    """Derive the expected result CSV path from a job name like qwen7b_arc_base_clean_s42."""
    parts = job_name.split("_")
    model_tag = parts[0]
    task = parts[1]
    rest = "_".join(parts[2:])
    return os.path.join(base_dir, "results", f"{model_tag}_{task}", f"results_{rest}.csv")


def run_all_jobs(jobs, gpus, slots_per_gpu, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    pool = GPUPool(gpus, slots_per_gpu)

    done_set   = set()
    failed_set = set()

    skipped = 0
    for j in jobs:
        csv = _result_csv_path(j.name, BASE_DIR)
        if os.path.exists(csv):
            j.done = True
            done_set.add(j.name)
            skipped += 1
    if skipped:
        print(f"  [RESUME] {skipped} jobs already have results → skipped")

    pending = [j for j in jobs if not j.done]
    threads    = []

    def worker(job, gpu_id, log_path):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["TOKENIZERS_PARALLELISM"] = "false"
        print(f"  [LAUNCH] {job.name}  →  GPU {gpu_id}  (slots={job.slots_needed})",
              flush=True)
        with open(log_path, "w") as flog:
            proc = subprocess.Popen(
                job.cmd, env=env, cwd=BASE_DIR,
                stdout=flog, stderr=subprocess.STDOUT)
            ret = proc.wait()
        if ret == 0:
            job.done = True
            done_set.add(job.name)
            print(f"  [DONE]   {job.name}  (GPU {gpu_id})", flush=True)
        else:
            job.failed = True
            failed_set.add(job.name)
            print(f"  [FAIL]   {job.name}  ret={ret}  log={log_path}", flush=True)
        pool.release(gpu_id, job.slots_needed)

    start = time.time()
    total = len(jobs)
    print(f"\n{'='*70}")
    print(f"  Launching {total} jobs on GPUs {gpus}  ({slots_per_gpu} slots/GPU)")
    print(f"{'='*70}\n")

    while pending or any(t.is_alive() for t in threads):
        launchable = [
            j for j in pending
            if all(d in done_set for d in j.deps)
            and not any(d in failed_set for d in j.deps)
        ]
        for job in launchable:
            pending.remove(job)
            gpu_id, device = pool.acquire(job.slots_needed)
            job.gpu = gpu_id
            if "--device" in job.cmd:
                di = job.cmd.index("--device")
                job.cmd[di + 1] = device
            log_path = os.path.join(log_dir, f"{job.name}.log")
            t = threading.Thread(target=worker, args=(job, gpu_id, log_path),
                                 daemon=True)
            threads.append(t)
            t.start()

        new_pending = []
        for j in pending:
            if any(d in failed_set for d in j.deps):
                print(f"  [SKIP]   {j.name}  (dep failed)", flush=True)
                failed_set.add(j.name)
            else:
                new_pending.append(j)
        pending = new_pending
        time.sleep(2)

    for t in threads:
        t.join()

    elapsed = time.time() - start
    n_done  = sum(1 for j in jobs if j.done)
    n_fail  = sum(1 for j in jobs if j.failed)
    print(f"\n{'='*70}")
    print(f"  Finished in {elapsed/60:.1f} min")
    print(f"  Done: {n_done}/{total}   Failed: {n_fail}/{total}")
    if n_fail:
        print(f"  Failed: {[j.name for j in jobs if j.failed]}")
    print(f"{'='*70}\n")
    return n_fail == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",   default="qwen05b,qwen7b,mistral7b")
    p.add_argument("--tasks",    default="arc,boolq")
    p.add_argument("--gpus",     default="0,1,2")
    p.add_argument("--slots",    type=int, default=4,
                   help="Total slot capacity per GPU (7B model uses 2 slots, 0.5B uses 1)")
    p.add_argument("--seeds",    default="42,123")
    p.add_argument("--gamma_qs", default="0.10,0.25,0.50")
    p.add_argument("--methods",  default=None,
                   help="Comma-sep methods (default: all P1+P2)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Jobs with existing results CSVs are auto-skipped by run_experiment")
    args = p.parse_args()

    models   = [m.strip() for m in args.models.split(",")]
    tasks    = [t.strip() for t in args.tasks.split(",")]
    gpus     = [int(g)    for g in args.gpus.split(",")]
    seeds    = [int(s)    for s in args.seeds.split(",")]
    gamma_qs = [float(q)  for q in args.gamma_qs.split(",")]

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = [
            "base_clean", "base_aug", "plugin",
            "r3f", "r3f_plugin",
            "smart", "smart_plugin",
            "awp", "awp_plugin",
        ]

    jobs = build_jobs(models, tasks, seeds, gamma_qs, methods)
    log_dir = os.path.join(BASE_DIR, "logs")

    print(f"Models: {models}")
    print(f"Tasks:  {tasks}")
    print(f"Methods: {methods}")
    print(f"Seeds:  {seeds}")
    print(f"Gamma quantiles: {gamma_qs}")
    print(f"Total jobs: {len(jobs)}")

    ok = run_all_jobs(jobs, gpus=gpus, slots_per_gpu=args.slots, log_dir=log_dir)

    # Run aggregate
    agg_cmd = [PYTHON, os.path.join(SCRIPT_DIR, "aggregate.py")]
    print("\n── Aggregating results ──")
    subprocess.run(agg_cmd, cwd=BASE_DIR, check=False)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
