#!/usr/bin/env python3
"""
Parallel experiment launcher for PAC-robust v2.

Distributes all experiments across available GPUs, respecting the
dependency that plugin methods require the base_aug checkpoint.

Strategy
--------
- Wave A (independent): base_clean, base_aug, r3f, smart, awp  (both seeds)
- Wait for base_aug to finish for each seed → gamma calibration auto-runs
  when the first plugin method starts (serialised per seed via a lock file)
- Wave B (plugin): plugin q10/q25/q50, r3f_plugin, smart_plugin, awp_plugin

GPU slots: SLOTS_PER_GPU concurrent jobs per GPU.
With 0.5B model + 4-bit, each job uses ~3-5 GB → A100-40 GB can hold 4+.
We default to 3 slots/GPU = 9 concurrent jobs for safety.

Usage
-----
  cd /LOCAL2/zhuoyun/PAC_robust/v2
  python scripts/parallel_run.py [--gpus 0,1,2] [--slots 3] [--seeds 42,123]
  python scripts/parallel_run.py --p2          # also run awp/awp_plugin
  python scripts/parallel_run.py --gamma_only  # only gamma sweep (q10/q25/q50)
"""
import argparse
import os
import subprocess
import sys
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)


# ── Job definition ─────────────────────────────────────────────────────────

@dataclass
class Job:
    name: str           # unique id
    cmd:  list          # argv (without CUDA_VISIBLE_DEVICES)
    deps: list = field(default_factory=list)  # list of job names that must finish first
    gpu:  int  = -1     # assigned at runtime
    proc: object = None
    done: bool   = False
    failed: bool = False


def _run_exp(method, seed, gamma_q=0.25, device="cuda:0"):
    """Build argv for run_experiment.py."""
    cmd = [
        PYTHON,
        os.path.join(SCRIPT_DIR, "run_experiment.py"),
        "--method",  method,
        "--seed",    str(seed),
        "--gamma_q", str(gamma_q),
        "--device",  device,
    ]
    return cmd


def build_jobs(seeds, include_p2=False, gamma_qs=None):
    """Return ordered list of Job objects with dependency annotations."""
    if gamma_qs is None:
        gamma_qs = [0.10, 0.25, 0.50]

    jobs = []

    # ── Wave A: independent methods ────────────────────────────────────────
    for seed in seeds:
        for method in ["base_clean", "base_aug"]:
            j = Job(
                name=f"{method}_s{seed}",
                cmd=_run_exp(method, seed),
                deps=[],
            )
            jobs.append(j)

        for method in ["r3f", "smart"]:
            j = Job(
                name=f"{method}_s{seed}",
                cmd=_run_exp(method, seed),
                deps=[],
            )
            jobs.append(j)

        if include_p2:
            j = Job(
                name=f"awp_s{seed}",
                cmd=_run_exp("awp", seed),
                deps=[],
            )
            jobs.append(j)

    # ── Wave B: plugin methods (need base_aug for gamma calibration) ────────
    for seed in seeds:
        aug_dep = f"base_aug_s{seed}"

        # plugin (AugStep inner) for each gamma_q
        for q in gamma_qs:
            j = Job(
                name=f"plugin_q{int(q*100):02d}_s{seed}",
                cmd=_run_exp("plugin", seed, gamma_q=q),
                deps=[aug_dep],
            )
            jobs.append(j)

        # r3f_plugin, smart_plugin (only default q25 unless gamma_only sweep)
        for method in ["r3f_plugin", "smart_plugin"]:
            for q in gamma_qs:
                j = Job(
                    name=f"{method}_q{int(q*100):02d}_s{seed}",
                    cmd=_run_exp(method, seed, gamma_q=q),
                    deps=[aug_dep],
                )
                jobs.append(j)

        if include_p2:
            for q in gamma_qs:
                j = Job(
                    name=f"awp_plugin_q{int(q*100):02d}_s{seed}",
                    cmd=_run_exp("awp_plugin", seed, gamma_q=q),
                    deps=[aug_dep],
                )
                jobs.append(j)

    return jobs


# ── GPU slot manager ──────────────────────────────────────────────────────

class GPUPool:
    """Round-robin GPU slot pool. Thread-safe."""

    def __init__(self, gpus: list, slots_per_gpu: int):
        self.gpus      = gpus
        self.slots     = slots_per_gpu
        self._lock     = threading.Lock()
        self._usage    = defaultdict(int)   # gpu_id → active job count

    def acquire(self):
        """Block until a GPU slot is free; return (gpu_id, device_str)."""
        while True:
            with self._lock:
                for g in self.gpus:
                    if self._usage[g] < self.slots:
                        self._usage[g] += 1
                        return g, f"cuda:0"   # inside the subprocess CUDA_VISIBLE_DEVICES=g makes it cuda:0
            time.sleep(2)

    def release(self, gpu_id: int):
        with self._lock:
            self._usage[gpu_id] = max(0, self._usage[gpu_id] - 1)


# ── Parallel scheduler ────────────────────────────────────────────────────

def run_all_jobs(jobs: list, gpus: list, slots_per_gpu: int, log_dir: str):
    """Launch jobs in dependency order, distributing across GPUs."""
    os.makedirs(log_dir, exist_ok=True)
    pool = GPUPool(gpus, slots_per_gpu)

    job_map = {j.name: j for j in jobs}
    pending  = list(jobs)
    active   = []
    done_set = set()
    failed_set = set()
    threads  = []

    def worker(job: Job, gpu_id: int, log_path: str):
        """Run one job in a subprocess on a given GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["TOKENIZERS_PARALLELISM"] = "false"

        print(f"  [LAUNCH] {job.name}  →  GPU {gpu_id}", flush=True)
        with open(log_path, "w") as flog:
            proc = subprocess.Popen(
                job.cmd,
                env=env,
                cwd=BASE_DIR,
                stdout=flog,
                stderr=subprocess.STDOUT,
            )
            job.proc = proc
            ret = proc.wait()

        if ret == 0:
            job.done   = True
            done_set.add(job.name)
            print(f"  [DONE]   {job.name}  (GPU {gpu_id})", flush=True)
        else:
            job.failed = True
            failed_set.add(job.name)
            print(f"  [FAIL]   {job.name}  ret={ret}  log={log_path}", flush=True)

        pool.release(gpu_id)

    start = time.time()
    print(f"\n{'='*60}")
    print(f"  Launching {len(jobs)} jobs on GPUs {gpus}  "
          f"({slots_per_gpu} slots/GPU)")
    print(f"{'='*60}\n")

    while pending or any(t.is_alive() for t in threads):
        # Find pending jobs whose deps are all satisfied
        launchable = [
            j for j in pending
            if all(d in done_set for d in j.deps)
            and not any(d in failed_set for d in j.deps)
        ]

        for job in launchable:
            pending.remove(job)
            gpu_id, device = pool.acquire()
            job.gpu = gpu_id
            # patch device into cmd
            if "--device" in job.cmd:
                di = job.cmd.index("--device")
                job.cmd[di + 1] = device
            log_path = os.path.join(log_dir, f"{job.name}.log")
            t = threading.Thread(target=worker, args=(job, gpu_id, log_path),
                                 daemon=True)
            threads.append(t)
            t.start()

        # Abort jobs whose deps failed (cascade)
        new_pending = []
        for j in pending:
            if any(d in failed_set for d in j.deps):
                print(f"  [SKIP]   {j.name}  (dep failed)", flush=True)
                failed_set.add(j.name)
            else:
                new_pending.append(j)
        pending = new_pending

        time.sleep(1)

    for t in threads:
        t.join()

    elapsed = time.time() - start
    total   = len(jobs)
    n_done  = sum(1 for j in jobs if j.done)
    n_fail  = sum(1 for j in jobs if j.failed)

    print(f"\n{'='*60}")
    print(f"  Finished in {elapsed/60:.1f} min")
    print(f"  Done: {n_done}/{total}   Failed: {n_fail}/{total}")
    if n_fail:
        print(f"  Failed jobs: {[j.name for j in jobs if j.failed]}")
    print(f"{'='*60}\n")
    return n_fail == 0


# ── Aggregate results ─────────────────────────────────────────────────────

def run_aggregate():
    cmd = [PYTHON, os.path.join(SCRIPT_DIR, "aggregate.py")]
    print("\n── Aggregating results ──")
    subprocess.run(cmd, cwd=BASE_DIR, check=False)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Parallel experiment launcher")
    p.add_argument("--gpus",    default="0,1,2",
                   help="Comma-separated GPU IDs (default: 0,1,2)")
    p.add_argument("--slots",   type=int, default=2,
                   help="Max concurrent jobs per GPU (default: 2)")
    p.add_argument("--seeds",   default="42,123",
                   help="Comma-separated seeds (default: 42,123)")
    p.add_argument("--gamma_qs", default="0.10,0.25,0.50",
                   help="Gamma quantiles for plugin methods")
    p.add_argument("--p2",      action="store_true",
                   help="Also run AWP / AWP+Plugin (P2 priority)")
    p.add_argument("--gamma_only", action="store_true",
                   help="Only run gamma sweep (skips non-plugin methods)")
    args = p.parse_args()

    gpus     = [int(g) for g in args.gpus.split(",")]
    seeds    = [int(s) for s in args.seeds.split(",")]
    gamma_qs = [float(q) for q in args.gamma_qs.split(",")]

    log_dir = os.path.join(BASE_DIR, "logs")
    jobs    = build_jobs(seeds, include_p2=args.p2, gamma_qs=gamma_qs)

    if args.gamma_only:
        # Keep only base_aug (for calibration dep) + plugin methods
        jobs = [j for j in jobs
                if "plugin" in j.name or j.name.startswith("base_aug")]

    ok = run_all_jobs(jobs, gpus=gpus, slots_per_gpu=args.slots,
                      log_dir=log_dir)

    run_aggregate()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
