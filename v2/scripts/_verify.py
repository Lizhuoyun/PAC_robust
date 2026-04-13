#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_all_combos import build_jobs
from config import MODEL_REGISTRY

for tag, path in MODEL_REGISTRY.items():
    exists = os.path.isdir(path)
    print(f"  {tag}: {'OK' if exists else 'MISSING'} -> {path}")

jobs = build_jobs(
    ["qwen05b", "qwen7b", "mistral7b"],
    ["arc", "boolq"],
    [42, 123],
    [0.10, 0.25, 0.50],
    ["base_clean","base_aug","plugin","r3f","r3f_plugin",
     "smart","smart_plugin","awp","awp_plugin"]
)
print(f"Total: {len(jobs)} jobs")
for j in jobs[:2]:
    bs = [j.cmd[i+1] for i, x in enumerate(j.cmd) if x == "--batch_size"]
    print(f"  {j.name}: slots={j.slots_needed} bs={bs}")
for j in [j2 for j2 in jobs if "qwen7b" in j2.name][:2]:
    bs = [j.cmd[i+1] for i, x in enumerate(j.cmd) if x == "--batch_size"]
    print(f"  {j.name}: slots={j.slots_needed} bs={bs}")
