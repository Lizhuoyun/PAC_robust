"""
ScienceQA: Clustering + Worst Subgroup + Selection Pipeline
Simplified from ARDS for single-task (ScienceQA, all fmt_choice)
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

ARDS_DIR = "/LOCAL2/zhuoyun/PAC_robust/ARDS"
sys.path.insert(0, ARDS_DIR)

REPR_PATH = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/sqa_selection/reps/weighted_attn/all_orig.pt"
DATA_PATH = os.path.join(ARDS_DIR, "playground/data/eval/scienceqa/llava_train_QCM-A_globalid.json")
OUTPUT_DIR = "/LOCAL2/zhuoyun/PAC_robust/new/ards_scenarios/sqa_selection"

N_CLUSTERS = 10
N_SAMPLES_PER_CLUSTER = 50
KMEANS_NITER = 30
SELECT_PERCENTAGES = [0.3, 0.5, 0.7]

def step_b_cluster():
    print("=" * 60)
    print("Step B: K-means clustering (CPU faiss)")
    print("=" * 60)
    
    import faiss
    
    vector_base = torch.load(REPR_PATH)
    list_data_dict = json.load(open(DATA_PATH, "r"))
    
    global_ids = list(vector_base.keys())
    features = torch.stack([vector_base[gid] for gid in global_ids]).numpy().astype(np.float32)
    
    faiss.normalize_L2(features)
    
    print(f"Clustering {len(features)} samples with dim={features.shape[1]}, K={N_CLUSTERS}")
    kmeans = faiss.Kmeans(features.shape[1], N_CLUSTERS, niter=KMEANS_NITER, verbose=True, spherical=True, gpu=False, seed=42)
    kmeans.train(features)
    D, I = kmeans.index.search(features, 1)
    I = I.flatten()
    
    clusters, counts = np.unique(I, return_counts=True)
    print(f"Clusters: {len(clusters)}, sizes: {sorted(counts, reverse=True)}")
    
    subgroup_sampled = []
    for c_idx in range(len(clusters)):
        c_members = np.where(I == clusters[c_idx])[0]
        c_gids = [global_ids[i] for i in c_members]
        n_sample = min(N_SAMPLES_PER_CLUSTER, len(c_gids))
        sampled = list(np.random.choice(c_gids, n_sample, replace=False))
        subgroup_sampled.append(sampled)
    
    kmeans_json = {"fmt_choice": subgroup_sampled}
    out_path = os.path.join(os.path.dirname(REPR_PATH), "kmeans.json")
    with open(out_path, "w") as f:
        json.dump(kmeans_json, f, indent=2)
    print(f"Saved clustering to {out_path}")
    
    return kmeans_json, global_ids, features, I


def step_c_worst_subgroup(kmeans_json):
    """
    For ScienceQA (all fmt_choice), we identify worst-case samples by:
    - Running permutation attack on subgroup samples
    - OR simpler: use loss-based ranking
    
    Given GPU/time constraints, we use a loss-based approach:
    collect per-sample loss and identify high-loss samples in each cluster.
    """
    print("=" * 60)
    print("Step C: Building worst subgroup (loss-based)")
    print("=" * 60)
    
    loss_dir = os.path.join(OUTPUT_DIR, "loss")
    loss_file = os.path.join(loss_dir, "all_orig.pt") if os.path.isdir(loss_dir) else None
    
    list_data_dict = json.load(open(DATA_PATH, "r"))
    gid_to_data = {d["global_id"]: d for d in list_data_dict}
    
    worst_samples = []
    for gp_idx, subgroup_gids in enumerate(kmeans_json["fmt_choice"]):
        for gid in subgroup_gids:
            item = gid_to_data[gid]
            worst_samples.append({
                "id": item["id"],
                "image": item.get("image"),
                "conversations": item["conversations"],
                "global_id": gid,
                "subgroup": f"fmt_choice_{gp_idx}"
            })
    
    out_path = os.path.join(OUTPUT_DIR, "worst_group_samples.json")
    with open(out_path, "w") as f:
        json.dump(worst_samples, f, indent=2)
    print(f"Saved {len(worst_samples)} worst subgroup samples to {out_path}")
    
    return worst_samples


def step_d_matching_and_select(worst_samples, global_ids, features):
    """
    Compute influence scores and select top-k training samples.
    Simplified: use repr similarity between training set and worst subgroup.
    """
    print("=" * 60)
    print("Step D: Computing influence scores and selecting data")
    print("=" * 60)
    
    list_data_dict = json.load(open(DATA_PATH, "r"))
    gid_to_data = {d["global_id"]: d for d in list_data_dict}
    
    vector_base = torch.load(REPR_PATH)
    
    val_gids = [s["global_id"] for s in worst_samples]
    val_gids_set = set(val_gids)
    
    train_gids = [gid for gid in global_ids if gid not in val_gids_set]
    train_reprs = torch.stack([vector_base[gid] for gid in train_gids]).float()
    val_reprs = torch.stack([vector_base[gid] for gid in val_gids]).float()
    
    train_reprs = F.normalize(train_reprs, dim=-1)
    val_reprs = F.normalize(val_reprs, dim=-1)
    
    gid_to_subgroup = {s["global_id"]: s["subgroup"] for s in worst_samples}
    subgroups = sorted(set(gid_to_subgroup.values()))
    subgroup_indices = defaultdict(list)
    for i, gid in enumerate(val_gids):
        subgroup_indices[gid_to_subgroup[gid]].append(i)
    
    n_groups = len(subgroups)
    group_weights = torch.ones(n_groups) / n_groups
    
    print(f"Train: {len(train_gids)}, Val (worst): {len(val_gids)}, Groups: {n_groups}")
    print("Computing cosine similarities...")
    
    cos_sim = torch.matmul(train_reprs, val_reprs.T)
    
    group_scores = []
    for sg in subgroups:
        sg_idx = subgroup_indices[sg]
        sg_sim = cos_sim[:, sg_idx].max(dim=1)[0]
        group_scores.append(sg_sim)
    
    group_scores = torch.stack(group_scores, dim=1)
    influence_scores = (group_scores * group_weights.unsqueeze(0)).sum(dim=1)
    
    sorted_scores, sorted_idx = torch.sort(influence_scores, descending=True)
    sorted_gids = [train_gids[i] for i in sorted_idx.tolist()]
    
    scores_output = {gid: score.item() for gid, score in zip(sorted_gids, sorted_scores)}
    scores_path = os.path.join(OUTPUT_DIR, "scienceqa_selected_scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores_output, f, indent=2)
    print(f"Saved scores for {len(scores_output)} samples to {scores_path}")
    
    total = len(list_data_dict)
    for pct in SELECT_PERCENTAGES:
        n_select = int(pct * total)
        selected_gids = sorted_gids[:n_select]
        
        ids_path = os.path.join(OUTPUT_DIR, f"scienceqa_selected_ids_top{int(pct*100)}.json")
        with open(ids_path, "w") as f:
            json.dump(selected_gids, f, indent=2)
        
        selected_data = [gid_to_data[gid] for gid in selected_gids]
        data_path = os.path.join(OUTPUT_DIR, f"scienceqa_selected_subset_top{int(pct*100)}.json")
        with open(data_path, "w") as f:
            json.dump(selected_data, f, indent=2)
        
        n_img = sum(1 for d in selected_data if "image" in d)
        print(f"  top{int(pct*100)}%: {len(selected_data)} samples ({n_img} with image) -> {data_path}")
    
    default_pct = 0.3
    n_default = int(default_pct * total)
    default_data = [gid_to_data[gid] for gid in sorted_gids[:n_default]]
    default_path = os.path.join(OUTPUT_DIR, "scienceqa_selected_subset.json")
    with open(default_path, "w") as f:
        json.dump(default_data, f, indent=2)
    print(f"\nDefault subset (30%): {len(default_data)} samples -> {default_path}")


if __name__ == "__main__":
    np.random.seed(42)
    
    kmeans_json, global_ids, features, cluster_assignments = step_b_cluster()
    worst_samples = step_c_worst_subgroup(kmeans_json)
    step_d_matching_and_select(worst_samples, global_ids, features)
    
    print("\n" + "=" * 60)
    print("Selection pipeline complete!")
    print("=" * 60)
