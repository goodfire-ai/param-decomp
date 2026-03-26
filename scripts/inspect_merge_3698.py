import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, "/mnt/polished-lake/home/oli/spd")
from spd.clustering.membership_snapshot import load_membership_snapshot
from spd.clustering.sample_membership import (
    compute_coactivation_matrix_from_csr,
    memberships_to_sample_component_csr,
)

from spd.clustering.compute_costs import compute_merge_costs
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory

hist = MergeHistory.read(Path("/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-651d85c4/history.zip"))
labels = hist.labels
n_comps = len(labels)
alpha = hist.merge_config.alpha

print("Loading 500k snapshot...", flush=True)
snap = load_membership_snapshot(Path("/mnt/polished-lake/artifacts/mechanisms/spd/clustering/harvests/ch-1c1f52d7"))
snap_labels = list(snap.labels)
snap_label_to_idx = {l: i for i, l in enumerate(snap_labels)}
n_samples = snap.n_samples

memberships = snap.to_memberships()
csr = memberships_to_sample_component_csr(memberships)
print("Computing component coactivation...", flush=True)
comp_coact = compute_coactivation_matrix_from_csr(csr)
print(f"Done: {comp_coact.shape}", flush=True)

pre_groups = hist.merges.group_idxs[3697]
unique_groups, inverse = pre_groups.unique(return_inverse=True)
k = len(unique_groups)
print(f"k={k} groups at iter 3697", flush=True)

group_to_labels = {}
group_to_snap = {}
for ci in range(n_comps):
    g = inverse[ci].item()
    group_to_labels.setdefault(g, []).append(labels[ci])
    if labels[ci] in snap_label_to_idx:
        group_to_snap.setdefault(g, []).append(snap_label_to_idx[labels[ci]])

print("Building group activity sparse matrix...", flush=True)
t0 = time.time()
rows_list, cols_list = [], []
for g in range(k):
    idxs = group_to_snap.get(g, [])
    if idxs:
        active = (csr[:, idxs].sum(axis=1) > 0).A1
        sample_idxs = np.where(active)[0]
        rows_list.append(sample_idxs)
        cols_list.append(np.full(len(sample_idxs), g))

all_rows = np.concatenate(rows_list)
all_cols = np.concatenate(cols_list)
G = sparse.csc_matrix((np.ones(len(all_rows), dtype=np.float32), (all_rows, all_cols)), shape=(n_samples, k))
print(f"G: {G.shape}, nnz={G.nnz}, {time.time()-t0:.0f}s", flush=True)

print("Computing G^T @ G...", flush=True)
t0 = time.time()
group_coact = torch.tensor((G.T @ G).toarray(), dtype=torch.float32)
print(f"Group coact: {group_coact.shape}, {time.time()-t0:.0f}s", flush=True)

print("Computing costs...", flush=True)
gm = GroupMerge(group_idxs=inverse, k_groups=k, old_to_new_idx=None)
coact_gpu = group_coact.cuda()
gm_gpu = GroupMerge(group_idxs=inverse.cuda(), k_groups=k, old_to_new_idx=None)
costs = compute_merge_costs(coact=coact_gpu / n_samples, merges=gm_gpu, alpha=alpha)

upper = torch.triu(torch.ones(k, k, dtype=torch.bool, device='cuda'), diagonal=1)
cm = costs.clone()
cm[~upper] = float('inf')
top20_vals, top20_flat = cm.view(-1).topk(20, largest=False)
s_diag = torch.diag(coact_gpu).float()
ranks = gm_gpu.components_per_group.float().to('cuda')

comp_a_idx = labels.index("h.0.attn.v_proj:319")
comp_b_idx = labels.index("h.1.mlp.c_fc:1739")
ga = inverse[comp_a_idx].item()
gb = inverse[comp_b_idx].item()

print(f"\n{'='*120}", flush=True)
print(f"Iter 3698 in c-651d85c4 (alpha={alpha}, exp_rank decay=0.8): k={k}", flush=True)
print(f"Selected: group {ga} ({len(group_to_labels[ga])} comps) x group {gb} ({len(group_to_labels[gb])} comps)", flush=True)
print(f"Selected cost: {costs[ga, gb].item():.6f}, coact={group_coact[ga, gb]:.0f}", flush=True)
print(f"{'='*120}", flush=True)
print(f"{'Rk':>2} {'Cost':>12} {'Coact':>8} {'Fire_A':>8} {'Fire_B':>8} {'RkA':>4} {'RkB':>4} {'Jaccard':>8} {'Sel':>3}  Group A / Group B", flush=True)
print(f"{'-'*120}", flush=True)

for ri in range(20):
    fi = top20_flat[ri].item()
    a, b = fi // k, fi % k
    cab = coact_gpu[a, b].item()
    fa, fb = s_diag[a].item(), s_diag[b].item()
    ra, rb = ranks[a].item(), ranks[b].item()
    union = fa + fb - cab
    jac = cab / union if union > 0 else 0
    sel = "<<<" if {a,b} == {ga,gb} else ""
    la = group_to_labels[a]
    lb = group_to_labels[b]
    da = la[0] if len(la) == 1 else f"[{len(la)}: {la[0]}...]"
    db = lb[0] if len(lb) == 1 else f"[{len(lb)}: {lb[0]}...]"
    print(f"{ri:2d} {top20_vals[ri].item():12.6f} {cab:8.0f} {fa:8.0f} {fb:8.0f} {ra:4.0f} {rb:4.0f} {jac:8.4f} {sel:>3}  {da} / {db}", flush=True)

all_costs = cm.view(-1)
sel_cost = costs[ga, gb].item()
rank_of_selected = (all_costs < sel_cost).sum().item()
print(f"\nSelected pair rank: {rank_of_selected} out of {upper.sum().item()} pairs", flush=True)
