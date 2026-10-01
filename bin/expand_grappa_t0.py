"""Expand a pretrained GrapPA checkpoint with zero weights for the t0 features.

The expanded model reproduces the pretrained predictions exactly until it is
fine-tuned with the new `add_t0` node (+3) and edge (+2) features.

usage: python expand_grappa_t0.py <in.ckpt> <out.ckpt> <old node_feats> [prefix]
"""

import sys

import torch

DN, DE = 3, 2


def insert_zero_cols(weight, pos, num):
    """Insert `num` zero input columns before column `pos` of a Linear weight."""
    zeros = weight.new_zeros(weight.shape[0], num)
    return torch.cat([weight[:, :pos], zeros, weight[:, pos:]], dim=1)


src, dst, n_old = sys.argv[1], sys.argv[2], int(sys.argv[3])
prefix = sys.argv[4] if len(sys.argv) > 4 else ""  # e.g. "grappa_shower." in a full-chain checkpoint
ckpt = torch.load(src, map_location="cpu", weights_only=False)
state = ckpt["state_dict"]

# Input normalization: append neutral entries for the new features
for key in list(state):
    if not key.startswith(prefix) or key.endswith("num_batches_tracked"):
        continue
    for bn, num in ((".node_bn.", DN), (".edge_bn.", DE)):
        if bn in key:
            fill = 1.0 if key.endswith(("weight", "running_var")) else 0.0
            state[key] = torch.cat([state[key], state[key].new_full((num,), fill)])

# First message-passing layer: zero input columns where the new features enter
first = lambda block: f"{prefix}gnn.mp_layers.0.{block}.model.0.weight"
w = state[first("edge_model.mlp")]
w = insert_zero_cols(w, 2 * n_old, DN)  # after the destination-node block
w = insert_zero_cols(w, n_old, DN)      # after the source-node block
state[first("edge_model.mlp")] = torch.cat([w, w.new_zeros(w.shape[0], DE)], dim=1)
for block in ("node_model.message_mlp", "node_model.aggr_mlp"):
    state[first(block)] = insert_zero_cols(state[first(block)], n_old, DN)

# Optimizer moments no longer match the expanded parameters
ckpt.pop("optimizer", None)

for key, value in state.items():
    if key.startswith(prefix) and ("_bn." in key or ".mp_layers.0." in key) and key.endswith(".weight"):
        if "_bn." in key or key.endswith("model.0.weight"):
            print(key, tuple(value.shape))
torch.save(ckpt, dst)
