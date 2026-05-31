"""Generate per-dataset benchmark configs.

Produces, for cora / dblp / amazon-computers / roman-empire:
  - <dataset>.yaml              -> gcn & gat baselines (model sweep £[gcn,gat])
  - <dataset>_gcn_modified.yaml -> the tunedGNN strong baseline (GCN_modified)

Each config sweeps all 13 robustness methods x a noise sweep.
Run from repo root:  python configs/_generate_configs.py
Then:                python main.py -c configs/cora.yaml
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_METHODS = ("standard, positive_eigenvalues, gcod, nrgnn, pi_gnn, cr_gnn, "
               "community_defense, rtgnn, graphcleaner, unionnet, gnn_cleaner, "
               "erase, gnnguard")

# Shared method-specific hyperparameters (paper defaults).
METHOD_PARAMS = """\
positive_eigenvalues_params:
  batch_size: 32          # unused (PE is full-batch now); kept for back-compat

nrgnn_params:
  edge_hidden: 64         # official EnyanDai/NRGNN --edge_hidden default
  n_p: 50                 # paper default; NOTE: NRGNN builds O(|unlabeled|x|confident|)
  p_u: 0.8                # edges -> can OOM/oversmooth on large/dense graphs (a finding)
  alpha: 0.03
  beta: 1.0
  t_small: 0.1
  n_n: 50

pi_gnn_params:
  start_epoch: 200        # official TianBian95/pi-gnn default (absolute epoch; needs epochs > 200)
  miself: false
  norm: 10000             # official sentinel (analytic loss_norm); same behaviour as null
  vanilla: false

cr_gnn_params:
  T: 2
  tau: 0.5
  p: 0.8
  alpha: 0.2
  beta: 0.2
  pr: 0.3

community_defense_params:
  community_method: louvain
  num_communities: null
  lambda_comm: 1.0

rtgnn_params:
  edge_hidden: 64
  co_lambda: 0.1
  alpha: 1.0
  th: 0.95                # official GhostQ99/RobustTrainingGNN --th default
  K: 100                  # official --K default
  tau: 0.05
  n_neg: 100
  decay_w: 0.1

graphcleaner_params:
  k: 3                    # official lywww/GraphCleaner --k default
  sample_rate: 0.5
  max_iter_classifier: 500  # mirrors official binary-detector budget (bc_epochs=500)
  held_split: valid

unionnet_params:
  k: 10
  alpha: 0.5
  beta: 1
  feat_norm: true

gnn_cleaner_params:
  label_propagation_iterations: 50   # paper: "execute label propagation 50 iterations each epoch for all experiments"
  similarity_epsilon: 1.0e-8   # float (1e-8 parses as a string in YAML 1.1)

erase_params:
  n_embedding: 512        # official eraseai/erase --n_embedding default
  n_heads: 8
  use_layer_norm: false
  use_residual: false
  use_residual_linear: false
  gam1: 1.0
  gam2: 2.0
  eps: 0.05
  alpha: 0.6
  beta: 0.6
  T: 5                    # official default / Cora (3 is the PubMed value)

gnnguard_params:
  P0: 0.1                 # official mims-harvard/GNNGuard pruning threshold (sim<0.1=0)
  K: 2
  D2: 16
  attention: true
"""

# ── Per-dataset backbone settings for the gcn/gat baselines ──────────────────
# (homophilic: self-loops on, no residual/jk; heterophilic: self-loops OFF,
#  residual + JK on, features not row-normalized.)
BASELINE = {
    "cora": dict(kind="homophilic sparse", normalize="true", self_loop="true",
                 use_residual="false", jk="none", normalization="layer",
                 n_layers=2, hidden=64, dropout=0.5, lr=0.01, wd="5e-4",
                 epochs=300, patience=100),
    "dblp": dict(kind="homophilic citation", normalize="true", self_loop="true",
                 use_residual="false", jk="none", normalization="layer",
                 n_layers=2, hidden=64, dropout=0.5, lr=0.01, wd="5e-4",
                 epochs=300, patience=100),
    "amazon-computers": dict(kind="homophilic DENSE", normalize="true", self_loop="true",
                 use_residual="false", jk="none", normalization="batch",
                 n_layers=2, hidden=64, dropout=0.5, lr=0.01, wd="5e-4",
                 epochs=300, patience=100),
    "roman-empire": dict(kind="HETEROPHILIC", normalize="false", self_loop="false",
                 use_residual="true", jk="cat", normalization="layer",
                 n_layers=5, hidden=64, dropout=0.3, lr=0.01, wd="5e-4",
                 epochs=500, patience=200),
}

# ── Per-dataset GCN_modified (tunedGNN) recipes ──────────────────────────────
GCNMOD = {
    "cora": dict(normalize="true", self_loop="true", pre_linear="false", lin_res="false",
                 mod_norm="none", jk="false", n_layers=3, hidden=512, dropout=0.7,
                 lr=0.001, wd="5e-4", epochs=1000, patience=300),
    "dblp": dict(normalize="true", self_loop="true", pre_linear="false", lin_res="true",
                 mod_norm="ln", jk="false", n_layers=3, hidden=256, dropout=0.5,
                 lr=0.001, wd="5e-4", epochs=1000, patience=300),
    "amazon-computers": dict(normalize="true", self_loop="true", pre_linear="false", lin_res="false",
                 mod_norm="ln", jk="false", n_layers=3, hidden=512, dropout=0.5,
                 lr=0.001, wd="5e-5", epochs=1000, patience=300),
    "roman-empire": dict(normalize="false", self_loop="true", pre_linear="true", lin_res="true",
                 mod_norm="bn", jk="false", n_layers=9, hidden=512, dropout=0.5,
                 lr=0.001, wd="0.0", epochs=2500, patience=2500),
}


def gcod_block(ds):
    """Per-dataset GCOD (author's own method, arXiv:2412.08419) — tuned by VAL
    accuracy under noise, NOT paper Table 4 (which is bs=32, u-lr=1). Honest finding:
    uncertainty_lr depends on per-node signal reliability, not just homo/hetero —
    cora's small/sparse graph gives a trustworthy signal so an aggressive detector
    (0.1) helps; dblp/amazon-computers/roman-empire have entangled or heterophilic
    signals so a slow detector (0.001) is needed (high u_lr collapses them).
    momentum=0 is uniformly best."""
    u_lr = 0.1 if ds == 'cora' else 0.001
    return f"""\
gcod_params:                 # DELIBERATELY TUNED per dataset (do NOT "fix" to paper values)
  batch_size: 64
  uncertainty_lr: {u_lr}        # cora=0.1 (sparse, reliable signal); dblp/amazon/roman=0.001
  kl_start_epoch: 2
  momentum: 0
  temperature: 1.0
  similarity_mode: correction
"""


def baseline_yaml(ds, c):
    return f"""\
# {ds} — gcn & gat baselines ({c['kind']}).  All 13 methods x noise sweep.
# Run:  python main.py -c configs/{ds}.yaml
seed: 42
device: cuda
num_runs: 3
save_checkpoint: false       # NO .pt checkpoints — only JSON logs (experiment.json + training_log.json). Set true if you want best_run_N.pt for later --eval-only.

dataset:
  name: {ds}
  root: data
  normalize: {c['normalize']}        # row-normalize features (false for heterophilic)

noise:
  type: £[clean, uniform, pair]   # add/edit rates & types as needed
  rate: 0.3
  seed: 42

model:
  name: £[gcn, gat]
  hidden_channels: {c['hidden']}
  n_layers: {c['n_layers']}
  dropout: {c['dropout']}
  self_loop: {c['self_loop']}          # FALSE for heterophilic (self-loops dilute neighbour signal)
  use_residual: {c['use_residual']}      # true for heterophilic
  jk: {c['jk']}                # none|cat|max — 'cat' for heterophilic
  normalization: {c['normalization']}     # none|batch|layer (batch is strongest on dense)
  heads: 8
  mlp_layers: 1
  train_eps: false
  use_pe: false
  pe_dim: 8

training:
  method: £[{ALL_METHODS}]
  lr: {c['lr']}
  weight_decay: {c['wd']}
  epochs: {c['epochs']}
  patience: {c['patience']}
  oversmoothing_every: 20
  mode: transductive
  checkpoint_every_epoch: false  # also save a .pt EVERY epoch (huge #files). Keep false; best-epoch is always kept.

{gcod_block(ds)}
{METHOD_PARAMS}"""


def gcnmod_yaml(ds, c):
    return f"""\
# {ds} — GCN_modified (tunedGNN strong baseline, arXiv:2406.08993).
# Separate backbone (learned linear residual + pre_linear + LN/BN). All 13 methods x noise sweep.
# NOTE: hidden={c['hidden']} full-batch can be memory-heavy on dense graphs — use a >=24GB GPU.
# Run:  python main.py -c configs/{ds}_gcn_modified.yaml
seed: 42
device: cuda
num_runs: 3
save_checkpoint: false       # NO .pt checkpoints — only JSON logs (experiment.json + training_log.json). Set true if you want best_run_N.pt for later --eval-only.

dataset:
  name: {ds}
  root: data
  normalize: {c['normalize']}

noise:
  type: £[clean, uniform, pair]
  rate: 0.3
  seed: 42

model:
  name: gcn_modified
  hidden_channels: {c['hidden']}
  n_layers: {c['n_layers']}
  dropout: {c['dropout']}
  self_loop: {c['self_loop']}
  # --- GCN_modified (MPNNs) knobs ---
  pre_linear: {c['pre_linear']}     # project features before message passing
  lin_res: {c['lin_res']}        # learned linear residual conv(x)+W_i*x (key for heterophily)
  mod_norm: {c['mod_norm']}         # none|ln|bn (per-layer normalization)
  jk: {c['jk']}              # additive jumping-knowledge (bool)
  pre_ln: false
  inner_gnn: gcn          # gcn|gat|sage  -> tunedGNN GCN* | GAT* | GraphSAGE*
  heads: 1                # tunedGNN uses heads=1 for GAT*

training:
  method: £[{ALL_METHODS}]
  lr: {c['lr']}
  weight_decay: {c['wd']}
  epochs: {c['epochs']}
  patience: {c['patience']}
  oversmoothing_every: 50
  mode: transductive
  checkpoint_every_epoch: false  # also save a .pt EVERY epoch (huge #files). Keep false; best-epoch is always kept.

{gcod_block(ds)}
{METHOD_PARAMS}"""


def main():
    for ds, c in BASELINE.items():
        with open(os.path.join(HERE, f"{ds}.yaml"), "w") as f:
            f.write(baseline_yaml(ds, c))
        with open(os.path.join(HERE, f"{ds}_gcn_modified.yaml"), "w") as f:
            f.write(gcnmod_yaml(ds, GCNMOD[ds]))
        print(f"wrote configs/{ds}.yaml and configs/{ds}_gcn_modified.yaml")


if __name__ == "__main__":
    main()
