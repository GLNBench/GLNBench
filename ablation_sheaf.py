"""Ablation study for SheafNN components.

Runs the full SheafNN and several ablated variants on a given config,
reporting test accuracy/F1 for each. This file is self-contained and does
NOT modify any existing source code.

Usage:
    python ablation_sheaf.py --config configs/minesweeper_sheafnn.yaml
    python ablation_sheaf.py --config configs/amazon-ratings_sheafnn.yaml --num-runs 3
    python ablation_sheaf.py --config configs/minesweeper_sheafnn.yaml --ablations no_attention no_ego
"""

import argparse
import copy
import gc
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from util.experiment import initialize_experiment
from util.profiling import get_model
from methods.registry import get_helper
from training.training_loop import TrainingLoop
from evaluation.metrics import ClassificationMetrics


# ─── Ablation definitions ────────────────────────────────────────────────────
# Each ablation is a (name, description, config_modifier) triple.
# config_modifier takes a deep-copied config and returns the modified version.
# For ablations that require model-level patching (not just config changes),
# we use a post_model_hook that modifies the instantiated model in-place.


def _ablation_no_attention(config):
    """Disable attention on restriction maps."""
    config['model']['attention'] = False
    return config


def _ablation_no_ego(config):
    """Disable ego skip connection (no concatenation of pre-diffusion features)."""
    config['model']['ego'] = False
    return config


def _ablation_linear_maps(config):
    """Use shared restriction maps across layers (non_linear=False)."""
    config['model']['non_linear'] = False
    return config


def _ablation_scalar_stalk(config):
    """Collapse to scalar sheaf (stalk=1), equivalent to weighted graph diffusion."""
    config['model']['stalk'] = 1
    return config


def _ablation_no_diffusion(config):
    """Skip sheaf diffusion entirely: MLP encoder -> linear head."""
    config['model']['_skip_diffusion'] = True
    return config


def _ablation_identity_maps(config):
    """Use identity restriction maps (reduces sheaf Laplacian to standard graph Laplacian)."""
    config['model']['_identity_maps'] = True
    return config


def _ablation_no_learnable_epsilon(config):
    """Fix diffusion step size to 1 (disable learnable epsilon)."""
    config['model']['_fix_epsilon'] = True
    return config


def _ablation_fewer_layers(config):
    """Use only 2 diffusion layers regardless of the original config."""
    config['model']['n_layers'] = 2
    return config


ABLATIONS = {
    'full': (
        'Full SheafNN (baseline)',
        lambda c: c,
    ),
    'no_attention': (
        'No attention on restriction maps',
        _ablation_no_attention,
    ),
    'no_ego': (
        'No ego skip connection',
        _ablation_no_ego,
    ),
    'linear_maps': (
        'Shared maps across layers (non_linear=False)',
        _ablation_linear_maps,
    ),
    'scalar_stalk': (
        'Scalar stalk (d=1, weighted graph)',
        _ablation_scalar_stalk,
    ),
    'no_diffusion': (
        'No sheaf diffusion (MLP only)',
        _ablation_no_diffusion,
    ),
    'identity_maps': (
        'Identity restriction maps (standard Laplacian)',
        _ablation_identity_maps,
    ),
    'no_learnable_epsilon': (
        'Fixed step size (epsilon=1)',
        _ablation_no_learnable_epsilon,
    ),
    'fewer_layers': (
        'Only 2 diffusion layers',
        _ablation_fewer_layers,
    ),
}


# ─── Model patching ─────────────────────────────────────────────────────────

def _patch_model(model, config):
    """Apply model-level patches based on special config flags."""

    if config['model'].get('_skip_diffusion'):
        original_forward_body = model._forward_body

        def _forward_body_no_diffusion(data):
            x = data.x
            model.N = x.size(0)
            if model.edge_index is None:
                model.edge_index = data.edge_index
            x = F.dropout(x, p=model.dropout_in, training=model.training)
            x = model.MLP_in(x)
            return x

        model._forward_body = _forward_body_no_diffusion

    if config['model'].get('_identity_maps'):
        original_init_maps = model._init_maps

        def _init_maps_identity(edge_index, x, layer=0):
            num_edges = edge_index.size(1)
            eye = torch.eye(model.stalk, device=x.device).unsqueeze(0).expand(num_edges, -1, -1)
            return eye

        model._init_maps = _init_maps_identity

    if config['model'].get('_fix_epsilon'):
        with torch.no_grad():
            for eps in model.epsilons:
                # tanh(large) ~ 1, so 1 + tanh(eps) ~ 2; we want coeff=1
                # Setting eps to 0 gives coeff = 1 + tanh(0) = 1
                eps.fill_(0.0)
                eps.requires_grad_(False)


# ─── Single ablation run ─────────────────────────────────────────────────────

def run_single_ablation(config, run_id=1):
    """Run one training+eval cycle for a given (possibly modified) config.

    Returns dict with test_accuracy, test_f1, val_accuracy, val_f1, time_s.
    """
    init_data = initialize_experiment(config, run_id)
    print("at least here")
    device = init_data['device']
    data = init_data['data_for_training']
    backbone_model = init_data['backbone_model']

  
    _patch_model(backbone_model, config)
    backbone_model.initialize()

    helper = get_helper('SheafNN_Helper')
    loop = TrainingLoop(helper, log_epoch_fn=None, verbose=False)

    t0 = time.perf_counter()
    loop.run(backbone_model, data, config, device, init_data)
    train_time = time.perf_counter() - t0

    state = loop.state
    cls_eval = ClassificationMetrics(average='macro')

    backbone_model.eval()
    with torch.no_grad():
        pred = backbone_model(data).argmax(dim=1)

    test_mask = data.test_mask
    val_mask = data.val_mask

    test_acc = cls_eval.compute_accuracy(pred[test_mask], data.y[test_mask])
    test_f1 = cls_eval.compute_f1(pred[test_mask], data.y[test_mask])
    val_acc = cls_eval.compute_accuracy(pred[val_mask], data.y[val_mask])
    val_f1 = cls_eval.compute_f1(pred[val_mask], data.y[val_mask])

    return {
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'time_s': round(train_time, 2),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SheafNN ablation study')
    parser.add_argument('--config', '-c', required=True,
                        help='Path to a SheafNN YAML config file')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs per ablation (default: 3)')
    parser.add_argument('--ablations', nargs='*', default=None,
                        help=f'Subset of ablations to run. Available: {list(ABLATIONS.keys())}')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON path (default: results/ablation_<dataset>.json)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    assert base_config['model']['name'].lower() == 'sheafnn', \
        "This ablation script is designed for SheafNN configs only."

    ablation_keys = args.ablations if args.ablations else list(ABLATIONS.keys())
    for k in ablation_keys:
        if k not in ABLATIONS:
            raise ValueError(f"Unknown ablation '{k}'. Choose from: {list(ABLATIONS.keys())}")

    dataset_name = base_config['dataset']['name']
    print(f"\n{'='*60}")
    print(f"SheafNN Ablation Study — Dataset: {dataset_name}")
    print(f"Runs per ablation: {args.num_runs}")
    print(f"Ablations: {ablation_keys}")
    print(f"{'='*60}\n")

    requested_device = base_config.get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        base_config['device'] = 'cpu'
        print("[INFO] CUDA not available, falling back to CPU\n")

    results = {}

    for abl_key in ablation_keys:
        description, modifier = ABLATIONS[abl_key]
        print(f"\n--- Ablation: {abl_key} ---")
        print(f"    {description}")

        run_metrics = defaultdict(list)

        for run_id in range(1, args.num_runs + 1):
            abl_config = modifier(copy.deepcopy(base_config))
            print(f"  Run {run_id}/{args.num_runs}...", end=' ', flush=True)

            try:
                result = run_single_ablation(abl_config, run_id=run_id)
                for k, v in result.items():
                    run_metrics[k].append(v)
            except Exception as e:
                print(f"FAILED ({type(e).__name__}: {str(e)[:100]})")
                for k in ('test_accuracy', 'test_f1', 'val_accuracy', 'val_f1', 'time_s'):
                    run_metrics[k].append(float('nan'))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        import numpy as np
        summary = {}
        for k, vals in run_metrics.items():
            clean = [v for v in vals if not (isinstance(v, float) and v != v)]
            if clean:
                summary[f'{k}_mean'] = round(float(np.mean(clean)), 4)
                summary[f'{k}_std'] = round(float(np.std(clean)), 4)
            else:
                summary[f'{k}_mean'] = float('nan')
                summary[f'{k}_std'] = float('nan')
        summary['description'] = description
        summary['runs'] = dict(run_metrics)
        results[abl_key] = summary

    
    output_path = args.output or f"results/ablation_{dataset_name}.json"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'base_config': base_config,
            'num_runs': args.num_runs,
            'results': results,
        }, f, indent=2, default=str)
    print(f"\n[SAVED] {output_path}")


if __name__ == '__main__':
    main()
