from __future__ import annotations

import argparse
import os
import sys

import yaml

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, os.path.join(_REPO_DIR, "src"))
sys.path.insert(0, _SCRIPTS_DIR)

import torch
from train_benchmark import run_cv, set_seed 


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)
    #load yaml
    #with open(path) as fh:
    #    return yaml.load(fh, Loader=yaml.FullLoader)


def _cfg_get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YAML-driven benchmark CV runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    #p.add_argument("--config-path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")
    p.add_argument("--max-grad-norm", type=float, default=None, dest="max_grad_norm")
    #both weights 1
    p.add_argument("--bag-weight", type=float, default=None, dest="bag_weight")
    p.add_argument("--inst-weight", type=float, default=None, dest="inst_weight")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=None, dest="n_folds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    #p.add_argument("--save-predictions",default=None)
    return p.parse_args()


def build_args_namespace(cfg: dict, overrides: argparse.Namespace) -> argparse.Namespace:
    """merge YAML config"""
    b= cfg.get("benchmark", {})
    train_cfg = b.get("train", {})
    cv_cfg= b.get("cv", {})
    cl_cfg= b.get("custom_loss", {})

    def ov(override_val, yaml_val, default):
        return override_val if override_val is not None else (yaml_val if yaml_val is not None else default)

    ns = argparse.Namespace(
        mat_dir = os.path.dirname(_cfg_get(b, "dataset", "mat_path", default="")),
        dataset = _cfg_get(b, "dataset", "name", default="unknown"),
        epochs=ov(overrides.epochs, train_cfg.get("epochs"), 40),
        lr=ov(overrides.lr, train_cfg.get("lr"), 5e-5),
        weight_decay=ov(overrides.weight_decay, train_cfg.get("weight_decay"), 1e-4),
        max_grad_norm=ov(overrides.max_grad_norm, train_cfg.get("max_grad_norm"), 1.0),
        device=ov(overrides.device, train_cfg.get("device"), "cuda" if torch.cuda.is_available() else "cpu"),
        runs=ov(overrides.runs, cv_cfg.get("runs"), 5),
        n_folds=ov(overrides.n_folds, cv_cfg.get("n_folds"), 10),
        seed=ov(overrides.seed, cfg.get("seed"), 0),
        bag_weight=ov(overrides.bag_weight, _cfg_get(cl_cfg, "bag_loss", "weight"), 1.0),
        inst_weight=ov(overrides.inst_weight, _cfg_get(cl_cfg, "instance_loss", "weight"), 1.0),
        dropout=ov(getattr(overrides, "dropout", None), _cfg_get(b, "model", "dropout", default=None), 0.0),
        backbone_dim=_cfg_get(b, "model", "backbone_dim", default=None),
        inst_head_num_layers=_cfg_get(b, "model", "inst_head_num_layers", default=1),
    )
    ns._mat_path = _cfg_get(b, "dataset", "mat_path", default=None)
    return ns


def main() -> None:
    cli = parse_args()
    cfg = _load_yaml(cli.config)
    args = build_args_namespace(cfg, cli)

    print(f"Config : {cli.config}")
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False).rstrip())
    print(f"Resolved:  epochs={args.epochs}  lr={args.lr}  wd={args.weight_decay}  "
          f"bag_w={args.bag_weight}  inst_w={args.inst_weight}  "
          f"folds={args.n_folds}  runs={args.runs}  seed={args.seed}  device={args.device}")

    print("Starting training...")

    set_seed(args.seed)
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    mat_path = args._mat_path
    if not mat_path or not os.path.exists(mat_path):
        candidates = [
            os.path.join(args.mat_dir, f"{args.dataset}.mat"),
            os.path.join(args.mat_dir, f"{args.dataset.upper()}.mat"),
        ]
        mat_path = next((c for c in candidates if os.path.exists(c)), None)
    if not mat_path or not os.path.exists(mat_path):  # give up
        print(f"ERROR: .mat file not found for dataset '{args.dataset}'.")
        sys.exit(1)

    run_cv(mat_path, args, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
