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
from train_camelyon16_csv import run_experiment, set_seed 



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
        description="YAML-driven Camelyon16 CSV runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to camelyon16_csv_*.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")
    p.add_argument("--max-grad-norm", type=float, default=None, dest="max_grad_norm")
    #both weights 1
    p.add_argument("--bag-weight", type=float, default=None, dest="bag_weight")
    p.add_argument("--inst-weight", type=float, default=None, dest="inst_weight")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--seed",  type=int,  default=None)
    p.add_argument("--device", default=None)
    #p.add_argument("--save-predictions",default=None)
    return p.parse_args()


def build_args_namespace(cfg: dict, overrides: argparse.Namespace) -> argparse.Namespace:
    """merge YAML config"""
    c16 = cfg.get("camelyon16_csv", {})
    ds_cfg = c16.get("dataset", {})
    train_cfg = c16.get("train", {})
    cl_cfg = c16.get("custom_loss", {})
    sched_cfg = train_cfg.get("scheduler", {})

    def ov(override_val, yaml_val, default):
        return override_val if override_val is not None else (yaml_val if yaml_val is not None else default)

    ns = argparse.Namespace(
        manifest=ds_cfg.get("manifest", "/path/to/Camelyon16_MIL/Camelyon16.csv"),
        dataset_root=ds_cfg.get("dataset_root", "/path/to/Camelyon16_MIL"),
        patch_labels_root=ds_cfg.get("patch_labels_root", None),
        epochs=ov(overrides.epochs, train_cfg.get("epochs"), 100),
        lr=ov(overrides.lr, train_cfg.get("lr"), 2e-4),
        weight_decay=ov(overrides.weight_decay, train_cfg.get("weight_decay"), 1e-4),
        max_grad_norm=ov(overrides.max_grad_norm, train_cfg.get("max_grad_norm"), 1.0),
        device=ov(overrides.device, train_cfg.get("device"), "cuda" if torch.cuda.is_available() else "cpu"),
        scheduler_type=sched_cfg.get("type", "constant"),
        scheduler_eta_min=sched_cfg.get("eta_min", 1e-6),
        runs=ov(overrides.runs, c16.get("runs"), 5),
        n_folds=ov(getattr(overrides, "n_folds", None), c16.get("n_folds"), 5),
        log_every=c16.get("log_every", 10),
        seed=ov(overrides.seed, cfg.get("seed"), 0),
        bag_weight=ov(overrides.bag_weight, _cfg_get(cl_cfg, "bag_loss", "weight"), 1.0),
        inst_weight=ov(overrides.inst_weight, _cfg_get(cl_cfg, "instance_loss", "weight"), 1.0),
        dropout=ov(getattr(overrides, "dropout", None), _cfg_get(c16, "model", "dropout", default=None), 0.0),
        shared_dim=_cfg_get(c16, "model", "shared_dim", default=None),
        inst_head_num_layers=_cfg_get(c16, "model", "inst_head_num_layers", default=1),
    )
    return ns


def main() -> None:
    cli = parse_args()
    cfg = _load_yaml(cli.config)
    args = build_args_namespace(cfg, cli)

    print(f"Config : {cli.config}")
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False).rstrip())
    print(
        f"Resolved:  epochs={args.epochs}  lr={args.lr}  wd={args.weight_decay} "
        f"bag_w={args.bag_weight}  inst_w={args.inst_weight} "
        f"runs={args.runs}  folds={args.n_folds}  seed={args.seed}  "
        f"device={args.device}"
    )

    set_seed(args.seed)
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    run_experiment(args.manifest, args.dataset_root, args, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
