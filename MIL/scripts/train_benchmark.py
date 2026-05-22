from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#workaround to allow imports from src/ and scripts/ without installing the package
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from conveyance.data.benchmark_mil import BenchmarkMILDataset
from conveyance.models.abmil import ABMIL
from conveyance.losses.combined import CombinedBagInstanceLoss


BENCHMARKS_MIL = ["musk1", "musk2", "elephant", "fox", "tiger"]

def set_seed(seed: int) -> None:
    #random.seed(0)
    #np.random.seed(0)
    #torch.manual_seed(0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_and_loss(
    feat_dim: int,
    num_classes: int,
    bag_weight: float,
    inst_weight: float,
    device: torch.device,
    dropout: float = 0.0,
    backbone_dim: int | None = None,
    inst_head_num_layers: int = 1,
) -> Tuple[nn.Module, nn.Module]:
    model = ABMIL(
        input_dim=feat_dim,
        num_classes=num_classes,
        dropout=dropout,
        backbone_dim=backbone_dim,
        inst_head_num_layers=inst_head_num_layers,
    ).to(device)
    
    #criterion = Conveyance(
    #        alpha=1.0,
    #        beta=1.0,
    #        delta=0.0,
    #        trans_matrix=[[1.0, 1.0], [0.0, 1.0]],
    #        reduction="mean",
    #    )
    
    criterion = CombinedBagInstanceLoss(
        bag_weight=bag_weight,
        instance_weight=inst_weight,
    ).to(device)
    return model, criterion

def train_epoch(
    model: nn.Module,
    criterion,
    #scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    optimizer: optim.Optimizer,
    bags: List[torch.Tensor],
    labels: List[int],
    device: torch.device,
    max_grad_norm: float,
    #add num_bags to to avg (list)
    num_bags_list: List[int],
) -> Tuple[float, float]:

    model.train()
    total_loss = 0.0
    bag_loss_sum = 0.0
    for i in num_bags_list:
        x = bags[i].to(device)
        y = torch.tensor(labels[i], dtype=torch.long, device=device).unsqueeze(0)

        bag_logits, inst_logits = model(x) 

        loss = criterion(bag_logits, inst_logits, y)#, model=model)
        bag_loss_sum += criterion.last_bag_loss_val

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        #if scheduler is not None:
        #    scheduler.step()
        #else:
        #    optimizer.step()
        optimizer.step()
        total_loss += loss.item()

    n = len(num_bags_list)
    return total_loss/n, bag_loss_sum/n
    #return total_loss, bag_loss_sum

def evaluate(
    model: nn.Module,
    bags: List[torch.Tensor],
    labels: List[int],
    device: torch.device,
) -> Tuple[float, float | None]:

    model.eval()
    correct = 0
    scores_list: List[float] = []
    with torch.no_grad():
        for bag_feats, bag_label in zip(bags, labels):
            x = bag_feats.to(device)
            bag_logits, _ = model(x) 
            #bag_logits = bag_logits[:, :2]
            probs = torch.softmax(bag_logits, dim=1)
            pred_id = bag_logits.argmax(dim=1)
            pred = int(pred_id.item())
            if pred == bag_label:
                correct += 1
            scores_list.append(float(probs[0, 1].item()))

    acc = correct/len(bags) if bags else 0.0
    #auc fails for single class batch
    try:
        auc = float(roc_auc_score(labels, scores_list))
    except ValueError:
        auc = None
    return acc, auc

def run_fold(
    train_bags: List[torch.Tensor],
    train_labels: List[int],
    val_bags: List[torch.Tensor],
    val_labels: List[int],
    feat_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
) -> Tuple[float, float, Optional[float]]:

    model, criterion = build_model_and_loss(
        feat_dim=feat_dim,
        num_classes=2,
        bag_weight=args.bag_weight,
        inst_weight=args.inst_weight,
        device=device,
        dropout=getattr(args, "dropout", 0.0),
        backbone_dim=getattr(args, "backbone_dim", None),
        inst_head_num_layers=getattr(args, "inst_head_num_layers", 1),
    )
    #betas=(0.9, 0.999)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc =0
    best_train_acc =0
    best_auc = None
    best_state: Optional[dict] = None
    num_bags_list = list(range(len(train_bags)))[:]

    for epoch in range(args.epochs):
        rng.shuffle(num_bags_list)
        train_epoch(
            model, criterion,
            #scheduler,
            optimizer,
            train_bags, train_labels,
            device, args.max_grad_norm,
            num_bags_list=num_bags_list,
        )
        #train_acc, train_auc = evaluate(model, train_bags, train_labels, device)
        val_acc, val_auc = evaluate(model, val_bags, val_labels, device)
        #if train_acc> best_acc:
        if val_acc > best_acc:
            #update
            best_acc = val_acc
            best_auc = val_auc
            best_train_acc, _ = evaluate(model, train_bags, train_labels, device)

    return best_train_acc, best_acc, best_auc


def run_cv(mat_path: str, args: argparse.Namespace, device: torch.device) -> None:
    name = os.path.splitext(os.path.basename(mat_path))[0].upper()
    print(f"  Dataset : {name}")
    print(f"  Folds   : {args.n_folds}   Runs: {args.runs}   Epochs/fold: {args.epochs}")
    print(f"  LR={args.lr}  wd={args.weight_decay}  bag_w={args.bag_weight}  inst_w={args.inst_weight}")

    #load labels cheaply first
    labels = BenchmarkMILDataset.get_labels_from_mat(mat_path)
    full_ds = BenchmarkMILDataset(mat_path)
    all_bags = [full_ds[i][0] for i in range(len(full_ds))]
    all_labels = [full_ds.labels[i] for i in range(len(full_ds))]
    feat_dim = all_bags[0].shape[1]

    print(f"  Bags: {len(all_bags)}  (pos={sum(all_labels)}, neg={len(all_labels)-sum(all_labels)})  feat_dim={feat_dim}")

    fold_accs: List[float] = []
    fold_aucs: List[float] = []

    for run in range(args.runs):
        seed = args.seed + run * 1000
        set_seed(seed)
        rng = random.Random(seed)

        folds = [val_idx.tolist() for _, val_idx in
                 StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
                 .split(range(len(all_labels)), all_labels)]

        for fold_idx, val_idx_list in enumerate(folds):
            val_set = set(val_idx_list)
            train_idx = [i for i in range(len(all_bags)) if i not in val_set]

            #first get/bags lebels per fold
            train_bags = [all_bags[i] for i in train_idx]
            train_labels_fold = [all_labels[i] for i in train_idx]
            val_bags = [all_bags[i] for i in val_idx_list]
            val_labels_fold = [all_labels[i] for i in val_idx_list]

            fold_seed = seed + fold_idx + 1
            set_seed(fold_seed)
            #shuffle before each fold
            fold_rng = random.Random(fold_seed)

            best_train_acc, best_acc, best_auc = run_fold(
                train_bags, train_labels_fold,
                val_bags, val_labels_fold,
                feat_dim, args, device, fold_rng,
            )
            fold_accs.append(best_acc)
            if best_auc is not None:
                fold_aucs.append(best_auc)

            line_log = (
                f"run={run+1:2d}  fold={fold_idx+1:2d}  "
                f"train_acc={best_train_acc:.4f}  val_acc={best_acc:.4f}"
            )
            if best_auc is not None:
                line_log += f"  val_auc={best_auc:.4f}"
            print(line, flush=True)

    mean_acc = float(np.mean(fold_accs)) *100
    std_acc = float(np.std(fold_accs)) *100
    print(f"\n {name}  acc = {mean_acc:.2f}% ± {std_acc:.2f}%", end="")
    if fold_aucs:
        print(f" bag_AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}", end="")
    print()



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="10-fold X N-run CV on classic MIL benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mat-dir", default="/path/to/mat_files",
                   help="Directory containing the .mat files")
    p.add_argument("--dataset", default="all",
                   help=f"Dataset name (one of {BENCHMARKS_MIL}) or 'all'")
    p.add_argument("--runs", type=int, default=5, help="Number of CV repetitions")
    p.add_argument("--n-folds", type=int, default=10, help="Number of folds")
    p.add_argument("--epochs", type=int, default=40, help="Epochs per fold")
    #p.add_argument("--scheduler-type", type=str, default="cosine")
    p.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=1.0)
    #both weights are 1
    p.add_argument("--bag-weight",   type=float, default=1.0)
    p.add_argument("--inst-weight",  type=float, default=1.0)
    
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    if args.dataset.lower() == "all":
        datasets = BENCHMARKS_MIL
    else:
        datasets = [args.dataset.lower()]

    for ds_name in datasets:
        mat_path = os.path.join(args.mat_dir, f"{ds_name}.mat")
        if not os.path.exists(mat_path):
            mat_path_upper = os.path.join(args.mat_dir, f"{ds_name.upper()}.mat")
            if os.path.exists(mat_path_upper):
                mat_path = mat_path_upper
            else:
                print("mat path not found, skipping.")
                continue
        run_cv(mat_path, args, device)

    print("\ntraining done.")


if __name__ == "__main__":
    main()
