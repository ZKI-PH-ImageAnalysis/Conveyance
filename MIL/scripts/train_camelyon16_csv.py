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
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from conveyance.data.camelyon16_csv import Camelyon16CSVDataset
from conveyance.models.abmil import ABMIL   
from conveyance.losses.combined import CombinedBagInstanceLoss 

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

def _safe_auc(scores: List[float], labels: List[int]) -> Optional[float]:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def _fmt(val: Optional[float], fmt: str = ".4f") -> str:
    return f"{val:{fmt}}" if val is not None else "N/A"


def full_metrics(
    preds: List[int], scores: List[float], labels: List[int]
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels) if labels else 0.0
    auc = _safe_auc(scores, labels)
    f1 = float(f1_score(labels, preds, zero_division=0))
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else None
    spec = tn / (tn + fp) if (tn + fp) > 0 else None
    return acc, auc, f1, sens, spec


def make_stratified_folds(labels: List[int], n_folds: int, rng: random.Random) -> List[List[int]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 2**31 - 1))
    return [val_idx.tolist() for _, val_idx in skf.split(range(len(labels)), labels)]

def build_model_and_loss(
    feat_dim: int,
    num_classes: int,
    bag_weight: float,
    inst_weight: float,
    device: torch.device,
    dropout: float = 0.0,
    shared_dim: int | None = None,
    inst_head_num_layers: int = 1,
) -> Tuple[nn.Module, nn.Module]:
    model = ABMIL(
        input_dim=feat_dim,
        num_classes=num_classes,
        dropout=dropout,
        shared_dim=shared_dim,
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
) -> Tuple[float, float, float]:

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

def predict_bags(
    model: nn.Module,
    bags: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[int], List[float]]:

    model.eval()
    preds: List[int] = []
    scores: List[float] = []
    with torch.no_grad():
        for x_t in bags:
            x = x_t.to(device)
            bag_logits, _ = model(x)
            #bag_logits = bag_logits[:, :2]
            probs = torch.softmax(bag_logits, dim=1)
            pred_id = bag_logits.argmax(dim=1)
            pred = int(pred_id.item())
            preds.append(pred)
            scores.append(float(probs[0, 1].item()))
    return preds,scores


def evaluate_bags(
    model: nn.Module,
    bags: List[torch.Tensor],
    labels: List[int],
    device: torch.device,
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    preds, scores = predict_bags(model, bags, device)
    #get all metrics at once
    return full_metrics(preds, scores, labels)



def _rank_normalize(a: np.ndarray) -> np.ndarray:
    return ((rankdata(a) - 1) / len(a)).astype(np.float32)


def _patch_scores_for_bag(
    #only for UNI, for simclr embeddings, no patch labels
    model: nn.Module,
    x: torch.Tensor,
    num_classes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        _, inst_logits = model(x)
        attn = model.attention_weights(x).cpu().numpy()
        p1 = torch.softmax(inst_logits[:, :num_classes], dim=1)
        inst_p1 = p1[:, 1].cpu().numpy()
    return _rank_normalize(attn), inst_p1


def run_fold(
    train_bags: List[torch.Tensor],
    train_labels: List[int],
    val_bags: List[torch.Tensor],
    val_labels: List[int],
    test_bags: List[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
    fold_label: str = "",
) -> Tuple[List[int], List[float], nn.Module]:
    model, criterion = build_model_and_loss(
        feat_dim=train_bags[0].shape[1],
        num_classes=2,
        bag_weight=args.bag_weight,
        inst_weight=args.inst_weight,
        device=device,
        dropout=getattr(args, "dropout", 0.0),
        shared_dim=getattr(args, "shared_dim", None),
        inst_head_num_layers=getattr(args, "inst_head_num_layers", 1),
    )
    #betas=(0.9, 0.999)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if getattr(args, "scheduler_type", "constant") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=getattr(args, "scheduler_eta_min", 1e-6)
        )

    log_every = getattr(args, "log_every", 10)
    prefix = f"[{fold_label}] " if fold_label else "  "

    import copy
    num_bags_list = list(range(len(train_bags)))[:]
    best_val_acc = -1
    best_test_preds: List[int] = []
    best_test_scores: List[float] = []
    best_state_dict = None

    for epoch in range(args.epochs):
        rng.shuffle(num_bags_list)
        loss, bag_l = train_epoch(
            model, criterion, 
            #scheduler,
            optimizer,
            train_bags, train_labels,
            device, args.max_grad_norm,
            num_bags_list=num_bags_list,
        )
        if scheduler is not None:
            scheduler.step()

        val_acc, val_auc, val_f1, val_sens, val_spec = evaluate_bags(
            model, val_bags, val_labels, device
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_test_preds, best_test_scores = predict_bags(model, test_bags, device)
            best_state_dict = copy.deepcopy(model.state_dict())

        if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0 or epoch == args.epochs - 1):
            marker = " *" if is_best else ""
            loss_detail = f"loss={loss:.4f}  bag_l={bag_l:.4f}"
            print(
                f"{prefix}ep {epoch+1:3d}/{args.epochs}  "
                f"{loss_detail}  "
                f"val_acc={_fmt(val_acc)}  "
                f"val_auc={_fmt(val_auc)}  "
                f"val_f1={_fmt(val_f1)}  "
                f"sens={_fmt(val_sens)}  spec={_fmt(val_spec)}"
                f"{marker}",
                flush=True,
            )
    #log after all epochs
    print(
        f"{prefix}-- best val_acc={_fmt(best_val_acc)} --",
        flush=True,
    )
    #return model with best val acc
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return best_test_preds, best_test_scores, model


def run_once(
    train_bags: List[torch.Tensor],
    train_labels: List[int],
    test_bags: List[torch.Tensor],
    test_labels: List[int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    run_idx: int = 0,
    n_runs: int = 1,
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    fold_val_indices = make_stratified_folds(train_labels, args.n_folds, random.Random(seed))
    n_test = len(test_bags)
    vote_counts = [0] * n_test
    score_sums = [0.0] * n_test

    for fold_idx, val_idx_list in enumerate(fold_val_indices):
        fold_seed = seed + fold_idx + 1

        set_seed(fold_seed)
        fold_rng = random.Random(fold_seed)

        val_set = set(val_idx_list)
        train_idx = [i for i in range(len(train_bags)) if i not in val_set]

        #first get/bags lebels per fold
        fold_train_bags = [train_bags[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_bags = [train_bags[i] for i in val_idx_list]
        fold_val_labels = [train_labels[i] for i in val_idx_list]

        fold_label = f"run {run_idx+1}/{n_runs}  fold {fold_idx+1}/{args.n_folds}"
        print(f"{fold_label}  (train={len(fold_train_bags)}  val={len(fold_val_bags)})",
              flush=True)

        test_preds, test_scores, fold_model = run_fold(fold_train_bags, fold_train_labels,
            fold_val_bags, fold_val_labels,
            test_bags,
            args, device, fold_rng,
            fold_label=fold_label,
        )

        for i, (p, s) in enumerate(zip(test_preds, test_scores)):
            vote_counts[i] += p
            score_sums[i] += s


    #average votes and scores over folds, get final metrics
    threshold = args.n_folds / 2.0
    final_preds = [1 if v > threshold else 0 for v in vote_counts]
    mean_scores = [s / args.n_folds for s in score_sums]

    acc, auc, f1, sens, spec = full_metrics(final_preds, mean_scores, test_labels)
    print(
        f"run {run_idx+1}/{n_runs} metrics: "
        f"acc={_fmt(acc)}  auc={_fmt(auc)}  f1={_fmt(f1)}  "
        f"sens={_fmt(sens)}  spec={_fmt(spec)}",
        flush=True,
    )
    #get patch labels, makes sense only for UNI
    #pa_attn, pa_attn_hard, pa_inst, pa_inst_hard, p1_pos, p1_neg = get_patch_labels(args,
    # test_patch_labels,test_labels, fold_model, test_bags, device)

    return acc, auc, f1, sens, spec#pa_attn, pa_attn_hard, pa_inst, pa_inst_hard, p1_pos, p1_neg

def run_experiment(
    manifest_csv: str,
    dataset_root: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    #verbose
    print(f"Camelyon16 CSV")
    print(f"manifest : {manifest_csv}")
    print(f"runs={args.runs}  Folds={args.n_folds}  epochs per fold={args.epochs}")
    print(f"lr={args.lr}  wd={args.weight_decay}  dropout={getattr(args, 'dropout', 0.0)}")

    train_ds = Camelyon16CSVDataset(manifest_csv, dataset_root, split="train")
    test_ds = Camelyon16CSVDataset(manifest_csv, dataset_root, split="test")

    train_bags = [train_ds[i][0] for i in range(len(train_ds))]
    train_labels = list(train_ds.labels)
    test_bags = [test_ds[i][0] for i in range(len(test_ds))]
    test_labels = list(test_ds.labels)

    feat_dim = train_bags[0].shape[1]
    print(f"train bags : {len(train_bags)}"
          f"(pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)})")
    print(f"test  bags : {len(test_bags)}"
          f"(pos={sum(test_labels)},  neg={len(test_labels)-sum(test_labels)})")
    print(f"feature dim: {feat_dim}")

    run_accs: List[float] = []
    run_aucs: List[float] = []
    run_f1s: List[float] = []
    run_senss: List[float] = []
    run_specs: List[float] = []

    for run in range(args.runs):
        seed = args.seed + run * 1000
        set_seed(seed)

        acc, auc, f1, sens, spec = run_once(
            train_bags, train_labels,
            test_bags,  test_labels,
            args, device, seed,
            run_idx=run, n_runs=args.runs,
        )

        run_accs.append(acc)
        #try:
        #    run_aucs.append(auc)
        #    print(sum(run_aucs)/len(run_aucs))
        #except ValueError:
        #    print(f"run {run+1} AUC is undefined (single class in test set), skipping.")
        #try:
        #    run_f1s.append(f1)
        #    print(sum(run_f1s)/len(run_f1s))
        #except ValueError:
        #    print(f"run {run+1} F1 is undefined (single class in test set), skipping.")
        #try:
        #    run_senss.append(sens)
        #    print(sum(run_senss)/len(run_senss))
        #except ValueError:
        #    print(f"run {run+1} Sensitivity is undefined (single class in test set), skipping.")
        #try:
        #    run_specs.append(spec)
        #    print(sum(run_specs)/len(run_specs))
        #except ValueError:
        #    print(f"run {run+1} Specificity is undefined (single class in test set), skipping.")
        if auc is not None: run_aucs.append(auc)
        if f1 is not None: run_f1s.append(f1)
        if sens  is not None: run_senss.append(sens)
        if spec  is not None: run_specs.append(spec)
        #if pa_attn      is not None: run_pa_attn.append(pa_attn)
        #if pa_attn_hard is not None: run_pa_attn_hard.append(pa_attn_hard)
        #if pa_inst      is not None: run_pa_inst.append(pa_inst)
        #if pa_inst_hard is not None: run_pa_inst_hard.append(pa_inst_hard)
        #if p1_pos       is not None: run_p1_pos.append(p1_pos)
        #if p1_neg       is not None: run_p1_neg.append(p1_neg)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camelyon16 CSV MIL - k-fold CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",
        default="/path/to/Camelyon16_MIL/Camelyon16.csv")
    p.add_argument("--dataset-root", dest="dataset_root",
        default="/path/to/Camelyon16_MIL")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5, dest="n_folds")
    p.add_argument("--epochs", type=int, default=100)
    #p.add_argument("--scheduler-type", type=str, default="cosine")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--max-grad-norm", type=float, default=1.0, dest="max_grad_norm")
    p.add_argument("--dropout", type=float, default=0.0)
    #both weights are 1
    p.add_argument("--bag-weight", type=float, default=1.0, dest="bag_weight")
    p.add_argument("--inst-weight", type=float, default=1.0, dest="inst_weight")
    
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
        default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.scheduler_type = "constant"

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    run_experiment(args.manifest, args.dataset_root, args, device)
    print("\ntraining done.")


if __name__ == "__main__":
    main()
