"""Data loader for the UBC PPG cardiac/non-cardiac/occlusion dataset.

Loads pre-processed low-pass (LP) PPG windows from `Code & Data/PPG_Raw_Processed/`
and returns numpy arrays ready for modeling, plus group labels for
participant-disjoint cross-validation.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPG_DIR = os.path.join(REPO_ROOT, "Code & Data", "PPG_Raw_Processed")

CLASSES = ["cardiac", "non_cardiac", "occlusion"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

WIN_SIZE = 512
FS = 50  # Hz


@dataclass
class Dataset:
    X: np.ndarray              # (N, 512) float32, per-window z-scored
    y: np.ndarray              # (N,) int  in {0,1,2}
    groups: np.ndarray         # (N,) str  participant ID for group splits
    placements: np.ndarray     # (N,) str  raw placement code
    class_names: list[str]


def _load_all_lp() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(PPG_DIR, "*_LP.csv")))
    if not files:
        raise FileNotFoundError(f"No *_LP.csv files in {PPG_DIR}")
    return pd.concat([pd.read_csv(fp) for fp in files], ignore_index=True)


def load_dataset(placement: str = "all", zscore: bool = True) -> Dataset:
    """Load LP-filtered PPG windows.

    Args:
        placement: one of {"all", "fingertip", "finger_base", "wrist"}.
            "all" keeps every on-body placement plus dedicated off-body (OB)
            non_cardiac recordings.
        zscore: per-window standardization (mean 0, std 1). Required for CNN
            training because raw values are large unsigned ints.
    """
    df = _load_all_lp()

    placement_groups = {
        "all": None,
        "fingertip": {"MFT_LP", "IFT_LP", "OB"},
        "finger_base": {"IFB_LP", "OB"},
        "wrist": {"WI_LP", "WO_LP", "OB"},
    }
    if placement not in placement_groups:
        raise ValueError(f"placement must be one of {list(placement_groups)}")
    keep = placement_groups[placement]
    if keep is not None:
        df = df[df.placement.isin(keep)].reset_index(drop=True)

    sig_cols = [f"t{i}" for i in range(1, WIN_SIZE + 1)]
    X = df[sig_cols].to_numpy(dtype=np.float32)

    if zscore:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mu) / sd

    y = df.label.map(LABEL_TO_IDX).to_numpy(dtype=np.int64)
    groups = df.participant.to_numpy()
    placements = df.placement.to_numpy()

    return Dataset(X=X, y=y, groups=groups, placements=placements, class_names=CLASSES)


def stratified_split(ds: Dataset, test_size: float = 0.2, val_size: float = 0.1,
                     seed: int = 42):
    """Random stratified split. Same window may share a participant across splits
    (matches the original paper's split — use participant_split for honest eval)."""
    idx = np.arange(len(ds.y))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=ds.y, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size / (1 - test_size),
        stratify=ds.y[train_val_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def participant_split(ds: Dataset, test_size: float = 0.2, val_size: float = 0.1,
                      seed: int = 42):
    """Subject-disjoint split: no patient overlaps train/val/test.

    OB (off-body) windows have no patient identity, so they are split
    randomly and concatenated into each fold proportionally. This avoids
    the pathological case where the single ``OB`` group lands entirely in
    one split."""
    rng = np.random.default_rng(seed)
    is_ob = ds.groups == "OB"
    pat_idx = np.where(~is_ob)[0]
    ob_idx = np.where(is_ob)[0]

    # subject-disjoint for real participants
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_pat, test_pat = next(gss.split(pat_idx, groups=ds.groups[pat_idx]))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size),
                              random_state=seed)
    train_pat, val_pat = next(gss2.split(trval_pat, groups=ds.groups[pat_idx][trval_pat]))

    # random split for OB (no patient correlation possible)
    ob_perm = rng.permutation(ob_idx)
    n_te = int(round(len(ob_perm) * test_size))
    n_va = int(round(len(ob_perm) * val_size))
    ob_test = ob_perm[:n_te]
    ob_val = ob_perm[n_te:n_te + n_va]
    ob_train = ob_perm[n_te + n_va:]

    train_idx = np.concatenate([pat_idx[trval_pat[train_pat]], ob_train])
    val_idx = np.concatenate([pat_idx[trval_pat[val_pat]], ob_val])
    test_idx = np.concatenate([pat_idx[test_pat], ob_test])
    return train_idx, val_idx, test_idx


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights for imbalanced training."""
    counts = np.bincount(y, minlength=len(CLASSES))
    total = counts.sum()
    return {i: total / (len(counts) * c) if c > 0 else 0.0 for i, c in enumerate(counts)}


if __name__ == "__main__":
    for plc in ["all", "fingertip", "finger_base", "wrist"]:
        ds = load_dataset(placement=plc)
        cls_counts = np.bincount(ds.y, minlength=3)
        print(f"\nplacement={plc}: X={ds.X.shape}, "
              f"classes(cardiac/non_cardiac/occlusion)={cls_counts.tolist()}, "
              f"participants={len(set(ds.groups))}")
        tr, va, te = participant_split(ds)
        print(f"  participant split sizes: train={len(tr)} val={len(va)} test={len(te)}")
        print(f"  train participants: {sorted(set(ds.groups[tr]))[:5]}...")
        print(f"  test participants: {sorted(set(ds.groups[te]))}")
