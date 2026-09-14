"""
Fuzzy_neural_network_CP.ipynb
Neuro‑Fuzzy (TSK/ANFIS‑style) Regressor to predict Collapse Potential (%)
from soil experiment features.

What we have in this single script
----------------------------------
• Clean data loading.
• Feature scaling (StandardScaler on X only, fit on train).
• A first‑order Takagi–Sugeno–Kang (TSK) neuro‑fuzzy network implemented in PyTorch:
    - Gaussian membership functions (MFs) per input feature.
    - Grid partition to form rules (R = M^D, where M = MFs per feature, D = #features).
    - Normalized firing strengths and linear consequents per rule.
• Robust initialization of MF centers/sigmas from feature percentiles.
• AdamW optimizer + cosine schedule; early stopping on validation MSE.
• Full evaluation: RMSE, MAE, R², MAPE; residual analysis.
• Visualizations: training curves, parity plot, residual histogram, learning curves.
• Simple permutation feature importance on the validation set.
• Model checkpointing to ./artifacts/.

Notes
-----
• Default #MFs per feature is M=3 → with D=6 features gives R = 3^6 = 729 rules (tractable on CPU/GPU).
• For very modest machine, reduce M to 2.
• Tuning M, batch size, learning rate, epochs at the bottom of the script.
"""

import os
import math
import json
import time
import random
from dataclasses import dataclass
from typing import Tuple, List


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config
DATA_CSV = "/content/drive/MyDrive/PINNs/Suction_vsCP-modified_1.xlsx"
TARGET_COL = "Collapse Potential (%)"
FEATURE_COLS = [
"Suction (kPa)",
"Silica fume (%)",
"Lime (%)",
"Gypsum content (%)",
"Applied vertical stress (kPa)",
"Degree of Saturation (%)",
]

ARTIFACTS_DIR = "/content/drive/MyDrive/NNsGA/FNNs/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utilities

def rmse(y_true, y_pred):
  return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred, eps=1e-8):
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  return np.mean(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), eps, None)))) * 100.0

def metrics(y_true, y_pred):
  return {
    "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "MAE": float(mean_absolute_error(y_true, y_pred)),
    "R2": float(r2_score(y_true, y_pred)),
  }

@dataclass
class TrainConfig:
    mfs_per_feature: int = 3 # M
    batch_size: int = 128
    max_epochs: int = 400
    lr: float       = 1e-3
    weight_decay: float = 1e-4
    patience: int   = 40    # early stopping
    warmup_epochs: int = 10

# Data
class TabDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray):
    self.X = torch.from_numpy(X.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

# Neuro‑Fuzzy (TSK) Model
class TSKFuzzyRegressor(nn.Module):
    """First‑order TSK neuro‑fuzzy network with Gaussian MFs and grid rules.

    Input: x in R^D
    - For each feature j, we have M Gaussian MFs: mu_{j,m}(x_j) = exp(-0.5 * ((x_j - c_{j,m})/s_{j,m})^2)
    - Rules are the Cartesian product of feature MFs → R = M^D rules.
    - Firing strength w_r(x) = Π_j mu_{j, m_j}(x_j)
    - Consequent per rule r: y_r(x) = a_{r,0} + Σ_j a_{r,j} * x_j
    - Output: y(x) = Σ_r [ (w_r / Σ_k w_k) * y_r(x) ]
    """

    def __init__(self, D: int, M: int):
        super().__init__()
        self.D = D
        self.M = M
        self.R = M ** D

        # MF parameters per feature
        # centers: (D, M), sigmas: (D, M) (positivity via softplus)
        self.centers = nn.Parameter(torch.zeros(D, M))
        self.log_sigmas = nn.Parameter(torch.zeros(D, M))  # sigma = softplus(log_sigma)

        # Rule index tensor: (R, D) with values in [0, M-1]
        combos = np.stack(np.meshgrid(*[np.arange(M) for _ in range(D)], indexing='ij'), axis=-1).reshape(-1, D)
        self.register_buffer('rule_index', torch.from_numpy(combos).long())

        # Linear consequents per rule: a0 (bias) + a per feature
        self.consequents = nn.Linear(D, self.R, bias=True)  # will output (N, R) of Σ_j a_{r,j} x_j + a_{r,0}

        # small epsilon to stabilize normalization
        self.eps = 1e-8

    def gaussian_mf(self, x):
        """Compute membership values for all features & MFs.
        x: (N, D)
        return: mu of shape (N, D, M)
        """
        N, D = x.shape
        centers = self.centers  # (D, M)
        sigmas = torch.nn.functional.softplus(self.log_sigmas) + 1e-4  # (D, M)
        # expand for broadcasting
        x_exp = x.unsqueeze(-1)              # (N, D, 1)
        c_exp = centers.unsqueeze(0)        # (1, D, M)
        s_exp = sigmas.unsqueeze(0)         # (1, D, M)
        z = (x_exp - c_exp) / s_exp
        mu = torch.exp(-0.5 * z * z)        # (N, D, M)
        return mu

    def rule_firing(self, mu):
        """Compute rule firing strengths w_r via product across selected MFs.
        mu: (N, D, M)
        returns: w of shape (N, R)
        """
        N, D, M = mu.shape
        gather_list = []
        for j in range(D):
            mu_j = mu[:, j, :]                       # (N, M)
            mu_jg = mu_j.index_select(dim=1, index=self.rule_index[:, j]).view(N, -1)  # (N, R)
            gather_list.append(mu_jg)
        w = torch.ones_like(gather_list[0])
        for g in gather_list:
            w = w * g
        return w  # (N, R)

    def forward(self, x):
        # x: (N, D)
        mu = self.gaussian_mf(x)           # (N, D, M)
        w = self.rule_firing(mu)           # (N, R)
        w_sum = w.sum(dim=1, keepdim=True) # (N, 1)
        beta = w / (w_sum + self.eps)      # normalized firing strengths

        # linear consequents per rule for each sample
        # consequents(x): (N, R) representing Σ_j a_{r,j} x_j + a_{r,0}
        y_lin = self.consequents(x)        # (N, R)
        y = (beta * y_lin).sum(dim=1, keepdim=True)  # (N, 1)
        return y, w_sum

# Initialization helpers

def init_mfs_from_data(model: TSKFuzzyRegressor, X_train: np.ndarray):
    """Initialize MF centers using feature percentiles and sigmas using spread."""
    D = X_train.shape[1]
    M = model.M
    for j in range(D):
        # centers from percentiles between 5th..95th
        perc = np.linspace(5, 95, M)
        c = np.percentile(X_train[:, j], perc)
        # ensure sorted and unique-ish
        c = np.unique(np.round(c, 6))
        if c.size < M:
            # pad by small jitter around median
            med = np.median(X_train[:, j])
            pad = np.linspace(-1, 1, M - c.size) * np.std(X_train[:, j]) * 0.1 + med
            c = np.sort(np.concatenate([c, pad]))
        s = np.full(M, np.std(X_train[:, j]) + 1e-3)
        with torch.no_grad():
            model.centers[j].copy_(torch.from_numpy(c.astype(np.float32)))
            model.log_sigmas[j].copy_(torch.log(torch.from_numpy(s.astype(np.float32))))

# Training loop
def train_model(model, train_loader, val_loader, cfg: TrainConfig):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.max_epochs - cfg.warmup_epochs))

    best_val = float('inf')
    best_state = None
    history = {"train": [], "val": [], "lr": []}
    patience = cfg.patience

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))

        # LR scheduling (simple: step after warmup period)
        if epoch > cfg.warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(current_lr)

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | train MSE={train_loss:.4f} | val MSE={val_loss:.4f} | lr={current_lr:.2e}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return history

# Evaluation helpers
def evaluate(model, X: np.ndarray, y: np.ndarray) -> Tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        y_hat, _ = model(X_t)
        y_hat = y_hat.cpu().numpy().reshape(-1)
    metrics = {
        "RMSE": rmse(y, y_hat),
        "MAE": mean_absolute_error(y, y_hat),
        "R2": r2_score(y, y_hat),
        "MAPE_%": mape(y, y_hat),
    }
    return metrics, y_hat

def plot_training(history: dict, outdir: str):
    plt.figure()
    plt.plot(history["train"], label="Train MSE")
    plt.plot(history["val"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Training/Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=160)
    plt.close()

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, split_name: str):
    plt.figure()
    plt.scatter(y_true, y_pred, s=14, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("Actual Collapse Potential (%)")
    plt.ylabel("Predicted Collapse Potential (%)")
    plt.title(f"Parity Plot — {split_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"parity_{split_name.lower()}.png"), dpi=160)
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, split_name: str):
    res = y_pred - y_true
    plt.figure()
    plt.hist(res, bins=40)
    plt.xlabel("Residual (Pred − True)")
    plt.ylabel("Count")
    plt.title(f"Residuals — {split_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"residuals_{split_name.lower()}.png"), dpi=160)
    plt.close()

def permutation_feature_importance(model, X_val, y_val, scaler: StandardScaler, n_repeats: int = 8):
    # simple, model-agnostic permutation importance
    base_metrics, base_pred = evaluate(model, X_val, y_val)
    base_rmse = base_metrics["RMSE"]
    D = X_val.shape[1]
    importances = np.zeros(D)
    for j in range(D):
        worsens = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            np.random.shuffle(Xp[:, j])
            m, _ = evaluate(model, Xp, y_val)
            worsens.append(m["RMSE"] - base_rmse)
        importances[j] = np.mean(worsens)
    return importances


# Revised main: 70/30 split, leakage-free preprocessing, 4-fold tuning
if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import KFold

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="revised_fnn_results")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_excel(args.data)
    df = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Raw split BEFORE fitting any transform.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    print(f"Development={len(X_dev)}, independent test={len(X_test)}")

    def preprocess_fit(Xtr):
        Xtr = Xtr.copy()
        Xtr[:, 0] = np.log1p(np.clip(Xtr[:, 0], 0, None))
        sc = StandardScaler()
        return sc, sc.fit_transform(Xtr)

    def preprocess_transform(Xv, sc):
        Xv = Xv.copy()
        Xv[:, 0] = np.log1p(np.clip(Xv[:, 0], 0, None))
        return sc.transform(Xv)

    def train_one(Xtr, ytr, Xv, yv, mfs, lr):
        sc, Xtrs = preprocess_fit(Xtr)
        Xvs = preprocess_transform(Xv, sc)
        model = TSKFuzzyRegressor(D=Xtrs.shape[1], M=mfs)
        init_mfs_from_data(model, Xtrs)
        cfg = TrainConfig(mfs_per_feature=mfs, batch_size=64,
                          max_epochs=args.epochs, lr=lr,
                          weight_decay=1e-4, patience=35, warmup_epochs=10)
        train_ds = TabDataset(Xtrs, ytr)
        val_ds = TabDataset(Xvs, yv)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        train_model(model, train_loader, val_loader, cfg)
        pred, _ = evaluate(model, Xvs, yv)
        return model, sc, pred

    # 4-fold tuning of FNN membership-function count and learning rate.
    kf = KFold(n_splits=4, shuffle=True, random_state=SEED)
    candidates = [(m, lr) for m in (2, 3) for lr in (5e-4, 1e-3, 2e-3)]
    rows = []
    for mfs, lr in candidates:
        fold_rmse = []
        for tr, va in kf.split(X_dev):
            model, sc, pred = train_one(X_dev[tr], y_dev[tr], X_dev[va], y_dev[va], mfs, lr)
            fold_rmse.append(rmse(y_dev[va], pred))
        rows.append({"mfs_per_feature": mfs, "learning_rate": lr,
                     "mean_cv_RMSE": float(np.mean(fold_rmse)),
                     "std_cv_RMSE": float(np.std(fold_rmse))})
        print(rows[-1])

    tuning = pd.DataFrame(rows).sort_values("mean_cv_RMSE")
    tuning.to_csv(os.path.join(args.out, "fnn_4fold_tuning.csv"), index=False)
    best = tuning.iloc[0]
    best_mfs = int(best["mfs_per_feature"])
    best_lr = float(best["learning_rate"])
    print("Best FNN:", best_mfs, best_lr)

    # Final FNN: internal validation is used only for early stopping; test remains untouched.
    Xtr, Xval, ytr, yval = train_test_split(X_dev, y_dev, test_size=0.15, random_state=SEED)
    model, sc, _ = train_one(Xtr, ytr, Xval, yval, best_mfs, best_lr)
    Xtest_s = preprocess_transform(X_test, sc)
    test_metrics, test_pred = evaluate(model, Xtest_s, y_test)

    high = y_test > 50
    high_metrics = {"n": int(high.sum()), "threshold": 50.0}
    if high.sum() >= 2:
        high_metrics.update({"high_" + k: v for k, v in metrics(y_test[high], test_pred[high]).items()})

    out = {"best_mfs_per_feature": best_mfs, "best_learning_rate": best_lr,
           "test_metrics": test_metrics, "high_collapse_gt_50": high_metrics,
           "split": "70% development / 30% independent test",
           "preprocessing": "log1p(Suction) then StandardScaler fitted only on training folds"}
    with open(os.path.join(args.out, "fnn_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("FNN test:", test_metrics)
    print("FNN CP > 50%:", high_metrics)
