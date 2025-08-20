"""
Fuzzy_neural_network_CP.ipynb
Neuro‑Fuzzy (TSK/ANFIS‑style) Regressor to predict Collapse Potential (%)
from soil experiment features.

What you get in this single script
----------------------------------
• Clean data loading (expects a CSV path) and train/val/test split (70/15/15).
• Feature scaling (StandardScaler on X only, fit on train).
• A first‑order Takagi–Sugeno–Kang (TSK) neuro‑fuzzy network implemented in PyTorch:
- Gaussian membership functions (MFs) per input feature.
- Grid partition to form rules (R = M^D, where M = MFs per feature, D = #features).
- Normalized firing strengths and linear consequents per rule.
• Robust initialization of MF centers/sigmas from feature percentiles.
• AdamW optimizer + cosine schedule; early stopping on validation MSE.
• Full evaluation: RMSE, MAE, R², MAPE; residual analysis.
• Visualizations: training curves, parity plot, residual histogram, learning curves.
• Optional: simple permutation feature importance on the validation set.
• Model checkpointing to ./artifacts/.

Notes
-----
• Default #MFs per feature is M=3 → with D=6 features gives R = 3^6 = 729 rules (tractable on CPU/GPU).
• If you have a very modest machine, reduce M to 2.
• You can tune M, batch size, learning rate, epochs at the bottom of the script.
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

from google.colab import drive
drive.mount('/content/drive')

!mkdir "/content/drive/MyDrive/NNsGA/FNNs"

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

# Main
if __name__ == "__main__":
    cfg = TrainConfig()

    # 1) Load data
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"File not found at {DATA_CSV}. Please set DATA_CSV to your file path.\n"
            "Expected columns: ['Suction_kPa','SilicaFume_pct','Lime_pct','Gypsum_pct',\n"
            " 'AppliedVerticalStress_kPa','DegreeSaturation_pct','CollapsePotential_pct']"
        )

    df = pd.read_excel(DATA_CSV)
    for col in FEATURE_COLS + [TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Basic sanity prints
    print("Data shape:", df.shape)
    print(df[FEATURE_COLS + [TARGET_COL]].describe())

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Split 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    # Scale features (fit only on train)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Build model
    D = X_train_s.shape[1]
    M = cfg.mfs_per_feature
    model = TSKFuzzyRegressor(D=D, M=M)
    init_mfs_from_data(model, X_train_s)

    # Dataloaders
    train_ds= TabDataset(X_train_s, y_train)
    val_ds  = TabDataset(X_val_s, y_val)
    test_ds = TabDataset(X_test_s, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Train
    start   = time.time()
    history = train_model(model, train_loader, val_loader, cfg)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed/60:.1f} min. Best val MSE: {min(history['val']):.4f}")

    # Save artifacts
    model_path = os.path.join(ARTIFACTS_DIR, "tsk_fuzzy_regressor.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
    }, model_path)
    with open(os.path.join(ARTIFACTS_DIR, "scaler.json"), "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)

    # Evaluate
    train_metrics, yhat_train = evaluate(model, X_train_s, y_train)
    val_metrics, yhat_val   = evaluate(model, X_val_s, y_val)
    test_metrics, yhat_test = evaluate(model, X_test_s, y_test)

    # Convert float32 values to standard floats for JSON serialization
    train_metrics_json = {k: float(v) for k, v in train_metrics.items()}
    val_metrics_json   = {k: float(v) for k, v in val_metrics.items()}
    test_metrics_json  = {k: float(v) for k, v in test_metrics.items()}

    print("\nMetrics (Train)", json.dumps(train_metrics_json, indent=2))
    print("Metrics (Val)  ", json.dumps(val_metrics_json, indent=2))
    print("Metrics (Test) ", json.dumps(test_metrics_json, indent=2))

    # Plots
    plot_training(history, ARTIFACTS_DIR)
    plot_parity(y_train, yhat_train, ARTIFACTS_DIR, "Train")
    plot_parity(y_val, yhat_val, ARTIFACTS_DIR, "Val")
    plot_parity(y_test, yhat_test, ARTIFACTS_DIR, "Test")
    plot_residuals(y_train, yhat_train, ARTIFACTS_DIR, "Train")
    plot_residuals(y_val, yhat_val, ARTIFACTS_DIR, "Val")
    plot_residuals(y_test, yhat_test, ARTIFACTS_DIR, "Test")

    # Permutation importance (on validation split)
    print("\nComputing permutation feature importance on validation split…")
    importances = permutation_feature_importance(model, X_val_s, y_val, scaler, n_repeats=5)
    imp_df      = pd.DataFrame({"feature": FEATURE_COLS, "importance_RMSE_worsening": importances})
    imp_df      = imp_df.sort_values("importance_RMSE_worsening", ascending=False)
    imp_path    = os.path.join(ARTIFACTS_DIR, "permutation_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print("Permutation importance saved to:", imp_path)

    # quick bar plot
    plt.figure()
    plt.barh(imp_df["feature"], imp_df["importance_RMSE_worsening"])  
    plt.gca().invert_yaxis()
    plt.xlabel("ΔRMSE after permutation (Validation)")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "perm_importance.png"), dpi=160)
    plt.close()

    # Save predictions
    pred_df = pd.DataFrame({
        "split": ["train"] * len(y_train) + ["val"] * len(y_val) + ["test"] * len(y_test),
        "y_true": np.concatenate([y_train, y_val, y_test]),
        "y_pred": np.concatenate([yhat_train, yhat_val, yhat_test]),
    })
    pred_path = os.path.join(ARTIFACTS_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print("Predictions saved to:", pred_path)

    # Final summary
    summary = {
        "train": train_metrics_json,
        "val": val_metrics_json,
        "test": test_metrics_json,
        "artifacts": {
            "model": model_path,
            "loss_curves": os.path.join(ARTIFACTS_DIR, "loss_curves.png"),
            "parity_train": os.path.join(ARTIFACTS_DIR, "parity_train.png"),
            "parity_val": os.path.join(ARTIFACTS_DIR, "parity_val.png"),
            "parity_test": os.path.join(ARTIFACTS_DIR, "parity_test.png"),
            "residuals_train": os.path.join(ARTIFACTS_DIR, "residuals_train.png"),
            "residuals_val": os.path.join(ARTIFACTS_DIR, "residuals_val.png"),
            "residuals_test": os.path.join(ARTIFACTS_DIR, "residuals_test.png"),
            "perm_importance": os.path.join(ARTIFACTS_DIR, "perm_importance.png"),
            "perm_importance_csv": imp_path,
            "predictions_csv": pred_path,
        }
    }
    with open(os.path.join(ARTIFACTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to:", os.path.join(ARTIFACTS_DIR, "summary.json"))

