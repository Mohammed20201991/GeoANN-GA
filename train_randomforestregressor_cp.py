# %pip install optuna

from google.colab import drive
drive.mount('/content/drive')

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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import optuna
from datetime import datetime

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config
DATA_CSV     = "/content/drive/MyDrive/PINNs/Suction_vsCP-modified_1.xlsx"
TARGET_COL   = "Collapse Potential (%)"
FEATURE_COLS = [
    "Suction (kPa)",
    "Silica fume (%)",
    "Lime (%)",
    "Gypsum content (%)",
    "Applied vertical stress (kPa)",
    "Degree of Saturation (%)",
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ARTIFACTS_DIR = f"/content/drive/MyDrive/NNsGA/FNNs/artifacts_advanced_{timestamp}"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Utilities
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.clip(np.abs(y_true), eps, None)))) * 100.0

@dataclass
class EnhancedTrainConfig:
    mfs_per_feature: int = 2
    batch_size: int = 64
    max_epochs: int = 500
    lr: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 50
    warmup_epochs: int = 15
    dropout_rate: float = 0.2
    rule_dropout_rate: float = 0.1
    use_feature_attention: bool = True
    use_rule_importance: bool = True

@dataclass
class AdvancedTrainConfig(EnhancedTrainConfig):
    use_ensemble: bool = True
    ensemble_size: int = 3
    use_feature_interactions: bool = True
    use_adaptive_learning: bool = True
    use_advanced_regularization: bool = True

# Data
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Enhanced TSK Model
class EnhancedTSKFuzzyRegressor(nn.Module):
    def __init__(self, D: int, M: int, config: EnhancedTrainConfig):
        super().__init__()
        self.D = D
        self.M = M
        self.config = config
        self.R = M ** D

        # MF parameters
        self.centers    = nn.Parameter(torch.zeros(D, M))
        self.log_sigmas = nn.Parameter(torch.zeros(D, M))

        # Rule index tensor
        combos = np.stack(np.meshgrid(*[np.arange(M) for _ in range(D)], indexing='ij'), axis=-1).reshape(-1, D)
        self.register_buffer('rule_index', torch.from_numpy(combos).long())

        # Enhanced components
        if config.use_rule_importance:
            self.rule_importance = nn.Parameter(torch.ones(self.R))

        if config.use_feature_attention:
            self.feature_attention = nn.Linear(D, D)

        # Consequents
        self.consequents  = nn.Linear(D, self.R, bias=True)

        # Regularization
        self.dropout      = nn.Dropout(config.dropout_rate)
        self.rule_dropout = nn.Dropout(config.rule_dropout_rate)

        self.eps = 1e-8

    def gaussian_mf(self, x):
        N, D = x.shape
        centers = self.centers
        sigmas  = torch.nn.functional.softplus(self.log_sigmas) + 1e-4

        # Apply feature attention if enabled
        if self.config.use_feature_attention:
            x_att = torch.sigmoid(self.feature_attention(x)) * x
        else:
            x_att = x

        x_exp = x_att.unsqueeze(-1)
        c_exp = centers.unsqueeze(0)
        s_exp = sigmas.unsqueeze(0)

        z = (x_exp - c_exp) / s_exp
        mu = torch.exp(-0.5 * z * z)
        return mu

    def rule_firing(self, mu):
        N, D, M = mu.shape
        gather_list = []
        for j in range(D):
            mu_j  = mu[:, j, :]
            mu_jg = mu_j.index_select(dim=1, index=self.rule_index[:, j]).view(N, -1)
            gather_list.append(mu_jg)
        w = torch.ones_like(gather_list[0])
        for g in gather_list:
            w = w * g
        return w

    def forward(self, x):
        mu = self.gaussian_mf(x)
        w = self.rule_firing(mu)

        # Apply rule importance if enabled
        if self.config.use_rule_importance:
            w = w * torch.sigmoid(self.rule_importance).unsqueeze(0)

        # Apply rule dropout
        w = self.rule_dropout(w)

        w_sum = w.sum(dim=1, keepdim=True)
        beta  = w / (w_sum + self.eps)

        # Apply dropout to input for consequents
        x_drop = self.dropout(x)
        y_lin  = self.consequents(x_drop)

        y = (beta * y_lin).sum(dim=1, keepdim=True)
        return y, w_sum

# Advanced TSK Ensemble Model
class AdvancedTSKEnsemble(nn.Module):
    def __init__(self, D: int, M: int, config: AdvancedTrainConfig):
        super().__init__()
        self.D = D
        self.M = M
        self.config = config
        self.ensemble_size = config.ensemble_size

        # Create ensemble of TSK models
        self.models = nn.ModuleList([
            EnhancedTSKFuzzyRegressor(D, M, config) for _ in range(self.ensemble_size)
        ])

        # Feature interaction layer
        if config.use_feature_interactions:
            self.interaction_weights = nn.Parameter(torch.ones(D, D))

        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(self.ensemble_size))

    def forward(self, x):
        # Apply feature interactions if enabled
        if self.config.use_feature_interactions:
            x_interacted = x @ torch.sigmoid(self.interaction_weights)
            x_enhanced = torch.cat([x, x_interacted], dim=1)
        else:
            x_enhanced = x

        # Get predictions from all ensemble members
        predictions = []
        for model in self.models:
            pred, _ = model(x_enhanced[:, :self.D])  # Use original features for individual models
            predictions.append(pred)

        # Weighted ensemble average
        weights = torch.softmax(self.ensemble_weights, dim=0)
        final_pred = sum(w * pred for w, pred in zip(weights, predictions))

        return final_pred, None

# Enhanced Initialization
def enhanced_init_mfs(model, X_train, y_train=None):
    D = X_train.shape[1]
    M = model.M

    # Get feature importance if y_train is provided
    if y_train is not None:
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
        rf.fit(X_train, y_train)
        feature_importance = rf.feature_importances_
    else:
        feature_importance = np.ones(D) / D

    for j in range(D):
        # Use K-means for better center initialization
        kmeans = KMeans(n_clusters=M, random_state=SEED+j)
        kmeans.fit(X_train[:, j].reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())

        # Ensure we have exactly M centers
        if len(centers) < M:
            min_val, max_val = np.min(X_train[:, j]), np.max(X_train[:, j])
            additional_centers = np.linspace(min_val, max_val, M - len(centers) + 2)[1:-1]
            centers = np.sort(np.concatenate([centers, additional_centers]))

        # Adaptive sigma based on feature importance
        sigma_base = np.std(X_train[:, j]) * (0.3 + 0.7 * feature_importance[j])
        sigmas = np.full(M, sigma_base)

        with torch.no_grad():
            # Ensemble case
            if hasattr(model, 'models'):
                for m in model.models:
                    m.centers[j].copy_(torch.from_numpy(centers.astype(np.float32)))
                    m.log_sigmas[j].copy_(torch.log(torch.from_numpy(sigmas.astype(np.float32))))
            else:  # Single model case
                model.centers[j].copy_(torch.from_numpy(centers.astype(np.float32)))
                model.log_sigmas[j].copy_(torch.log(torch.from_numpy(sigmas.astype(np.float32))))

# Training Functions
def validate_model(model, val_loader, criterion):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            val_losses.append(loss.item())
    return float(np.mean(val_losses))

def enhanced_train_model(model, train_loader, val_loader, cfg: EnhancedTrainConfig):
    model.to(DEVICE)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

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

            # Add L1 regularization for consequents
            l1_reg = torch.tensor(0.).to(DEVICE)
            for param in model.consequents.parameters():
                l1_reg += torch.norm(param, 1)
            loss += 1e-4 * l1_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = validate_model(model, val_loader, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience   = cfg.patience
        else:
            patience -= 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        if patience <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

def advanced_train_model(model, train_loader, val_loader, cfg: AdvancedTrainConfig):
    model.to(DEVICE)

    # Adaptive loss function
    def adaptive_loss(pred, target):
        mse = nn.MSELoss()(pred, target)
        mae = nn.L1Loss()(pred, target)
        return 0.7 * mse + 0.3 * mae

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Multi-step learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[50, 100, 150],
                                                   gamma=0.5)

    best_val   = float('inf')
    best_state = None
    history    = {"train": [], "val": [], "lr": []}
    patience   = cfg.patience

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred, _ = model(xb)
            loss    = adaptive_loss(pred, yb)

            # Advanced regularization
            if cfg.use_advanced_regularization:
                l1_reg, l2_reg = torch.tensor(0.).to(DEVICE), torch.tensor(0.).to(DEVICE)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                    l2_reg += torch.norm(param, 2)
                loss += 5e-5 * l1_reg + 1e-5 * l2_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = validate_model(model, val_loader, adaptive_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience   = cfg.patience
        else:
            patience -= 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

        if patience <= 0:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

# Feature Engineering
def engineer_features(X, feature_names):
    """Create domain-specific feature interactions"""
    X_engineered = X.copy()
    new_features = []

    # Soil composition interactions
    if all(col in feature_names for col in ["Silica fume (%)", "Lime (%)", "Gypsum content (%)"]):
        silica_idx = feature_names.index("Silica fume (%)")
        lime_idx   = feature_names.index("Lime (%)")
        gypsum_idx = feature_names.index("Gypsum content (%)")

        # Binder ratio features
        binder_total = X[:, silica_idx] + X[:, lime_idx] + X[:, gypsum_idx] + 1e-8
        silica_ratio = X[:, silica_idx] / binder_total
        lime_ratio   = X[:, lime_idx] / binder_total
        gypsum_ratio = X[:, gypsum_idx] / binder_total

        X_engineered = np.column_stack([X_engineered, silica_ratio, lime_ratio, gypsum_ratio])
        new_features.extend(["Silica_Ratio", "Lime_Ratio", "Gypsum_Ratio"])

    # Stress-saturation interaction
    if all(col in feature_names for col in ["Applied vertical stress (kPa)", "Degree of Saturation (%)"]):
        stress_idx        = feature_names.index("Applied vertical stress (kPa)")
        saturation_idx    = feature_names.index("Degree of Saturation (%)")
        stress_saturation = X[:, stress_idx] * (X[:, saturation_idx] / 100.0)
        X_engineered      = np.column_stack([X_engineered, stress_saturation])
        new_features.append("Stress_Saturation_Interaction")

    # Suction-stress interaction
    if all(col in feature_names for col in ["Suction (kPa)", "Applied vertical stress (kPa)"]):
        suction_idx    = feature_names.index("Suction (kPa)")
        stress_idx     = feature_names.index("Applied vertical stress (kPa)")
        suction_stress = X[:, suction_idx] * X[:, stress_idx]
        X_engineered   = np.column_stack([X_engineered, suction_stress])
        new_features.append("Suction_Stress_Interaction")

    return X_engineered, new_features

# Evaluation and Plotting
def evaluate(model, X: np.ndarray, y: np.ndarray) -> Tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        X_t      = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        y_hat, _ = model(X_t)
        y_hat    = y_hat.cpu().numpy().reshape(-1)
    metrics = {
        "RMSE": rmse(y, y_hat),
        "MAE": mean_absolute_error(y, y_hat),
        "R2": r2_score(y, y_hat),
        "MAPE_%": mape(y, y_hat),
    }
    return metrics, y_hat

def plot_training(history: dict, outdir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train"], label="Train Loss", alpha=0.8)
    plt.plot(history["val"], label="Validation Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=160)
    plt.close()

def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, split_name: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, s=30, alpha=0.6, edgecolors='w', linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', alpha=0.8)
    plt.xlabel("Actual Collapse Potential (%)")
    plt.ylabel("Predicted Collapse Potential (%)")
    plt.title(f"Parity Plot — {split_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"parity_{split_name.lower()}.png"), dpi=160)
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, outdir: str, split_name: str):
    res = y_pred - y_true
    plt.figure(figsize=(8, 6))
    plt.hist(res, bins=40, alpha=0.7, edgecolor='black')
    plt.xlabel("Residual (Pred − True)")
    plt.ylabel("Count")
    plt.title(f"Residuals — {split_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"residuals_{split_name.lower()}.png"), dpi=160)
    plt.close()

# Main Execution
def main():
    print("=" * 60)
    print("ADVANCED FUZZY NEURAL NETWORK FOR COLLAPSE PREDICTION")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    print("=" * 60)

    # Load data
    print("Loading data...")
    df = pd.read_excel(DATA_CSV)

    # Verify all columns exist
    for col in FEATURE_COLS + [TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    print(f"Data shape: {df.shape}")
    print(f"Original features: {FEATURE_COLS}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Feature engineering
    print("Engineering features...")
    X_engineered, new_feature_names = engineer_features(X, FEATURE_COLS)
    all_feature_names = FEATURE_COLS + new_feature_names
    print(f"Added {len(new_feature_names)} engineered features: {new_feature_names}")
    print(f"Total features: {len(all_feature_names)}")

    # Handle outliers
    y = np.clip(y, np.percentile(y, 2), np.percentile(y, 98))

    # Split data (70/15/15)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_engineered, y, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Scale all features
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Advanced configuration
    cfg = AdvancedTrainConfig(
        lr=0.005,
        mfs_per_feature=2,
        dropout_rate=0.25,
        use_ensemble=True,
        ensemble_size=3,
        use_feature_interactions=True,
        use_adaptive_learning=True,
        use_advanced_regularization=True
    )

    # Build advanced model
    D = X_train_s.shape[1]
    model = AdvancedTSKEnsemble(D, cfg.mfs_per_feature, cfg)
    enhanced_init_mfs(model, X_train_s, y_train)

    # Create data loaders
    train_ds = TabDataset(X_train_s, y_train)
    val_ds   = TabDataset(X_val_s, y_val)
    test_ds  = TabDataset(X_test_s, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Train model
    print("\nTraining advanced ensemble model...")
    start_time = time.time()
    history = advanced_train_model(model, train_loader, val_loader, cfg)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.1f} minutes")

    # Evaluate
    print("\nEvaluating model...")
    train_metrics, yhat_train = evaluate(model, X_train_s, y_train)
    val_metrics, yhat_val = evaluate(model, X_val_s, y_val)
    test_metrics, yhat_test = evaluate(model, X_test_s, y_test)

    # Convert to JSON serializable
    train_metrics_json = {k: float(v) for k, v in train_metrics.items()}
    val_metrics_json = {k: float(v) for k, v in val_metrics.items()}
    test_metrics_json = {k: float(v) for k, v in test_metrics.items()}

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Metrics (Train)", json.dumps(train_metrics_json, indent=2))
    print("Metrics (Val)  ", json.dumps(val_metrics_json, indent=2))
    print("Metrics (Test) ", json.dumps(test_metrics_json, indent=2))

    # Save plots
    print("\nSaving plots...")
    plot_training(history, ARTIFACTS_DIR)
    plot_parity(y_train, yhat_train, ARTIFACTS_DIR, "Train")
    plot_parity(y_val, yhat_val, ARTIFACTS_DIR, "Validation")
    plot_parity(y_test, yhat_test, ARTIFACTS_DIR, "Test")
    plot_residuals(y_train, yhat_train, ARTIFACTS_DIR, "Train")
    plot_residuals(y_val, yhat_val, ARTIFACTS_DIR, "Validation")
    plot_residuals(y_test, yhat_test, ARTIFACTS_DIR, "Test")

    # Save model and results
    print("Saving model and results...")
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "feature_cols": all_feature_names,
        "original_feature_cols": FEATURE_COLS,
        "engineered_feature_cols": new_feature_names,
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "training_time": training_time,
        "timestamp": timestamp
    }, os.path.join(ARTIFACTS_DIR, "advanced_tsk_model.pt"))

    # Save metrics
    results = {
        "timestamp": timestamp,
        "training_time_minutes": training_time / 60,
        "config": cfg.__dict__,
        "metrics": {
            "train": train_metrics_json,
            "validation": val_metrics_json,
            "test": test_metrics_json
        },
        "feature_names": all_feature_names,
        "dataset_info": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "original_features": FEATURE_COLS,
            "engineered_features": new_feature_names
        }
    }

    with open(os.path.join(ARTIFACTS_DIR, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "split": ["train"] * len(y_train) + ["val"] * len(y_val) + ["test"] * len(y_test),
        "y_true": np.concatenate([y_train, y_val, y_test]),
        "y_pred": np.concatenate([yhat_train, yhat_val, yhat_test]),
    })
    pred_df.to_csv(os.path.join(ARTIFACTS_DIR, "predictions.csv"), index=False)

    print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")
    print("=" * 60)

    return train_metrics, val_metrics, test_metrics

if __name__ == "__main__":
    train_metrics, val_metrics, test_metrics = main()