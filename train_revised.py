# -*- coding: utf-8 -*-
"""
GeoANN-GA revised training/benchmark script.

Changes implemented for peer-review revision:
1. Raw data are split BEFORE any preprocessing (70% development / 30% untouched test).
2. Suction is log1p-transformed, then StandardScaler is fitted ONLY on the training data.
3. Four-fold CV is used inside the 70% development set for GA model selection.
4. GA chromosome contains ONLY (H1, H2, learning rate). ANN weights/biases are
   trained by Adam inside each CV fold; they are NOT optimized by the GA.
5. An ordinary ANN and Random Forest baselines are tuned with four-fold CV.
6. Test performance is reported overall and separately for CP > 50% to expose
   the limitation caused by sparse severe-collapse observations.
7. MC Dropout is implemented correctly: the dropout ANN is trained with dropout
   from the beginning; dropout is not introduced after training.

to run:
python train_revised.py --data /path/to/Suction_vsCP-modified_1.xlsx
"""

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from deap import base, creator, tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, GridSearchCV, train_test_split
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

FEATURE_COLS = [
    "Suction (kPa)",
    "Silica fume (%)",
    "Lime (%)",
    "Gypsum content (%)",
    "Applied vertical stress (kPa)",
    "Degree of Saturation (%)",
]
TARGET_COL = "Collapse Potential (%)"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Expected columns:
# Suction (kPa), Silica fume (%), Lime (%), Gypsum content (%),
# Applied vertical stress (kPa), Degree of Saturation (%), Collapse Potential (%)
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.do1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.do2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.do1(x)
        x = torch.relu(self.fc2(x))
        x = self.do2(x)
        return self.out(x)


def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def preprocess_fit(X):
    """Fit preprocessing on training data only."""
    X = np.asarray(X, dtype=float).copy()
    X[:, 0] = np.log1p(np.clip(X[:, 0], 0, None))
    scaler = StandardScaler()
    return scaler.fit(X), scaler.transform(X)


def preprocess_transform(X, scaler):
    X = np.asarray(X, dtype=float).copy()
    X[:, 0] = np.log1p(np.clip(X[:, 0], 0, None))
    return scaler.transform(X)


def fit_torch_model(Xtr, ytr, Xva, yva, h1, h2, lr, epochs=150, dropout=0.0):
    model = ANNModel(Xtr.shape[1], h1, h2, dropout=dropout).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE).view(-1, 1)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), torch.tensor(yva, dtype=torch.float32, device=DEVICE).view(-1, 1)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy().reshape(-1)


def make_stratified_folds(y, n_splits=4):
    """Create approximately stratified folds by binning the regression target."""
    y = np.asarray(y)
    try:
        bins = pd.qcut(y, q=min(8, len(y) // n_splits), labels=False, duplicates="drop")
        bins = np.asarray(bins)
        if len(np.unique(bins)) >= n_splits and np.min(np.bincount(bins)) >= n_splits:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            return list(skf.split(np.zeros(len(y)), bins))
    except Exception:
        pass
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(kf.split(np.zeros(len(y))))


def cv_score_ga(X, y, ind, folds, epochs=150):
    h1, h2, lr = int(round(ind[0])), int(round(ind[1])), float(ind[2])
    scores = []
    for tr_idx, va_idx in folds:
        scaler, Xtr = preprocess_fit(X[tr_idx])
        Xva = preprocess_transform(X[va_idx], scaler)
        # Scale target using training fold only.
        y_mean = y[tr_idx].mean()
        y_std = y[tr_idx].std() + 1e-12
        ytr = (y[tr_idx] - y_mean) / y_std
        yva = (y[va_idx] - y_mean) / y_std
        model = fit_torch_model(Xtr, ytr, Xva, yva, h1, h2, lr, epochs=epochs)
        pred_s = predict_torch(model, Xva)
        pred = pred_s * y_std + y_mean
        scores.append(np.sqrt(mean_squared_error(y[va_idx], pred)))
    return (float(np.mean(scores)),)


def run_ga(X_dev, y_dev, pop_size=20, ngen=15, epochs=150):
    if not hasattr(creator, "FitnessMinRevised"):
        creator.create("FitnessMinRevised", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualRevised"):
        creator.create("IndividualRevised", list, fitness=creator.FitnessMinRevised)

    folds = make_stratified_folds(y_dev, 4)
    toolbox = base.Toolbox()
    toolbox.register("attr_h1", random.randint, 4, 64)
    toolbox.register("attr_h2", random.randint, 4, 64)
    toolbox.register("attr_lr", random.uniform, 0.0005, 0.01)
    toolbox.register(
        "individual", tools.initCycle, creator.IndividualRevised,
        (toolbox.attr_h1, toolbox.attr_h2, toolbox.attr_lr), n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: cv_score_ga(X_dev, y_dev, ind, folds, epochs=epochs))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[4, 4, 0.0005], up=[64, 64, 0.01], eta=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    hist = []
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)

    for gen in range(ngen):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                if c1.fitness.valid: del c1.fitness.values
                if c2.fitness.valid: del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)
                if m.fitness.valid: del m.fitness.values
        for ind in [x for x in offspring if not x.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring
        hof.update(pop)
        vals = [i.fitness.values[0] for i in pop]
        hist.append({"generation": gen + 1, "best_cv_rmse": float(min(vals)),
                     "mean_cv_rmse": float(np.mean(vals)), "worst_cv_rmse": float(max(vals))})
        print(f"GA generation {gen+1}/{ngen}: best 4-fold CV RMSE={min(vals):.4f}")

    best = hof[0]
    return {"h1": int(round(best[0])), "h2": int(round(best[1])), "lr": float(best[2])}, hist


def train_final_ga(X_dev, y_dev, params, epochs=500):
    scaler, Xs = preprocess_fit(X_dev)
    y_mean, y_std = y_dev.mean(), y_dev.std() + 1e-12
    ys = (y_dev - y_mean) / y_std
    model = ANNModel(Xs.shape[1], params["h1"], params["h2"]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=params["lr"])
    crit = nn.MSELoss()
    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ys, dtype=torch.float32, device=DEVICE).view(-1, 1)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
    return model, scaler, y_mean, y_std


def evaluate_torch(model, scaler, y_mean, y_std, X, y):
    Xs = preprocess_transform(X, scaler)
    pred_s = predict_torch(model, Xs)
    pred = pred_s * y_std + y_mean
    return metrics(y, pred), pred


def high_collapse_report(y, pred, threshold=50.0):
    mask = y > threshold
    out = {"n": int(mask.sum()), "threshold": threshold}
    if mask.sum() >= 2:
        out.update({f"high_{k}": v for k, v in metrics(y[mask], pred[mask]).items()})
    else:
        out["note"] = "Too few observations above threshold for stable standalone metrics."
    return out


def baseline_models(X_dev, y_dev, X_test, y_test):
    # Same development/test split and same leakage-free preprocessing.
    scaler, Xd = preprocess_fit(X_dev)
    Xt = preprocess_transform(X_test, scaler)

    # Ordinary ANN baseline: 4-fold CV tuning.
    ann_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(random_state=SEED, early_stopping=True,
                              validation_fraction=0.15, max_iter=1500))
    ])
    ann_grid = {
        "mlp__hidden_layer_sizes": [(16, 8), (32, 16), (56, 32), (64, 48)],
        "mlp__alpha": [1e-5, 1e-4, 1e-3],
        "mlp__learning_rate_init": [5e-4, 1e-3, 5e-3],
    }
    ann_search = GridSearchCV(ann_pipe, ann_grid, cv=4, scoring="neg_root_mean_squared_error", n_jobs=-1)
    ann_search.fit(Xd, y_dev)
    ann_pred = ann_search.best_estimator_.predict(Xt)

    # Random forest baseline: 4-fold CV tuning.
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    rf_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 8, 12, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", 0.7],
    }
    rf_search = RandomizedSearchCV(rf, rf_dist, n_iter=20, cv=4,
                                    scoring="neg_root_mean_squared_error",
                                    random_state=SEED, n_jobs=-1)
    rf_search.fit(Xd, y_dev)
    rf_pred = rf_search.best_estimator_.predict(Xt)

    return {
        "ANN": {"metrics": metrics(y_test, ann_pred), "high_collapse": high_collapse_report(y_test, ann_pred),
                "best_params": ann_search.best_params_},
        "RFR": {"metrics": metrics(y_test, rf_pred), "high_collapse": high_collapse_report(y_test, rf_pred),
                "best_params": rf_search.best_params_},
    }


def train_dropout_model(X_dev, y_dev, h1, h2, lr, epochs=500, p_drop=0.2):
    scaler, Xs = preprocess_fit(X_dev)
    y_mean, y_std = y_dev.mean(), y_dev.std() + 1e-12
    ys = (y_dev - y_mean) / y_std
    model = ANNModel(Xs.shape[1], h1, h2, dropout=p_drop).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ys, dtype=torch.float32, device=DEVICE).view(-1, 1)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()
    return model, scaler, y_mean, y_std


def mc_dropout_predictions(model, scaler, y_mean, y_std, X_test, T=200):
    Xs = preprocess_transform(X_test, scaler)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)
    # model.train() deliberately activates dropout for stochastic inference.
    model.train()
    preds = []
    for _ in range(T):
        with torch.no_grad():
            p = model(Xt).cpu().numpy().reshape(-1)
        preds.append(p * y_std + y_mean)
    return np.asarray(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the Excel dataset")
    parser.add_argument("--out", default="revised_results", help="Output directory")
    parser.add_argument("--ga-pop", type=int, default=20)
    parser.add_argument("--ga-gen", type=int, default=15)
    parser.add_argument("--cv-epochs", type=int, default=150)
    parser.add_argument("--final-epochs", type=int, default=500)
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.data)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)

    # IMPORTANT: split raw observations before fitting ANY transform.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    print(f"Raw split: development={len(X_dev)} (70%), independent test={len(X_test)} (30%)")
    print("Development CP summary:\n", pd.Series(y_dev).describe())
    print("Independent test CP summary:\n", pd.Series(y_test).describe())

    best_params, ga_hist = run_ga(X_dev, y_dev, pop_size=args.ga_pop,
                                  ngen=args.ga_gen, epochs=args.cv_epochs)
    print("Best GA parameters:", best_params)

    final_model, scaler, y_mean, y_std = train_final_ga(
        X_dev, y_dev, best_params, epochs=args.final_epochs
    )
    ga_metrics, ga_pred = evaluate_torch(final_model, scaler, y_mean, y_std, X_test, y_test)
    ga_high = high_collapse_report(y_test, ga_pred, threshold=50.0)

    # Proper MC Dropout: dropout is present during training, then kept active for stochastic inference.
    mc_model, mc_scaler, mc_mean, mc_std = train_dropout_model(
        X_dev, y_dev, best_params["h1"], best_params["h2"], best_params["lr"],
        epochs=args.final_epochs, p_drop=0.2
    )
    mc_pred_samples = mc_dropout_predictions(mc_model, mc_scaler, mc_mean, mc_std, X_test, T=200)
    mc_mean_pred = mc_pred_samples.mean(axis=0)
    mc_std_pred = mc_pred_samples.std(axis=0)
    mc_lower = mc_mean_pred - 1.96 * mc_std_pred
    mc_upper = mc_mean_pred + 1.96 * mc_std_pred
    mc_coverage = float(np.mean((y_test >= mc_lower) & (y_test <= mc_upper)))

    baselines = baseline_models(X_dev, y_dev, X_test, y_test)

    # Also report performance by vertical-stress level when enough test observations exist.
    stress_values = sorted(np.unique(X_test[:, 4]))
    stress_rows = []
    for stress in stress_values:
        mask = X_test[:, 4] == stress
        if mask.sum() >= 3:
            mm = metrics(y_test[mask], ga_pred[mask])
            stress_rows.append({"stress_kPa": float(stress), "n": int(mask.sum()), **mm})

    results = {
        "seed": SEED,
        "n_total": int(len(df)),
        "n_development": int(len(X_dev)),
        "n_test": int(len(X_test)),
        "split": "70% development / 30% independent test",
        "preprocessing": "log1p(Suction) then StandardScaler; all preprocessing fit on development/training folds only",
        "ga": {
            "optimization": "4-fold CV on development set",
            "chromosome": ["H1", "H2", "learning_rate"],
            "weights_biases": "trained by Adam within each fold; not GA chromosome",
            "best_params": best_params,
            "test_metrics": ga_metrics,
            "high_collapse_gt_50": ga_high,
        },
        "mc_dropout": {
            "dropout_rate": 0.2,
            "T": 200,
            "test_point_metrics": metrics(y_test, mc_mean_pred),
            "empirical_95pct_coverage": mc_coverage,
        },
        "baselines": baselines,
        "limitation": "The test set contains relatively few severe-collapse observations; performance for CP > 50% is therefore reported separately and interpreted cautiously rather than extrapolated to unseen severe-collapse conditions.",
    }

    pd.DataFrame([
        {"Model": "GA-ANN", **ga_metrics},
        {"Model": "ANN", **baselines["ANN"]["metrics"]},
        {"Model": "RFR", **baselines["RFR"]["metrics"]},
    ]).to_csv(outdir / "table5_metrics.csv", index=False)
    pd.DataFrame(ga_hist).to_csv(outdir / "ga_convergence.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(outdir / "ga_test_by_stress.csv", index=False)
    pd.DataFrame({"actual": y_test, "mc_mean": mc_mean_pred, "mc_lower95": mc_lower, "mc_upper95": mc_upper, "mc_std": mc_std_pred}).to_csv(outdir / "mc_dropout_test_intervals.csv", index=False)
    with open(outdir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "h1": best_params["h1"],
        "h2": best_params["h2"],
        "lr": best_params["lr"],
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "scaler": scaler,
        "target_mean": float(y_mean),
        "target_std": float(y_std),
        "seed": SEED,
    }, outdir / "ga_ann_final.pth")

    print("\n=== TEST RESULTS ===")
    print("GA-ANN:", ga_metrics)
    print("GA-ANN CP > 50%:", ga_high)
    print("ANN:", baselines["ANN"]["metrics"])
    print("RFR:", baselines["RFR"]["metrics"])
    print("\nSaved:", outdir.resolve())


if __name__ == "__main__":
    main()
