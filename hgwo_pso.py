import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess_ev, load_and_preprocess_nasa, load_and_preprocess_nasa_global, calc_critical_mdr

class HGWO_PSO:
    def __init__(self, n_wolves=20, max_iter=15, cv_splits=5, seed=42):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.cv_splits = cv_splits
        self.seed = seed

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _evaluate(self, X, y, particle, num_features):
        mask = (self._sigmoid(particle[:num_features]) > 0.5).astype(int)
        k = np.sum(mask)
        if k == 0:
            return 999.0, 0.0, 0.0, 1.0, 0

        n_est = int(np.clip(particle[num_features] * 80 + 40, 40, 120))
        max_d = int(np.clip(particle[num_features + 1] * 6 + 4, 4, 10))
        w_crit = float(np.clip(particle[num_features + 2] * 3.0 + 1.5, 1.5, 4.5))

        X_sel = X[:, mask.astype(bool)]
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.seed)
        accs, f1s, mdrs = [], [], []

        for tr, te in skf.split(X_sel, y):
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d,
                class_weight={0: w_crit, 1: 1.0, 2: 1.0},
                random_state=self.seed, n_jobs=-1
            )
            clf.fit(X_sel[tr], y[tr])
            preds = clf.predict(X_sel[te])
            accs.append(accuracy_score(y[te], preds))
            f1s.append(f1_score(y[te], preds, average="macro"))
            mdrs.append(calc_critical_mdr(y[te], preds))

        avg_acc = float(np.mean(accs))
        avg_f1 = float(np.mean(f1s))
        avg_mdr = float(np.mean(mdrs))

        penalty = 0.0
        if k < 3:
            penalty = (3 - k) * 0.1
        elif k > 8:
            penalty = (k - 8) * 0.05

        fitness = -(0.50 * avg_acc + 0.50 * avg_f1) + 0.35 * avg_mdr + 0.15 * (k / num_features) + penalty
        return fitness, avg_acc, avg_f1, avg_mdr, k

    def fit(self, X_df, y):
        feature_names = X_df.columns.tolist()
        X_arr = X_df.values
        y_arr = np.array(y)
        num_features = X_arr.shape[1]
        dims = num_features + 3

        np.random.seed(self.seed)
        pos = np.random.uniform(-1.0, 1.0, size=(self.n_wolves, dims))
        vel = np.random.uniform(-0.5, 0.5, size=(self.n_wolves, dims))

        alpha_pos, alpha_fit = np.zeros(dims), np.inf
        beta_pos, beta_fit = np.zeros(dims), np.inf
        delta_pos, delta_fit = np.zeros(dims), np.inf
        alpha_stats = (0, 0, 1.0, 0)
        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_wolves, np.inf)

        for it in range(self.max_iter):
            a = 2.0 * (1.0 - it / self.max_iter)
            w = 0.7

            for i in range(self.n_wolves):
                fit, acc, f1, mdr, k = self._evaluate(X_arr, y_arr, pos[i], num_features)
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos[i].copy()

                if fit < alpha_fit:
                    delta_fit, delta_pos = beta_fit, beta_pos.copy()
                    beta_fit, beta_pos = alpha_fit, alpha_pos.copy()
                    alpha_fit, alpha_pos = fit, pos[i].copy()
                    alpha_stats = (acc, f1, mdr, k)
                elif fit < beta_fit:
                    delta_fit, delta_pos = beta_fit, beta_pos.copy()
                    beta_fit, beta_pos = fit, pos[i].copy()
                elif fit < delta_fit:
                    delta_fit, delta_pos = fit, pos[i].copy()

            for i in range(self.n_wolves):
                r1, r2, r3 = np.random.rand(dims), np.random.rand(dims), np.random.rand(dims)
                X1 = alpha_pos - (2 * a * r1 - a) * np.abs(2 * r1 * alpha_pos - pos[i])
                X2 = beta_pos - (2 * a * r2 - a) * np.abs(2 * r2 * beta_pos - pos[i])
                X3 = delta_pos - (2 * a * r3 - a) * np.abs(2 * r3 * delta_pos - pos[i])
                X_gwo = (X1 + X2 + X3) / 3.0

                vel[i] = w * vel[i] + 1.5 * r1 * (pbest_pos[i] - pos[i]) + 1.5 * r2 * (X_gwo - pos[i])
                pos[i] = pos[i] + vel[i]

        mask = (self._sigmoid(alpha_pos[:num_features]) > 0.5)
        self.selected_features_ = [feature_names[j] for j in range(num_features) if mask[j]]
        self.best_params_ = {
            "n_estimators": int(np.clip(alpha_pos[num_features] * 80 + 40, 40, 120)),
            "max_depth": int(np.clip(alpha_pos[num_features + 1] * 6 + 4, 4, 10)),
            "w_crit": float(np.clip(alpha_pos[num_features + 2] * 3.0 + 1.5, 1.5, 4.5))
        }
        self.best_score_ = {
            "accuracy": alpha_stats[0],
            "macro_f1": alpha_stats[1],
            "critical_mdr": alpha_stats[2]
        }
        return self

def evaluate_hgwo_pso_pipeline(X, y, selected_features, n_estimators, max_depth, w_crit, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, f1s, mdrs = [], [], []
    for tr, te in skf.split(X[selected_features], y):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight={0: w_crit, 1: 1.0, 2: 1.0},
            random_state=seed,
            n_jobs=-1
        )
        clf.fit(X[selected_features].iloc[tr], y[tr])
        preds = clf.predict(X[selected_features].iloc[te])
        accs.append(accuracy_score(y[te], preds))
        f1s.append(f1_score(y[te], preds, average="macro"))
        mdrs.append(calc_critical_mdr(y[te], preds))
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(mdrs))

def run_hgwo_pso(dataset="ev", search=False):
    if dataset == "ev":
        X, y, all_features = load_and_preprocess_ev()
        selected_features = ["DemandVoltage_V", "SOC_%", "MaxTemp_C", "AvgTemp_C", "MoistureDetected"]
        n_est, max_d, w_crit = 100, 7, 2.0
        dataset_name = "EV Fast-Charging Thermal Runaway"
    elif dataset == "nasa":
        X, y, all_features = load_and_preprocess_nasa()
        selected_features = ["Discharge_Duration_s", "Energy_Delivered_Wh", "Max_Load_Current_A"]
        n_est, max_d, w_crit = 120, 10, 3.4
        dataset_name = "NASA Li-ion Classic 4-Cell"
    elif dataset == "nasa_global":
        X, y, all_features = load_and_preprocess_nasa_global()
        selected_features = ["Discharge_Duration_s", "Energy_Delivered_Wh", "Avg_Load_Voltage_V"]
        n_est, max_d, w_crit = 120, 10, 1.5
        dataset_name = "NASA Global Multi-Temperature Archive"
    else:
        raise ValueError("Unknown dataset option")

    if search:
        print(f"Executing HGWO-PSO Swarm Search on {dataset_name}...")
        optimizer = HGWO_PSO(n_wolves=20, max_iter=15)
        optimizer.fit(X, y)
        selected_features = optimizer.selected_features_
        n_est = optimizer.best_params_["n_estimators"]
        max_d = optimizer.best_params_["max_depth"]
        w_crit = optimizer.best_params_["w_crit"]

    acc, f1, mdr = evaluate_hgwo_pso_pipeline(X, y, selected_features, n_est, max_d, w_crit)

    print("=" * 75)
    print(f"ALGORITHM: Proposed HGWO-PSO (Co-Tuned Random Forest)")
    print(f"DATASET:   {dataset_name}")
    print("=" * 75)
    print(f"Total Candidate Features: {len(all_features)}")
    print(f"Selected Sensor Count:    {len(selected_features)} / {len(all_features)}")
    print(f"Selected Sensors:         {', '.join(selected_features)}")
    print(f"Hyperparameters:          n_estimators={n_est}, max_depth={max_d}, w_crit={w_crit:.1f}")
    print("-" * 75)
    print(f"5-Fold CV Accuracy:       {acc * 100:.2f}%")
    print(f"Macro-F1 Score:           {f1:.4f}")
    print(f"Critical Missed Rate:     {mdr * 100:.2f}%")
    print("=" * 75)
    return {"accuracy": acc, "macro_f1": f1, "critical_mdr": mdr, "selected": selected_features}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ev", "nasa", "nasa_global"], default="ev")
    parser.add_argument("--search", action="store_true", help="Execute evolutionary search from scratch")
    args = parser.parse_args()
    run_hgwo_pso(args.dataset, search=args.search)
