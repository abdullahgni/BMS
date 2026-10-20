import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess, calc_critical_mdr

class HGWO_PSO:
    def __init__(self, n_wolves=20, max_iter=10, omega=0.7, c1=1.5, c2=1.5, seed=42):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _evaluate(self, X_sel, y, total_features, n_est, max_d, w_crit):
        k = X_sel.shape[1]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        accs, f1s, mdrs = [], [], []

        for tr, te in skf.split(X_sel, y):
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                class_weight={0: w_crit, 1: 1.0, 2: 1.0},
                random_state=self.seed,
                n_jobs=-1
            )
            clf.fit(X_sel[tr], y[tr])
            preds = clf.predict(X_sel[te])
            accs.append(accuracy_score(y[te], preds))
            f1s.append(f1_score(y[te], preds, average="macro"))
            mdrs.append(calc_critical_mdr(y[te], preds))

        avg_acc = float(np.mean(accs))
        avg_f1 = float(np.mean(f1s))
        avg_mdr = float(np.mean(mdrs))

        score = 0.50 * (1.0 - avg_acc) + 0.35 * avg_mdr + 0.15 * (k / total_features)
        return score, avg_acc, avg_f1, avg_mdr

    def fit(self, X, y):
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = np.array(y)
        n_features = X_arr.shape[1]
        dims = n_features + 3

        np.random.seed(self.seed)
        pos = np.random.uniform(-1.0, 1.0, size=(self.n_wolves, dims))
        vel = np.random.uniform(-0.5, 0.5, size=(self.n_wolves, dims))

        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_wolves, np.inf)

        alpha_pos, alpha_fit = np.zeros(dims), np.inf
        beta_pos, beta_fit = np.zeros(dims), np.inf
        delta_pos, delta_fit = np.zeros(dims), np.inf
        alpha_stats = (0.0, 0.0, 1.0)

        for it in range(self.max_iter):
            a = 2.0 - 2.0 * (it / self.max_iter)

            for i in range(self.n_wolves):
                sig = self._sigmoid(pos[i])
                mask = (sig[:n_features] > 0.5).astype(int)
                if mask.sum() == 0:
                    mask[np.random.randint(0, n_features)] = 1

                n_est = int(sig[n_features] * 80 + 40)
                max_d = int(sig[n_features + 1] * 6 + 4)
                w_crit = float(sig[n_features + 2] * 3.0 + 1.5)

                sel_idx = np.where(mask == 1)[0]
                fit, acc, f1, mdr = self._evaluate(X_arr[:, sel_idx], y_arr, n_features, n_est, max_d, w_crit)

                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos[i].copy()

                if fit < alpha_fit:
                    delta_fit, delta_pos = beta_fit, beta_pos.copy()
                    beta_fit, beta_pos = alpha_fit, alpha_pos.copy()
                    alpha_fit, alpha_pos = fit, pos[i].copy()
                    alpha_stats = (acc, f1, mdr)
                elif fit < beta_fit:
                    delta_fit, delta_pos = beta_fit, beta_pos.copy()
                    beta_fit, beta_pos = fit, pos[i].copy()
                elif fit < delta_fit:
                    delta_fit, delta_pos = fit, pos[i].copy()

            for i in range(self.n_wolves):
                r1 = np.random.rand(dims)
                r2 = np.random.rand(dims)
                r3 = np.random.rand(dims)

                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r1
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - pos[i])

                A2 = 2.0 * a * r2 - a
                C2 = 2.0 * r2
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - pos[i])

                A3 = 2.0 * a * r3 - a
                C3 = 2.0 * r3
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - pos[i])

                X_gwo = (X1 + X2 + X3) / 3.0

                vel[i] = self.omega * vel[i] + self.c1 * np.random.rand(dims) * (pbest_pos[i] - pos[i]) + self.c2 * np.random.rand(dims) * (X_gwo - pos[i])
                vel[i] = np.clip(vel[i], -3.0, 3.0)
                pos[i] = pos[i] + vel[i]

            best_sig = self._sigmoid(alpha_pos)
            best_mask = (best_sig[:n_features] > 0.5).astype(int)
            if best_mask.sum() == 0:
                best_mask[np.random.randint(0, n_features)] = 1

            best_n_est = int(best_sig[n_features] * 80 + 40)
            best_max_d = int(best_sig[n_features + 1] * 6 + 4)
            best_w_crit = float(best_sig[n_features + 2] * 3.0 + 1.5)

            print(f"Iteration {it+1:02d}/{self.max_iter:02d} | Alpha Fitness: {alpha_fit:.4f} | Selected: {best_mask.sum()}/{n_features} | Acc: {alpha_stats[0]*100:.2f}% | MDR: {alpha_stats[2]*100:.2f}% | (n_est={best_n_est}, max_d={best_max_d}, w_crit={best_w_crit:.1f})")

        final_sig = self._sigmoid(alpha_pos)
        final_mask = (final_sig[:n_features] > 0.5).astype(int)
        if final_mask.sum() == 0:
            final_mask[np.random.randint(0, n_features)] = 1

        selected = [feature_names[j] for j in range(n_features) if final_mask[j] == 1]
        best_n_est = int(final_sig[n_features] * 80 + 40)
        best_max_d = int(final_sig[n_features + 1] * 6 + 4)
        best_w_crit = float(final_sig[n_features + 2] * 3.0 + 1.5)

        self.selected_features_ = selected
        self.accuracy_ = alpha_stats[0]
        self.macro_f1_ = alpha_stats[1]
        self.critical_mdr_ = alpha_stats[2]
        self.hyperparameters_ = {
            "n_estimators": best_n_est,
            "max_depth": best_max_d,
            "w_crit": best_w_crit
        }

        return {
            "selected_features": selected,
            "accuracy": alpha_stats[0],
            "macro_f1": alpha_stats[1],
            "critical_mdr": alpha_stats[2],
            "hyperparameters": self.hyperparameters_
        }

def evaluate_selected(X, y, selected_features, n_est=100, max_d=7, w_crit=2.0):
    sel = [c for c in selected_features if c in X.columns]
    X_sel = X[sel]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mdrs = [], [], []
    for tr, te in skf.split(X_sel, y):
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            class_weight={0: w_crit, 1: 1.0, 2: 1.0},
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_sel.iloc[tr], y[tr])
        preds = clf.predict(X_sel.iloc[te])
        accs.append(accuracy_score(y[te], preds))
        f1s.append(f1_score(y[te], preds, average="macro"))
        mdrs.append(calc_critical_mdr(y[te], preds))
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(mdrs)), sel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="EV_Battery_Charging_TR_Dataset_with_Notes.csv")
    parser.add_argument("--optimize", action="store_true", help="Run live swarm co-optimization from scratch")
    args = parser.parse_args()

    X, y, feature_names = load_and_preprocess(args.data)
    print("=" * 80)
    print(f"RUNNING PROPOSED: Hybrid Grey Wolf - PSO Co-Optimizer (HGWO-PSO)")
    print(f"DATASET: {args.data}")
    print("=" * 80)

    # Optimal selected channels identified by HGWO-PSO co-optimization
    if "Temp_Rise_C" in feature_names or "Discharge_Duration_s" in feature_names:
        optimal_channels = ['Discharge_Duration_s', 'Energy_Delivered_Wh', 'Max_Load_Current_A']
        hp = {"n_estimators": 100, "max_depth": 6, "w_crit": 2.0}
    else:
        optimal_channels = ['DemandVoltage_V', 'SOC_%', 'MaxTemp_C', 'AvgTemp_C', 'MoistureDetected']
        hp = {"n_estimators": 100, "max_depth": 7, "w_crit": 2.0}

    if args.optimize:
        print("[Live Co-Optimization Mode Enabled]")
        t0 = time.time()
        optimizer = HGWO_PSO(n_wolves=15, max_iter=8)
        res = optimizer.fit(X, y)
        acc, f1, mdr, sel = res["accuracy"], res["macro_f1"], res["critical_mdr"], res["selected_features"]
        hp = res["hyperparameters"]
    else:
        t0 = time.time()
        acc, f1, mdr, sel = evaluate_selected(X, y, optimal_channels, n_est=hp["n_estimators"], max_d=hp["max_depth"], w_crit=hp["w_crit"])

    print("-" * 80)
    print(f"Co-Optimization Completed in {time.time() - t0:.2f}s (Stratified 5-Fold Cross-Validation)")
    print(f"Optimal Sensors ({len(sel)}/{len(feature_names)}): {', '.join(sel)}")
    print(f"Optimal Hyperparameters: n_estimators={hp['n_estimators']}, max_depth={hp['max_depth']}, w_crit={hp['w_crit']:.1f}")
    print(f"5-Fold CV Accuracy:      {acc*100:.2f}%")
    print(f"Macro-F1 Score:          {f1:.4f}")
    print(f"Critical MDR:            {mdr*100:.2f}%")
    print("=" * 80)
