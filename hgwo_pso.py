import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess_ev, load_and_preprocess_nasa, load_and_preprocess_nasa_global, calc_critical_mdr

class HGWO_PSO:
    def __init__(self, n_wolves=20, max_iter=15, omega=0.7, c1=1.5, c2=1.5, seed=42):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _fitness(self, X_sel, y, total_features, n_est, max_d, w_crit):
        k = X_sel.shape[1]
        if k == 0:
            return 999.0, 0.0, 0.0, 1.0

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

        penalty = 0.0
        if k < 3:
            penalty = (3 - k) * 0.05
        elif k > 8:
            penalty = (k - 8) * 0.03

        score = 0.50 * (1.0 - avg_acc) + 0.35 * avg_mdr + 0.15 * (k / total_features) + penalty
        return score, avg_acc, avg_f1, avg_mdr

    def optimize(self, X, y, feature_names):
        np.random.seed(self.seed)
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = np.array(y)
        n_features = X_arr.shape[1]
        dims = n_features + 3

        pos = np.random.uniform(-1.0, 1.0, size=(self.n_wolves, dims))
        vel = np.random.uniform(-0.5, 0.5, size=(self.n_wolves, dims))

        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_wolves, np.inf)

        alpha_pos, alpha_fit = np.zeros(dims), np.inf
        beta_pos, beta_fit = np.zeros(dims), np.inf
        delta_pos, delta_fit = np.zeros(dims), np.inf
        alpha_stats = (0.0, 0.0, 1.0)

        for it in range(self.max_iter):
            a = 2.0 * (1.0 - it / self.max_iter)

            for i in range(self.n_wolves):
                mask = (self._sigmoid(pos[i, :n_features]) > 0.5).astype(int)
                if np.sum(mask) == 0:
                    mask[np.random.randint(0, n_features)] = 1

                n_est = int(np.clip(self._sigmoid(pos[i, n_features]) * 80 + 40, 40, 120))
                max_d = int(np.clip(self._sigmoid(pos[i, n_features + 1]) * 6 + 4, 4, 10))
                w_crit = float(np.clip(self._sigmoid(pos[i, n_features + 2]) * 3.0 + 1.5, 1.5, 4.5))

                sel_idx = np.where(mask == 1)[0]
                fit, acc, f1, mdr = self._fitness(X_arr[:, sel_idx], y_arr, n_features, n_est, max_d, w_crit)

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
                D_alpha = np.abs(C1 * alpha_pos - pos[i])
                X1 = alpha_pos - A1 * D_alpha

                A2 = 2.0 * a * r2 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - pos[i])
                X2 = beta_pos - A2 * D_beta

                A3 = 2.0 * a * r3 - a
                C3 = 2.0 * r3
                D_delta = np.abs(C3 * delta_pos - pos[i])
                X3 = delta_pos - A3 * D_delta

                X_gwo = (X1 + X2 + X3) / 3.0

                vel[i] = self.omega * vel[i] + self.c1 * np.random.rand(dims) * (pbest_pos[i] - pos[i]) + self.c2 * np.random.rand(dims) * (X_gwo - pos[i])
                vel[i] = np.clip(vel[i], -3.0, 3.0)
                pos[i] = pos[i] + vel[i]

            best_mask = (self._sigmoid(alpha_pos[:n_features]) > 0.5).astype(int)
            if np.sum(best_mask) == 0:
                best_mask[np.random.randint(0, n_features)] = 1

            best_n_est = int(np.clip(self._sigmoid(alpha_pos[n_features]) * 80 + 40, 40, 120))
            best_max_d = int(np.clip(self._sigmoid(alpha_pos[n_features + 1]) * 6 + 4, 4, 10))
            best_w_crit = float(np.clip(self._sigmoid(alpha_pos[n_features + 2]) * 3.0 + 1.5, 1.5, 4.5))

            print(f"Iteration {it+1:02d}/{self.max_iter:02d} | Alpha Fitness: {alpha_fit:.4f} | Selected: {np.sum(best_mask)}/{n_features} | Acc: {alpha_stats[0]*100:.2f}% | MDR: {alpha_stats[2]*100:.2f}% | (n_est={best_n_est}, max_d={best_max_d}, w_crit={best_w_crit:.1f})")

        final_mask = (self._sigmoid(alpha_pos[:n_features]) > 0.5).astype(int)
        if np.sum(final_mask) == 0:
            final_mask[np.random.randint(0, n_features)] = 1

        selected_features = [feature_names[j] for j in range(n_features) if final_mask[j] == 1]
        best_n_est = int(np.clip(self._sigmoid(alpha_pos[n_features]) * 80 + 40, 40, 120))
        best_max_d = int(np.clip(self._sigmoid(alpha_pos[n_features + 1]) * 6 + 4, 4, 10))
        best_w_crit = float(np.clip(self._sigmoid(alpha_pos[n_features + 2]) * 3.0 + 1.5, 1.5, 4.5))

        return {
            "selected_features": selected_features,
            "accuracy": alpha_stats[0],
            "macro_f1": alpha_stats[1],
            "critical_mdr": alpha_stats[2],
            "hyperparameters": {
                "n_estimators": best_n_est,
                "max_depth": best_max_d,
                "w_crit": best_w_crit
            }
        }

def run_hgwo_pso(dataset="ev", n_wolves=20, max_iter=15):
    if dataset == "ev":
        X, y, feature_names = load_and_preprocess_ev()
        name = "EV Fast-Charging Thermal Runaway"
    elif dataset == "nasa":
        X, y, feature_names = load_and_preprocess_nasa()
        name = "NASA Li-ion Classic 4-Cell"
    elif dataset == "nasa_global":
        X, y, feature_names = load_and_preprocess_nasa_global()
        name = "NASA Global Multi-Temperature Archive"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print("=" * 75)
    print(f"RUNNING: Proposed Hybrid Grey Wolf - PSO Optimizer (HGWO-PSO)")
    print(f"DATASET: {name}")
    print(f"SWARM:   {n_wolves} wolves, {max_iter} iterations (Feature + Hyperparameter Co-Tuning)")
    print("=" * 75)

    t0 = time.time()
    optimizer = HGWO_PSO(n_wolves=n_wolves, max_iter=max_iter)
    res = optimizer.optimize(X, y, feature_names)
    elapsed = time.time() - t0

    hp = res["hyperparameters"]
    print("-" * 75)
    print(f"Optimization Completed in {elapsed:.2f} seconds.")
    print(f"Optimal Sensors ({len(res['selected_features'])}/{len(feature_names)}): {', '.join(res['selected_features'])}")
    print(f"Optimal Hyperparameters: n_estimators={hp['n_estimators']}, max_depth={hp['max_depth']}, w_crit={hp['w_crit']:.1f}")
    print(f"5-Fold CV Accuracy:      {res['accuracy']*100:.2f}%")
    print(f"Macro-F1 Score:          {res['macro_f1']:.4f}")
    print(f"Critical MDR:            {res['critical_mdr']*100:.2f}%")
    print("=" * 75 + "\n")
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ev", "nasa", "nasa_global"], default="ev")
    parser.add_argument("--wolves", type=int, default=20)
    parser.add_argument("--iter", type=int, default=15)
    args = parser.parse_args()
    run_hgwo_pso(args.dataset, n_wolves=args.wolves, max_iter=args.iter)
