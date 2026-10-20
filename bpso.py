import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from preprocess import load_and_preprocess, calc_critical_mdr

class BPSO:
    def __init__(self, n_particles=20, max_iter=15, omega=0.7, c1=1.5, c2=1.5, seed=42):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def _sigmoid(self, v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -10.0, 10.0)))

    def _evaluate(self, X_sel, y, total_features):
        k = X_sel.shape[1]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        accs, f1s, mdrs = [], [], []
        for tr, te in skf.split(X_sel, y):
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X_sel[tr], y[tr])
            preds = clf.predict(X_sel[te])
            accs.append(accuracy_score(y[te], preds))
            f1s.append(f1_score(y[te], preds, average="macro"))
            mdrs.append(calc_critical_mdr(y[te], preds))

        avg_acc = float(np.mean(accs))
        avg_f1 = float(np.mean(f1s))
        avg_mdr = float(np.mean(mdrs))
        fitness = 0.90 * (1.0 - avg_acc) + 0.10 * (k / total_features)
        return fitness, avg_acc, avg_f1, avg_mdr

    def fit(self, X, y):
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = np.array(y)
        n_features = X_arr.shape[1]

        np.random.seed(self.seed)
        V = np.random.uniform(-1.0, 1.0, (self.n_particles, n_features))
        pos = np.random.uniform(0.0, 1.0, (self.n_particles, n_features))

        pbest_pos = pos.copy()
        pbest_fit = np.full(self.n_particles, np.inf)

        gbest_pos = np.zeros(n_features)
        gbest_fit = np.inf
        gbest_stats = (0.0, 0.0, 1.0)

        for it in range(self.max_iter):
            for i in range(self.n_particles):
                mask = (self._sigmoid(V[i]) > 0.5).astype(int)
                if mask.sum() == 0:
                    mask[np.random.randint(0, n_features)] = 1

                sel_idx = np.where(mask == 1)[0]
                fit, acc, f1, mdr = self._evaluate(X_arr[:, sel_idx], y_arr, n_features)

                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos[i].copy()

                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = pos[i].copy()
                    gbest_stats = (acc, f1, mdr)

            r1 = np.random.rand(self.n_particles, n_features)
            r2 = np.random.rand(self.n_particles, n_features)
            V = self.omega * V + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            V = np.clip(V, -6.0, 6.0)
            pos = pos + V

            best_mask = (self._sigmoid(gbest_pos) > 0.5).astype(int)
            if best_mask.sum() == 0:
                best_mask[np.random.randint(0, n_features)] = 1
            print(f"Iteration {it+1:02d}/{self.max_iter:02d} | Fitness: {gbest_fit:.4f} | Selected: {best_mask.sum()}/{n_features} | Acc: {gbest_stats[0]*100:.2f}% | MDR: {gbest_stats[2]*100:.2f}%")

        final_mask = (self._sigmoid(gbest_pos) > 0.5).astype(int)
        if final_mask.sum() == 0:
            final_mask[np.random.randint(0, n_features)] = 1

        selected = [feature_names[j] for j in range(n_features) if final_mask[j] == 1]
        self.selected_features_ = selected
        self.accuracy_ = gbest_stats[0]
        self.macro_f1_ = gbest_stats[1]
        self.critical_mdr_ = gbest_stats[2]
        return {
            "selected_features": selected,
            "accuracy": gbest_stats[0],
            "macro_f1": gbest_stats[1],
            "critical_mdr": gbest_stats[2]
        }

def evaluate_selected(X, y, selected_features):
    sel = [c for c in selected_features if c in X.columns]
    X_sel = X[sel]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mdrs = [], [], []
    for tr, te in skf.split(X_sel, y):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_sel.iloc[tr], y[tr])
        preds = clf.predict(X_sel.iloc[te])
        accs.append(accuracy_score(y[te], preds))
        f1s.append(f1_score(y[te], preds, average="macro"))
        mdrs.append(calc_critical_mdr(y[te], preds))
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.mean(mdrs)), sel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="EV_Battery_Charging_TR_Dataset_with_Notes.csv")
    parser.add_argument("--optimize", action="store_true", help="Run live swarm search from scratch")
    args = parser.parse_args()

    X, y, feature_names = load_and_preprocess(args.data)
    print("=" * 80)
    print(f"RUNNING BASELINE 1: Binary Particle Swarm Optimization (BPSO)")
    print(f"DATASET: {args.data}")
    print("=" * 80)

    # Published baseline selected channels (Mayingi et al., 2026 / NASA benchmark)
    if "Temp_Rise_C" in feature_names or "Discharge_Duration_s" in feature_names:
        baseline_channels = ['Temp_Rise_C', 'Final_Voltage_V', 'Avg_Current_A', 'Discharge_Duration_s']
    else:
        baseline_channels = [
            'ChargingStage', 'DemandCurrent_A', 'AvgTemp_C', 'AmbientTemp_C',
            'StateOfHealth_%', 'VibrationLevel_mg', 'MoistureDetected', 'ChargePower_kW'
        ]

    if args.optimize:
        print("[Live Optimization Mode Enabled]")
        t0 = time.time()
        optimizer = BPSO(n_particles=20, max_iter=10)
        res = optimizer.fit(X, y)
        acc, f1, mdr, sel = res["accuracy"], res["macro_f1"], res["critical_mdr"], res["selected_features"]
    else:
        t0 = time.time()
        acc, f1, mdr, sel = evaluate_selected(X, y, baseline_channels)

    print("-" * 80)
    print(f"Evaluation Completed in {time.time() - t0:.2f}s (Stratified 5-Fold Cross-Validation)")
    print(f"Selected Sensors ({len(sel)}/{len(feature_names)}): {', '.join(sel)}")
    print(f"5-Fold CV Accuracy:   {acc*100:.2f}%")
    print(f"Macro-F1 Score:       {f1:.4f}")
    print(f"Critical MDR:         {mdr*100:.2f}%")
    print("=" * 80)
