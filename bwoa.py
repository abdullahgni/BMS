import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from preprocess import load_and_preprocess, calc_critical_mdr

class BWOA:
    def __init__(self, n_whales=20, max_iter=15, b=1.0, seed=42):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.b = b
        self.seed = seed

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

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
        pop = np.random.uniform(-1.0, 1.0, (self.n_whales, n_features))
        leader_pos = np.zeros(n_features)
        leader_fit = np.inf
        leader_stats = (0.0, 0.0, 1.0)

        for it in range(self.max_iter):
            a = 2.0 - 2.0 * (it / self.max_iter)

            for i in range(self.n_whales):
                mask = (self._sigmoid(pop[i]) > 0.5).astype(int)
                if mask.sum() == 0:
                    mask[np.random.randint(0, n_features)] = 1

                sel_idx = np.where(mask == 1)[0]
                fit, acc, f1, mdr = self._evaluate(X_arr[:, sel_idx], y_arr, n_features)

                if fit < leader_fit:
                    leader_fit = fit
                    leader_pos = pop[i].copy()
                    leader_stats = (acc, f1, mdr)

            for i in range(self.n_whales):
                r1 = np.random.rand(n_features)
                r2 = np.random.rand(n_features)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = np.random.rand()
                l = np.random.uniform(-1.0, 1.0, n_features)

                if p < 0.5:
                    if np.mean(np.abs(A)) < 1.0:
                        D = np.abs(C * leader_pos - pop[i])
                        pop[i] = leader_pos - A * D
                    else:
                        rand_idx = np.random.randint(0, self.n_whales)
                        D = np.abs(C * pop[rand_idx] - pop[i])
                        pop[i] = pop[rand_idx] - A * D
                else:
                    D_prime = np.abs(leader_pos - pop[i])
                    pop[i] = D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l) + leader_pos

            best_mask = (self._sigmoid(leader_pos) > 0.5).astype(int)
            if best_mask.sum() == 0:
                best_mask[np.random.randint(0, n_features)] = 1
            print(f"Iteration {it+1:02d}/{self.max_iter:02d} | Fitness: {leader_fit:.4f} | Selected: {best_mask.sum()}/{n_features} | Acc: {leader_stats[0]*100:.2f}% | MDR: {leader_stats[2]*100:.2f}%")

        final_mask = (self._sigmoid(leader_pos) > 0.5).astype(int)
        if final_mask.sum() == 0:
            final_mask[np.random.randint(0, n_features)] = 1

        selected = [feature_names[j] for j in range(n_features) if final_mask[j] == 1]
        self.selected_features_ = selected
        self.accuracy_ = leader_stats[0]
        self.macro_f1_ = leader_stats[1]
        self.critical_mdr_ = leader_stats[2]
        return {
            "selected_features": selected,
            "accuracy": leader_stats[0],
            "macro_f1": leader_stats[1],
            "critical_mdr": leader_stats[2]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="EV_Battery_Charging_TR_Dataset_with_Notes.csv")
    parser.add_argument("--whales", type=int, default=20)
    parser.add_argument("--iter", type=int, default=15)
    args = parser.parse_args()

    X, y, feature_names = load_and_preprocess(args.data)
    print("=" * 75)
    print(f"RUNNING BWOA ON: {args.data}")
    print("=" * 75)
    t0 = time.time()
    optimizer = BWOA(n_whales=args.whales, max_iter=args.iter)
    res = optimizer.fit(X, y)
    print("-" * 75)
    print(f"Elapsed: {time.time() - t0:.2f}s")
    print(f"Selected ({len(res['selected_features'])}/{len(feature_names)}): {', '.join(res['selected_features'])}")
    print(f"Accuracy: {res['accuracy']*100:.2f}% | Macro-F1: {res['macro_f1']:.4f} | Critical MDR: {res['critical_mdr']*100:.2f}%")
    print("=" * 75)
