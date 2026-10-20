import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from preprocess import load_and_preprocess_ev, load_and_preprocess_nasa, load_and_preprocess_nasa_global, calc_critical_mdr

class BHPWOA_Optimizer:
    def __init__(self, n_agents=20, max_iter=15, omega=0.7, c1=1.5, c2=1.5, b=1.0, seed=42):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.b = b
        self.seed = seed

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _fitness(self, X_sel, y, total_features):
        k = X_sel.shape[1]
        if k == 0:
            return 999.0, 0.0, 0.0, 1.0

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
        score = 0.90 * (1.0 - avg_acc) + 0.10 * (k / total_features)
        return score, avg_acc, avg_f1, avg_mdr

    def optimize(self, X, y, feature_names):
        np.random.seed(self.seed)
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = np.array(y)
        n_features = X_arr.shape[1]

        pop = np.random.uniform(-1.0, 1.0, (self.n_agents, n_features))
        V = np.random.uniform(-0.5, 0.5, (self.n_agents, n_features))

        pbest_pos = pop.copy()
        pbest_fit = np.full(self.n_agents, np.inf)

        leader_pos = np.zeros(n_features)
        leader_fit = np.inf
        leader_stats = (0.0, 0.0, 1.0)

        for it in range(self.max_iter):
            for i in range(self.n_agents):
                mask = (self._sigmoid(pop[i]) > 0.5).astype(int)
                if np.sum(mask) == 0:
                    mask[np.random.randint(0, n_features)] = 1

                sel_idx = np.where(mask == 1)[0]
                fit, acc, f1, mdr = self._fitness(X_arr[:, sel_idx], y_arr, n_features)

                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pop[i].copy()

                if fit < leader_fit:
                    leader_fit = fit
                    leader_pos = pop[i].copy()
                    leader_stats = (acc, f1, mdr)

            for i in range(self.n_agents):
                p = np.random.rand()
                if p < 0.5:
                    r1 = np.random.rand(n_features)
                    r2 = np.random.rand(n_features)
                    V[i] = self.omega * V[i] + self.c1 * r1 * (pbest_pos[i] - pop[i]) + self.c2 * r2 * (leader_pos - pop[i])
                    V[i] = np.clip(V[i], -6.0, 6.0)
                    pop[i] = pop[i] + V[i]
                else:
                    l = np.random.uniform(-1.0, 1.0, n_features)
                    D_prime = np.abs(leader_pos - pop[i])
                    pop[i] = D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l) + leader_pos

            best_mask = (self._sigmoid(leader_pos) > 0.5).astype(int)
            if np.sum(best_mask) == 0:
                best_mask[np.random.randint(0, n_features)] = 1
            print(f"Iteration {it+1:02d}/{self.max_iter:02d} | Best Fitness: {leader_fit:.4f} | Selected: {np.sum(best_mask)}/{n_features} | Acc: {leader_stats[0]*100:.2f}% | MDR: {leader_stats[2]*100:.2f}%")

        final_mask = (self._sigmoid(leader_pos) > 0.5).astype(int)
        if np.sum(final_mask) == 0:
            final_mask[np.random.randint(0, n_features)] = 1

        selected_features = [feature_names[j] for j in range(n_features) if final_mask[j] == 1]
        return {
            "selected_features": selected_features,
            "accuracy": leader_stats[0],
            "macro_f1": leader_stats[1],
            "critical_mdr": leader_stats[2]
        }

def run_bhpwoa(dataset="ev", n_agents=20, max_iter=15):
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
    print(f"RUNNING: Binary Hybrid PSO-WOA (BHPWOA)")
    print(f"DATASET: {name}")
    print(f"SWARM:   {n_agents} agents, {max_iter} iterations")
    print("=" * 75)

    t0 = time.time()
    optimizer = BHPWOA_Optimizer(n_agents=n_agents, max_iter=max_iter)
    res = optimizer.optimize(X, y, feature_names)
    elapsed = time.time() - t0

    print("-" * 75)
    print(f"Optimization Completed in {elapsed:.2f} seconds.")
    print(f"Selected Sensors ({len(res['selected_features'])}/{len(feature_names)}): {', '.join(res['selected_features'])}")
    print(f"5-Fold CV Accuracy:   {res['accuracy']*100:.2f}%")
    print(f"Macro-F1 Score:       {res['macro_f1']:.4f}")
    print(f"Critical MDR:         {res['critical_mdr']*100:.2f}%")
    print("=" * 75 + "\n")
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ev", "nasa", "nasa_global"], default="ev")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--iter", type=int, default=15)
    args = parser.parse_args()
    run_bhpwoa(args.dataset, n_agents=args.agents, max_iter=args.iter)
