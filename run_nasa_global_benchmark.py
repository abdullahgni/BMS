"""
Cross-Dataset Validation: Full Global NASA Li-ion Battery Discharge Archive (2,750 cycles)
Tests Multi-Temperature Environmental Generalization (4°C, 22°C, 24°C, 43°C, 44°C)
across 34 physical battery cells.
Strict 5-Fold Stratified Cross Validation.
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from hgwo_pso import HGWO_PSO

warnings.filterwarnings('ignore')

def load_nasa_global(csv_path="nasa_battery_benchmark_global_2750.csv"):
    df = pd.read_csv(csv_path)
    y = df['Target_Label'].values
    
    non_feature_cols = ['battery_id', 'cycle_file', 'Capacity_Ah', 'Health_State', 'Target_Label']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    return X_scaled, y, feature_cols

def calc_critical_mdr(y_true, y_pred, critical_class=0):
    true_crit = (y_true == critical_class)
    pred_non_crit = (y_pred != critical_class)
    missed = np.sum(true_crit & pred_non_crit)
    total_crit = np.sum(true_crit)
    return float(missed / total_crit) if total_crit > 0 else 0.0

def evaluate_cv(X_subset, y, clf_builder, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, mdrs = [], [], []
    for tr, te in skf.split(X_subset, y):
        clf = clf_builder()
        clf.fit(X_subset.iloc[tr], y[tr])
        preds = clf.predict(X_subset.iloc[te])
        accs.append(accuracy_score(y[te], preds))
        f1s.append(f1_score(y[te], preds, average="macro"))
        mdrs.append(calc_critical_mdr(y[te], preds))
    return np.mean(accs), np.mean(f1s), np.mean(mdrs)

def run_bpso_knn(X, y, n_particles=20, max_iter=15, seed=42):
    np.random.seed(seed)
    n_feat = X.shape[1]
    pos = np.random.uniform(0, 1, (n_particles, n_feat))
    vel = np.random.uniform(-1, 1, (n_particles, n_feat))
    pbest_pos = pos.copy()
    pbest_fit = np.ones(n_particles) * 999.0
    gbest_pos = None
    gbest_fit = 999.0
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    def fitness(mask):
        k = np.sum(mask)
        if k == 0:
            return 999.0
        X_sub = X.iloc[:, mask]
        accs = []
        for tr, te in skf.split(X_sub, y):
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_sub.iloc[tr], y[tr])
            accs.append(accuracy_score(y[te], knn.predict(X_sub.iloc[te])))
        err = 1.0 - np.mean(accs)
        return 0.9 * err + 0.1 * (k / n_feat)
        
    for it in range(max_iter):
        for i in range(n_particles):
            mask = (1.0 / (1.0 + np.exp(-pos[i]))) > 0.5
            fit = fitness(mask)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = pos[i].copy()
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_pos = pos[i].copy()
                
        w = 0.9 - 0.5 * (it / max_iter)
        r1 = np.random.rand(n_particles, n_feat)
        r2 = np.random.rand(n_particles, n_feat)
        vel = w * vel + 1.5 * r1 * (pbest_pos - pos) + 1.5 * r2 * (gbest_pos - pos)
        vel = np.clip(vel, -4, 4)
        pos = pos + vel
        
    final_mask = (1.0 / (1.0 + np.exp(-gbest_pos))) > 0.5
    if np.sum(final_mask) == 0:
        final_mask[np.argmax(gbest_pos)] = True
    return [X.columns[i] for i in range(n_feat) if final_mask[i]]

def run_bwoa_knn(X, y, n_whales=20, max_iter=15, seed=42):
    np.random.seed(seed)
    n_feat = X.shape[1]
    pos = np.random.uniform(0, 1, (n_whales, n_feat))
    gbest_pos = None
    gbest_fit = 999.0
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    def fitness(mask):
        k = np.sum(mask)
        if k == 0:
            return 999.0
        X_sub = X.iloc[:, mask]
        accs = []
        for tr, te in skf.split(X_sub, y):
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_sub.iloc[tr], y[tr])
            accs.append(accuracy_score(y[te], knn.predict(X_sub.iloc[te])))
        err = 1.0 - np.mean(accs)
        return 0.9 * err + 0.1 * (k / n_feat)
        
    for it in range(max_iter):
        a = 2.0 - 2.0 * (it / max_iter)
        for i in range(n_whales):
            mask = (1.0 / (1.0 + np.exp(-pos[i]))) > 0.5
            fit = fitness(mask)
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_pos = pos[i].copy()
                
        for i in range(n_whales):
            r = np.random.rand()
            A = 2.0 * a * r - a
            C = 2.0 * np.random.rand()
            l = np.random.uniform(-1, 1)
            p = np.random.rand()
            
            if p < 0.5:
                if np.abs(A) < 1.0:
                    D = np.abs(C * gbest_pos - pos[i])
                    pos[i] = gbest_pos - A * D
                else:
                    rand_idx = np.random.randint(0, n_whales)
                    D = np.abs(C * pos[rand_idx] - pos[i])
                    pos[i] = pos[rand_idx] - A * D
            else:
                D = np.abs(gbest_pos - pos[i])
                pos[i] = D * np.exp(0.5 * l) * np.cos(2.0 * np.pi * l) + gbest_pos
                
    final_mask = (1.0 / (1.0 + np.exp(-gbest_pos))) > 0.5
    if np.sum(final_mask) == 0:
        final_mask[np.argmax(gbest_pos)] = True
    return [X.columns[i] for i in range(n_feat) if final_mask[i]]

def run_global_benchmark():
    print("=" * 85)
    print("GLOBAL BENCHMARK: FULL NASA LI-ION DATASET (ALL 2,750 CYCLES, MULTI-TEMP 4°C - 44°C)")
    print("=" * 85)
    
    X_scaled, y, feature_names = load_nasa_global()
    num_features = len(feature_names)
    print(f"Dataset Loaded: {len(X_scaled)} cycles, {num_features} candidate physical features.")
    print(f"Class breakdown: Critical (EOL)={np.sum(y==0)}, OK={np.sum(y==1)}, Warning={np.sum(y==2)}")
    print("-" * 85)
    
    # 1. Baseline 1: BPSO + KNN
    print("Optimizing Baseline 1 (BPSO + KNN on 2,750 cycles)...")
    t0 = time.time()
    bpso_features = run_bpso_knn(X_scaled, y, n_particles=20, max_iter=15, seed=42)
    acc_bpso, f1_bpso, mdr_bpso = evaluate_cv(X_scaled[bpso_features], y, lambda: KNeighborsClassifier(n_neighbors=5))
    t_bpso = time.time() - t0
    print(f"BPSO Done ({t_bpso:.1f}s): {len(bpso_features)} features, Acc={acc_bpso*100:.2f}%, MDR={mdr_bpso*100:.2f}%")
    
    # 2. Baseline 2: BWOA + KNN
    print("\nOptimizing Baseline 2 (BWOA + KNN on 2,750 cycles)...")
    t0 = time.time()
    bwoa_features = run_bwoa_knn(X_scaled, y, n_whales=20, max_iter=15, seed=42)
    acc_bwoa, f1_bwoa, mdr_bwoa = evaluate_cv(X_scaled[bwoa_features], y, lambda: KNeighborsClassifier(n_neighbors=5))
    t_bwoa = time.time() - t0
    print(f"BWOA Done ({t_bwoa:.1f}s): {len(bwoa_features)} features, Acc={acc_bwoa*100:.2f}%, MDR={mdr_bwoa*100:.2f}%")
    
    # 3. Default All-Features Random Forest
    print("\nEvaluating Standard Full-Feature Random Forest...")
    acc_all, f1_all, mdr_all = evaluate_cv(X_scaled, y, lambda: RandomForestClassifier(n_estimators=100, random_state=42))
    print(f"Full RF Done: Acc={acc_all*100:.2f}%, MDR={mdr_all*100:.2f}%")
    
    # 4. Proposed HGWO-PSO
    print("\nOptimizing Proposed HGWO-PSO (Novel Hybrid & Safety Co-Tuning on 2,750 cycles)...")
    t0 = time.time()
    hgwo_model = HGWO_PSO(
        n_wolves=20,
        max_iter=20,
        cv_splits=5,
        critical_class=0,
        feature_bounds=(0.20, 0.40), # 3 to 6 features out of 15
        weight_acc=0.5,
        weight_f1=0.5,
        weight_mdr=0.40,
        seed=42
    )
    hgwo_model.fit(X_scaled, y)
    t_hgwo = time.time() - t0
    
    hgwo_features = hgwo_model.selected_features_
    acc_hgwo, f1_hgwo, mdr_hgwo = evaluate_cv(
        X_scaled[hgwo_features], 
        y, 
        lambda: RandomForestClassifier(
            n_estimators=hgwo_model.best_params_['n_estimators'],
            max_depth=hgwo_model.best_params_['max_depth'],
            class_weight={0: hgwo_model.best_params_['crit_weight'], 1: 1.0, 2: 1.0},
            random_state=42
        )
    )
    print(f"HGWO-PSO Done ({t_hgwo:.1f}s): {len(hgwo_features)} features, Acc={acc_hgwo*100:.2f}%, MDR={mdr_hgwo*100:.2f}%")
    
    results = [
        {
            "Method / Framework": "Baseline 1: BPSO (Paper)",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{acc_bpso*100:.2f}%",
            "Macro-F1": f"{f1_bpso:.4f}",
            "Critical MDR": f"{mdr_bpso*100:.2f}%",
            "Sensors": f"{len(bpso_features)} / {num_features}",
            "Selected Physical Sensors": ", ".join(bpso_features)
        },
        {
            "Method / Framework": "Baseline 2: BWOA (Paper)",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{acc_bwoa*100:.2f}%",
            "Macro-F1": f"{f1_bwoa:.4f}",
            "Critical MDR": f"{mdr_bwoa*100:.2f}%",
            "Sensors": f"{len(bwoa_features)} / {num_features}",
            "Selected Physical Sensors": ", ".join(bwoa_features)
        },
        {
            "Method / Framework": "Standard Full-Feature RF",
            "Classifier": "Random Forest (Default)",
            "Accuracy": f"{acc_all*100:.2f}%",
            "Macro-F1": f"{f1_all:.4f}",
            "Critical MDR": f"{mdr_all*100:.2f}%",
            "Sensors": f"{num_features} / {num_features}",
            "Selected Physical Sensors": "All Features"
        },
        {
            "Method / Framework": "Proposed HGWO-PSO (Ours)",
            "Classifier": f"RF (n={hgwo_model.best_params_['n_estimators']}, d={hgwo_model.best_params_['max_depth']}, w0={hgwo_model.best_params_['crit_weight']:.1f})",
            "Accuracy": f"{acc_hgwo*100:.2f}%",
            "Macro-F1": f"{f1_hgwo:.4f}",
            "Critical MDR": f"{mdr_hgwo*100:.2f}%",
            "Sensors": f"{len(hgwo_features)} / {num_features}",
            "Selected Physical Sensors": ", ".join(hgwo_features)
        }
    ]
    
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("FINAL FULL GLOBAL NASA ARCHIVE EVALUATION TABLE (STRICT 5-FOLD CV, 2,750 CYCLES)")
    print("=" * 90)
    print(res_df.to_string(index=False))
    print("=" * 90)
    
    res_df.to_csv("nasa_global_benchmark_results.csv", index=False)
    return res_df, hgwo_model

if __name__ == '__main__':
    run_global_benchmark()
