"""
Hybrid Grey Wolf Optimizer - Particle Swarm Optimization (HGWO-PSO)
for Battery Management System (BMS) Safety-Critical Feature Selection & Parameter Co-Tuning.
Theme: Optimization of BMS in EV while focusing on critical data safety.
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

def load_bms_data(csv_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv"):
    df = pd.read_csv(csv_path)
    
    le = LabelEncoder()
    df['ChargingStage'] = le.fit_transform(df['ChargingStage'])
    df['MoistureDetected'] = df['MoistureDetected'].astype(int)

    # Ground-truth physical battery safety target (mapped from simulation EventFlag):
    # 0: Critical (Runaway - catastrophic thermal hazard)
    # 1: OK (Watch - safe normal operation)
    # 2: Warning (Alarm + Warning - elevated thermal/electrical stress)
    target_map = {'Runaway': 0, 'Watch': 1, 'Alarm': 2, 'Warning': 2}
    y = df['EventFlag'].map(target_map).values

    drop_cols = ['Timestamp', 'ChargerID', 'CellID', 'Notes', 'BMS_Status', 'EventFlag', 'TR_Probability']
    X = df.drop(columns=drop_cols)
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    return X_scaled, y, feature_names

def calc_critical_mdr(y_true, y_pred):
    true_crit = (y_true == 0)
    pred_non_crit = (y_pred != 0)
    missed = np.sum(true_crit & pred_non_crit)
    total_crit = np.sum(true_crit)
    return missed / total_crit if total_crit > 0 else 0.0

class HGWO_PSO_Optimizer:
    def __init__(self, X, y, feature_names, num_wolves=20, max_iter=20, seed=42):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.num_features = X.shape[1]
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.seed = seed
        self.dims = self.num_features + 3 # features + n_estimators + max_depth + crit_weight

    def evaluate(self, particle):
        mask = (particle[:self.num_features] > 0.5).astype(int)
        k = np.sum(mask)
        if k == 0:
            return 999.0, 0.0, 0.0, 1.0, 0

        # Sensor budget preference: 5 to 8 sensors
        sensor_penalty = 0.0
        if k < 5:
            sensor_penalty = (5 - k) * 0.05
        elif k > 8:
            sensor_penalty = (k - 8) * 0.05

        n_est = int(np.clip(particle[self.num_features] * 80 + 40, 40, 120))
        max_d = int(np.clip(particle[self.num_features + 1] * 6 + 4, 4, 10))
        w_crit = float(np.clip(particle[self.num_features + 2] * 3.0 + 1.5, 1.5, 4.5))

        X_sel = self.X.iloc[:, mask.astype(bool)]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        
        accs, f1s, mdrs = [], [], []
        cw = {0: w_crit, 1: 1.0, 2: 1.0}
        
        for tr, te in skf.split(X_sel, self.y):
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight=cw, random_state=self.seed, n_jobs=1)
            clf.fit(X_sel.iloc[tr], self.y[tr])
            preds = clf.predict(X_sel.iloc[te])
            accs.append(accuracy_score(self.y[te], preds))
            f1s.append(f1_score(self.y[te], preds, average="macro"))
            mdrs.append(calc_critical_mdr(self.y[te], preds))

        avg_acc = np.mean(accs)
        avg_f1 = np.mean(f1s)
        avg_mdr = np.mean(mdrs)

        # Multi-objective safety fitness: prioritize low MDR, high accuracy/F1, and preferred sensor footprint
        fitness = -(0.5 * avg_acc + 0.5 * avg_f1) + 0.35 * avg_mdr + 0.05 * (k / self.num_features) + sensor_penalty
        return fitness, avg_acc, avg_f1, avg_mdr, k

    def optimize(self):
        np.random.seed(self.seed)
        pos = np.random.uniform(0.0, 1.0, size=(self.num_wolves, self.dims))
        vel = np.random.uniform(-0.5, 0.5, size=(self.num_wolves, self.dims))
        
        alpha_pos = np.zeros(self.dims)
        alpha_fit = np.inf
        alpha_stats = (0, 0, 1.0, 0)
        
        beta_pos = np.zeros(self.dims)
        beta_fit = np.inf
        
        delta_pos = np.zeros(self.dims)
        delta_fit = np.inf
        
        pbest_pos = pos.copy()
        pbest_fit = np.full(self.num_wolves, np.inf)

        print(f"Starting HGWO-PSO optimization ({self.num_wolves} wolves, {self.max_iter} iterations)...", flush=True)
        start_time = time.time()

        for it in range(self.max_iter):
            a = 2.0 - 2.0 * (it / self.max_iter)
            w = 0.9 - 0.5 * (it / self.max_iter)

            for i in range(self.num_wolves):
                fit, acc, f1, mdr, k = self.evaluate(pos[i])
                
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

            for i in range(self.num_wolves):
                r1, r2, r3 = np.random.rand(self.dims), np.random.rand(self.dims), np.random.rand(self.dims)
                
                D_alpha = np.abs(2 * r1 * alpha_pos - pos[i])
                X1 = alpha_pos - (2 * a * np.random.rand(self.dims) - a) * D_alpha
                
                D_beta = np.abs(2 * r2 * beta_pos - pos[i])
                X2 = beta_pos - (2 * a * np.random.rand(self.dims) - a) * D_beta
                
                D_delta = np.abs(2 * r3 * delta_pos - pos[i])
                X3 = delta_pos - (2 * a * np.random.rand(self.dims) - a) * D_delta
                
                X_gwo = (X1 + X2 + X3) / 3.0
                vel[i] = w * vel[i] + 1.5 * r1 * (X1 - pos[i]) + 1.5 * r2 * (X2 - pos[i]) + 1.5 * r3 * (X3 - pos[i])
                pos[i] = 0.5 * (pos[i] + vel[i]) + 0.5 * X_gwo
                pos[i] = np.clip(pos[i], 0.0, 1.0)

            print(f"Iter {it+1:02d}/{self.max_iter} | Best Fit: {alpha_fit:.4f} | Acc: {alpha_stats[0]*100:.2f}% | F1: {alpha_stats[1]:.4f} | Crit-MDR: {alpha_stats[2]*100:.2f}% | Sensors: {alpha_stats[3]}/{self.num_features}", flush=True)

        elapsed = time.time() - start_time
        mask = (alpha_pos[:self.num_features] > 0.5).astype(int)
        sel_sensors = [self.feature_names[j] for j in range(self.num_features) if mask[j]]
        n_est = int(np.clip(alpha_pos[self.num_features] * 80 + 40, 40, 120))
        max_d = int(np.clip(alpha_pos[self.num_features + 1] * 6 + 4, 4, 10))
        w_crit = float(np.clip(alpha_pos[self.num_features + 2] * 3.0 + 1.5, 1.5, 4.5))

        return {
            "Model": "Proposed HGWO-PSO",
            "Classifier": f"RandomForest (Co-Tuned: {n_est} trees, depth {max_d})",
            "Accuracy": round(alpha_stats[0], 4),
            "Macro-F1": round(alpha_stats[1], 4),
            "MDR (Critical Class)": round(alpha_stats[2], 4),
            "Selected Sensors": f"{len(sel_sensors)}/{self.num_features}",
            "Sensor Names": ", ".join(sel_sensors),
            "Elapsed_s": round(elapsed, 1),
            "Hyperparams": {"n_estimators": n_est, "max_depth": max_d, "crit_weight": round(w_crit, 2)}
        }

if __name__ == "__main__":
    X, y, f_names = load_bms_data()
    opt = HGWO_PSO_Optimizer(X, y, f_names, num_wolves=20, max_iter=20, seed=42)
    res = opt.optimize()
    print("\n--- OPTIMIZATION RESULT ---")
    for k, v in res.items():
        print(f"{k}: {v}")
