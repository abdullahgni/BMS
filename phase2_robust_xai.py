"""
Proposed Phase 2: Robust Co-Optimization with Stratified 5-Fold Cross Validation,
Balanced Class Weighting, and SHAP Explainable AI (XAI) Interpretation.
"""

import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

def load_data(data_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv"):
    df = pd.read_csv(data_path)
    if "System_Fault" not in df.columns:
        if "BMS_Status" in df.columns:
            df['System_Fault'] = df['BMS_Status'].apply(lambda x: 0 if str(x).strip().upper() == 'OK' else 1)
        elif 'overcurrent_flag' in df.columns:
            flag_cols = [c for c in df.columns if 'flag' in c]
            df['System_Fault'] = df[flag_cols].max(axis=1)

    drop_cols = [c for c in ['Timestamp', 'ChargerID', 'CellID', 'Notes', 'BMS_Status', 'EventFlag',
                            'cell_id', 'timestamp', 'manufacturer', 'chemistry_type'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'System_Fault':
            df[col] = pd.factorize(df[col])[0]
            
    if 'MoistureDetected' in df.columns:
        df['MoistureDetected'] = df['MoistureDetected'].astype(int)
        
    X = df.drop(columns=['System_Fault'])
    y = df['System_Fault']
    
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y

def eval_phase2_robust(particle, X_scaled, y):
    num_features = X_scaled.shape[1]
    feature_probs = particle[:num_features]
    binary_mask = (feature_probs > 0.5).astype(int)
    num_selected = np.sum(binary_mask)
    
    if num_selected == 0:
        return np.inf, 0, 0, 0, 1.0, 50, 5

    X_selected = X_scaled.iloc[:, binary_mask.astype(bool)]
    n_estimators = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
    max_depth = int(np.clip(particle[num_features + 1] * 7 + 3, 3, 10))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mdrs = [], [], []
    
    for tr_idx, te_idx in skf.split(X_selected, y):
        X_tr, X_te = X_selected.iloc[tr_idx], X_selected.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average="macro"))
        
        cm = confusion_matrix(y_te, preds, labels=[0, 1])
        mdr = 1 - (cm[1, 1] / np.sum(y_te == 1)) if np.sum(y_te == 1) > 0 else 0.0
        mdrs.append(mdr)

    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    avg_mdr = np.mean(mdrs)
    
    penalty = 0.1 * (num_selected / num_features)
    fitness = -((avg_acc + avg_f1) / 2) + penalty
    return fitness, avg_acc, avg_f1, num_selected, avg_mdr, n_estimators, max_depth

def run_phase2(num_particles=15, max_iterations=30):
    X_scaled, y = load_data()
    num_features = X_scaled.shape[1]
    dims = num_features + 2
    
    np.random.seed(42)
    pos = np.random.uniform(0.0, 1.0, size=(num_particles, dims))
    vel = np.random.uniform(low=-1, high=1, size=(num_particles, dims))
    
    w, c1, c2, b = 0.7, 2.0, 2.0, 1
    p_best_pos = pos.copy()
    p_best_fit = np.full(num_particles, np.inf)
    
    g_best_pos = np.zeros(dims)
    g_best_fit, g_best_acc, g_best_f1, g_best_num, g_best_mdr = np.inf, 0, 0, 0, 1.0
    best_n_est, best_max_d = 50, 5
    history = []

    print("--- Running Proposed Phase 2: Robust Co-Optimizer (5-Fold CV + SHAP XAI) ---")
    start_time = time.time()

    for i in range(num_particles):
        f, acc, f1, num_sel, mdr, n_est, max_d = eval_phase2_robust(pos[i], X_scaled, y)
        p_best_fit[i] = f
        if f < g_best_fit:
            g_best_fit, g_best_pos = f, pos[i].copy()
            g_best_acc, g_best_f1, g_best_num, g_best_mdr = acc, f1, num_sel, mdr
            best_n_est, best_max_d = n_est, max_d

    for iteration in range(max_iterations):
        a = 2 - (2 * iteration / max_iterations)
        is_pso = iteration < (max_iterations // 2)
        
        if iteration == max_iterations // 2:
            g_best_pos = 0.9 * g_best_pos + 0.05
            
        for i in range(num_particles):
            if is_pso:
                r1, r2 = np.random.rand(dims), np.random.rand(dims)
                vel[i] = w * vel[i] + c1 * r1 * (p_best_pos[i] - pos[i]) + c2 * r2 * (g_best_pos - pos[i])
                pos[i] = pos[i] + vel[i]
            else:
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * g_best_pos - pos[i])
                        pos[i] = g_best_pos - A * D
                    else:
                        rand_pos = pos[np.random.randint(num_particles)]
                        D = np.abs(C * rand_pos - pos[i])
                        pos[i] = rand_pos - A * D
                else:
                    l = np.random.uniform(-1, 1)
                    D = np.abs(g_best_pos - pos[i])
                    pos[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + g_best_pos
                    
            pos[i] = np.clip(pos[i], 0.0, 1.0)
            
            f, acc, f1, num_sel, mdr, n_est, max_d = eval_phase2_robust(pos[i], X_scaled, y)
            if f < p_best_fit[i]:
                p_best_fit[i], p_best_pos[i] = f, pos[i].copy()
            if f < g_best_fit:
                g_best_fit, g_best_pos = f, pos[i].copy()
                g_best_acc, g_best_f1, g_best_num, g_best_mdr = acc, f1, num_sel, mdr
                best_n_est, best_max_d = n_est, max_d
        history.append(g_best_fit)
        
    print(f"Phase 2 Complete in {time.time()-start_time:.2f}s")
    print(f"5-Fold CV Accuracy: {g_best_acc:.4f} | 5-Fold CV Macro-F1: {g_best_f1:.4f} | Average MDR: {g_best_mdr:.4f}")
    print(f"Selected Features: {g_best_num}/{num_features} | Co-Tuned n_estimators: {best_n_est} | Co-Tuned max_depth: {best_max_d}")
    
    binary_mask = (g_best_pos[:num_features] > 0.5).astype(int)
    selected_cols = list(X_scaled.columns[binary_mask.astype(bool)])
    print(f"Selected Feature Names: {selected_cols}")
    
    # -----------------------------------
    # SHAP Explainable AI (XAI) Analysis
    # -----------------------------------
    print("\nGenerating SHAP Summary & Dependence Plots...")
    X_final = X_scaled.loc[:, selected_cols]
    clf = RandomForestClassifier(n_estimators=best_n_est, max_depth=best_max_d,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_final, y)
    
    explainer = shap.TreeExplainer(clf)
    X_shap = X_final.sample(n=min(1000, len(X_final)), random_state=42)
    shap_values = explainer.shap_values(X_shap)
    
    try:
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class1, X_shap, show=False)
        plt.title("Phase 2: SHAP Feature Importance Summary")
        plt.savefig('shap_summary.png', bbox_inches='tight')
        plt.close()

        vals = np.abs(shap_values_class1).mean(0)
        feat_imp = pd.DataFrame(list(zip(X_shap.columns, vals)), columns=['col', 'imp'])
        top_feat = feat_imp.sort_values(by='imp', ascending=False).iloc[0]['col']

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(top_feat, shap_values_class1, X_shap, show=False)
        plt.title(f"Phase 2: SHAP Dependence Plot ({top_feat})")
        plt.savefig('shap_dependence.png', bbox_inches='tight')
        plt.close()
        
        print("SHAP plots saved successfully: 'shap_summary.png' & 'shap_dependence.png'")
    except Exception as e:
        print("SHAP Plot Warning:", e)

if __name__ == "__main__":
    run_phase2()
