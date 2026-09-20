"""
Proposed Phase 1: Joint Co-Optimization of Feature Selection & Classifier Hyperparameters using BHPWOA.
Particle Vector Dimensions: [Feature_1, Feature_2, ..., Feature_N, n_estimators, max_depth]
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

def eval_phase1(particle, X_train, X_test, y_train, y_test):
    num_features = X_train.shape[1]
    feature_probs = particle[:num_features]
    binary_mask = (feature_probs > 0.5).astype(int)
    num_selected = np.sum(binary_mask)
    
    if num_selected == 0:
        return np.inf, 0, 0, 0, 1.0, 50, 5

    # Decode hyperparameters from extended dimensions
    n_estimators = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
    max_depth = int(np.clip(particle[num_features + 1] * 7 + 3, 3, 10))

    X_tr = X_train.iloc[:, binary_mask.astype(bool)]
    X_te = X_test.iloc[:, binary_mask.astype(bool)]

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    mdr = 1 - (cm[1, 1] / np.sum(y_test == 1)) if np.sum(y_test == 1) > 0 else 0.0

    penalty = 0.1 * (num_selected / num_features)
    fitness = -((acc + f1) / 2) + penalty
    return fitness, acc, f1, num_selected, mdr, n_estimators, max_depth

def run_phase1(num_particles=15, max_iterations=30):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num_features = X.shape[1]
    dims = num_features + 2  # Features + n_estimators + max_depth
    
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

    print("--- Running Proposed Phase 1: Joint Co-Optimization ---")
    start_time = time.time()

    for i in range(num_particles):
        f, acc, f1, num_sel, mdr, n_est, max_d = eval_phase1(pos[i], X_train, X_test, y_train, y_test)
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
            
            f, acc, f1, num_sel, mdr, n_est, max_d = eval_phase1(pos[i], X_train, X_test, y_train, y_test)
            if f < p_best_fit[i]:
                p_best_fit[i], p_best_pos[i] = f, pos[i].copy()
            if f < g_best_fit:
                g_best_fit, g_best_pos = f, pos[i].copy()
                g_best_acc, g_best_f1, g_best_num, g_best_mdr = acc, f1, num_sel, mdr
                best_n_est, best_max_d = n_est, max_d
        history.append(g_best_fit)
        
    print(f"Phase 1 Complete in {time.time()-start_time:.2f}s")
    print(f"Accuracy: {g_best_acc:.4f} | Macro-F1: {g_best_f1:.4f} | MDR: {g_best_mdr:.4f}")
    print(f"Selected Features: {g_best_num}/{num_features} | Co-Tuned n_estimators: {best_n_est} | Co-Tuned max_depth: {best_max_d}")
    selected_cols = list(X.columns[(g_best_pos[:num_features] > 0.5)])
    print(f"Selected Feature Names: {selected_cols}")

if __name__ == "__main__":
    run_phase1()
