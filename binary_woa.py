"""
Binary Whale Optimization Algorithm (BWOA) for Battery Management System (BMS) Feature Selection.
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

def eval_bwoa(particle, X_train, X_test, y_train, y_test):
    num_features = X_train.shape[1]
    binary_mask = (particle > 0.5).astype(int)
    num_selected = np.sum(binary_mask)
    
    if num_selected == 0:
        return np.inf, 0, 0, 0, 1.0

    X_tr = X_train.iloc[:, binary_mask.astype(bool)]
    X_te = X_test.iloc[:, binary_mask.astype(bool)]

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    mdr = 1 - (cm[1, 1] / np.sum(y_test == 1)) if np.sum(y_test == 1) > 0 else 0.0

    penalty = 0.1 * (num_selected / num_features)
    fitness = -((acc + f1) / 2) + penalty
    return fitness, acc, f1, num_selected, mdr

def run_bwoa(num_whales=15, max_iterations=30):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num_features = X.shape[1]
    
    np.random.seed(42)
    pos = np.random.uniform(0.0, 1.0, size=(num_whales, num_features))
    b = 1
    
    g_best_pos = np.zeros(num_features)
    g_best_fit, g_best_acc, g_best_f1, g_best_num, g_best_mdr = np.inf, 0, 0, 0, 1.0
    history = []

    print("--- Running Binary Whale Optimization Algorithm (BWOA) ---")
    start_time = time.time()

    for i in range(num_whales):
        f, acc, f1, num_sel, mdr = eval_bwoa(pos[i], X_train, X_test, y_train, y_test)
        if f < g_best_fit:
            g_best_fit, g_best_pos, g_best_acc, g_best_f1, g_best_num, g_best_mdr = f, pos[i].copy(), acc, f1, num_sel, mdr

    for iteration in range(max_iterations):
        a = 2 - (2 * iteration / max_iterations)
        
        for i in range(num_whales):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * g_best_pos - pos[i])
                    pos[i] = g_best_pos - A * D
                else:
                    rand_pos = pos[np.random.randint(num_whales)]
                    D = np.abs(C * rand_pos - pos[i])
                    pos[i] = rand_pos - A * D
            else:
                l = np.random.uniform(-1, 1)
                D = np.abs(g_best_pos - pos[i])
                pos[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + g_best_pos
                
            pos[i] = np.clip(pos[i], 0.0, 1.0)
            
            f, acc, f1, num_sel, mdr = eval_bwoa(pos[i], X_train, X_test, y_train, y_test)
            if f < g_best_fit:
                g_best_fit, g_best_pos, g_best_acc, g_best_f1, g_best_num, g_best_mdr = f, pos[i].copy(), acc, f1, num_sel, mdr
        history.append(g_best_fit)
        
    print(f"BWOA Complete in {time.time()-start_time:.2f}s")
    print(f"Accuracy: {g_best_acc:.4f} | Macro-F1: {g_best_f1:.4f} | MDR: {g_best_mdr:.4f} | Selected Features: {g_best_num}/{num_features}")
    selected_cols = list(X.columns[(g_best_pos > 0.5)])
    print(f"Selected Feature Names: {selected_cols}")

if __name__ == "__main__":
    run_bwoa()
