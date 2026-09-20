import pandas as pd
import numpy as np
import time
import warnings

print("Script starting...", flush=True)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# 1. Load Original 18-feature Dataset
df = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes.csv')
df['System_Fault'] = df['BMS_Status'].apply(lambda x: 0 if str(x).strip().upper() == 'OK' else 1)

drop_cols = ['Timestamp', 'ChargerID', 'CellID', 'Notes', 'BMS_Status', 'EventFlag']
df = df.drop(columns=drop_cols, errors='ignore')

for col in df.select_dtypes(include=['object']).columns:
    if col != 'System_Fault':
        df[col] = pd.factorize(df[col])[0]

if 'MoistureDetected' in df.columns:
    df['MoistureDetected'] = df['MoistureDetected'].astype(int)

X = df.drop(columns=['System_Fault'])
y = df['System_Fault']
num_features = X.shape[1]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

num_particles = 15
max_iterations = 30
w, c1, c2, b = 0.7, 2.0, 2.0, 1

def eval_std(particle, co_opt=False):
    mask = (particle[:num_features] > 0.5).astype(int)
    num_sel = np.sum(mask)
    if num_sel == 0: return np.inf, 0, 0, 0, 1.0
    
    X_tr = X_train.iloc[:, mask.astype(bool)]
    X_te = X_test.iloc[:, mask.astype(bool)]
    
    if co_opt:
        n_est = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
        max_d = int(np.clip(particle[num_features+1] * 7 + 3, 3, 10))
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=1)
    else:
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    mdr = 1 - (cm[1, 1] / np.sum(y_test == 1)) if np.sum(y_test == 1) > 0 else 0.0
    
    fitness = -((acc + f1)/2) + 0.1 * (num_sel / num_features)
    return fitness, acc, f1, num_sel, mdr

def eval_robust(particle):
    mask = (particle[:num_features] > 0.5).astype(int)
    num_sel = np.sum(mask)
    if num_sel == 0: return np.inf, 0, 0, 0, 1.0
    
    X_sel = X_scaled.iloc[:, mask.astype(bool)]
    n_est = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
    max_d = int(np.clip(particle[num_features+1] * 7 + 3, 3, 10))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mdrs = [], [], []
    for tr, te in skf.split(X_sel, y):
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42, n_jobs=1)
        clf.fit(X_sel.iloc[tr], y.iloc[tr])
        preds = clf.predict(X_sel.iloc[te])
        accs.append(accuracy_score(y.iloc[te], preds))
        f1s.append(f1_score(y.iloc[te], preds, average="macro"))
        cm = confusion_matrix(y.iloc[te], preds, labels=[0, 1])
        mdrs.append(1 - (cm[1, 1] / np.sum(y.iloc[te] == 1)) if np.sum(y.iloc[te] == 1) > 0 else 0.0)
        
    avg_acc, avg_f1, avg_mdr = np.mean(accs), np.mean(f1s), np.mean(mdrs)
    fitness = -((avg_acc + avg_f1)/2) + 0.1 * (num_sel / num_features)
    return fitness, avg_acc, avg_f1, num_sel, avg_mdr

def run_alg(name, fit_fn, co_opt=False, mode="pso"):
    dims = num_features + 2 if co_opt else num_features
    np.random.seed(42)
    pos = np.random.uniform(0, 1, size=(num_particles, dims))
    vel = np.random.uniform(-1, 1, size=(num_particles, dims))
    p_best_pos, p_best_fit = pos.copy(), np.full(num_particles, np.inf)
    g_best_pos, g_best_fit = np.zeros(dims), np.inf
    g_best_acc = g_best_f1 = g_best_num = 0
    g_best_mdr = 1.0
    
    for i in range(num_particles):
        f, acc, f1, num, mdr = fit_fn(pos[i])
        p_best_fit[i] = f
        if f < g_best_fit:
            g_best_fit, g_best_pos, g_best_acc, g_best_f1, g_best_num, g_best_mdr = f, pos[i].copy(), acc, f1, num, mdr
            
    print(f"\n--- Running {name} ---", flush=True)
    for iteration in range(max_iterations):
        a = 2 - 2 * iteration / max_iterations
        for i in range(num_particles):
            is_pso = mode == "pso" or (mode == "hybrid" and iteration < max_iterations // 2)
            if mode == "hybrid" and iteration == max_iterations // 2 and i == 0:
                g_best_pos = 0.9 * g_best_pos + 0.05
                
            if is_pso:
                r1, r2 = np.random.rand(dims), np.random.rand(dims)
                vel[i] = w * vel[i] + c1 * r1 * (p_best_pos[i] - pos[i]) + c2 * r2 * (g_best_pos - pos[i])
                pos[i] = pos[i] + vel[i]
            else:
                r1, r2, p = np.random.rand(), np.random.rand(), np.random.rand()
                A, C = 2*a*r1 - a, 2*r2
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
                    
            pos[i] = np.clip(pos[i], 0, 1)
            f, acc, f1, num, mdr = fit_fn(pos[i])
            if f < p_best_fit[i]: p_best_fit[i], p_best_pos[i] = f, pos[i].copy()
            if f < g_best_fit:
                g_best_fit, g_best_pos, g_best_acc, g_best_f1, g_best_num, g_best_mdr = f, pos[i].copy(), acc, f1, num, mdr
                
    print(f"Done {name} | F1: {g_best_f1:.4f} | MDR: {g_best_mdr:.4f} | Sensors: {g_best_num}/{num_features}", flush=True)
    return {"Model": name, "Accuracy": round(g_best_acc, 4), "Macro-F1": round(g_best_f1, 4), "MDR (Critical)": round(g_best_mdr, 4), "Selected Sensors": f"{g_best_num}/{num_features}"}

res = []
res.append(run_alg("1. BPSO", lambda p: eval_std(p, False), False, "pso"))
res.append(run_alg("2. BWOA", lambda p: eval_std(p, False), False, "woa"))
res.append(run_alg("3. BHPWOA (Paper)", lambda p: eval_std(p, False), False, "hybrid"))
res.append(run_alg("4. Phase 1 (Co-Opt)", lambda p: eval_std(p, True), True, "hybrid"))
res.append(run_alg("5. Phase 2 (Proposed)", eval_robust, True, "hybrid"))

df_out = pd.DataFrame(res)
print(df_out.to_string(index=False))
df_out.to_csv("og_dataset_benchmark.csv", index=False)
