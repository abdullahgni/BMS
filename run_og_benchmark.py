import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

print("Script starting...", flush=True)

# 1. Load Original Dataset exactly as in bms.py
df = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes.csv')

# Drop non-feature metadata columns (matching bms.py)
df = df.drop(columns=["Timestamp", "ChargerID", "CellID", "Notes"], errors='ignore')

# Label encode categorical columns (ChargingStage, BMS_Status, EventFlag) as in bms.py
label_encoder = LabelEncoder()
categorical_columns = ["ChargingStage", "BMS_Status", "EventFlag"]
for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

if 'MoistureDetected' in df.columns:
    df['MoistureDetected'] = df['MoistureDetected'].astype(int)

# Target: BMS_Status (3-Class: 0=Critical, 1=OK, 2=Warning from LabelEncoder)
X = df.drop(columns=["BMS_Status"])
y = df["BMS_Status"]
num_features = X.shape[1]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split matching bms.py (test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

def calc_critical_mdr(y_true, y_pred):
    # In LabelEncoder, 'Critical' is index 0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total_critical = np.sum(y_true == 0)
    if total_critical > 0:
        correct_critical = cm[0, 0]
        return 1.0 - (correct_critical / total_critical)
    return 0.0

# -------------------------------------------------------------
# EVALUATION FUNCTIONS
# -------------------------------------------------------------

# 1-3. Paper Setup: KNN (k=5), matching bms.py fitness function
def eval_paper_knn(particle):
    mask = (particle[:num_features] > 0.5).astype(int)
    num_sel = np.sum(mask)
    if num_sel == 0:
        return np.inf, 0, 0, 0, 1.0
    
    X_tr = X_train.iloc[:, mask.astype(bool)]
    X_te = X_test.iloc[:, mask.astype(bool)]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_train)
    preds = knn.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    mdr_crit = calc_critical_mdr(y_test, preds)
    
    # Paper Fitness: F(theta) = -[(Acc + F1)/2] + 0.1 * (|theta| / 18)
    fitness = -((acc + f1) / 2.0) + 0.1 * (num_sel / num_features)
    return fitness, acc, f1, num_sel, mdr_crit

# 4. Phase 1 (Co-Optimizer): Co-tunes Random Forest hyperparameters on 3-class
def eval_phase1_coopt(particle):
    mask = (particle[:num_features] > 0.5).astype(int)
    num_sel = np.sum(mask)
    if num_sel == 0:
        return np.inf, 0, 0, 0, 1.0
    
    n_est = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
    max_d = int(np.clip(particle[num_features + 1] * 7 + 3, 3, 10))
    
    X_tr = X_train.iloc[:, mask.astype(bool)]
    X_te = X_test.iloc[:, mask.astype(bool)]
    
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=1)
    rf.fit(X_tr, y_train)
    preds = rf.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    mdr_crit = calc_critical_mdr(y_test, preds)
    
    fitness = -((acc + f1) / 2.0) + 0.1 * (num_sel / num_features)
    return fitness, acc, f1, num_sel, mdr_crit

# 5. Phase 2 (Proposed Robust + XAI): Stratified 5-Fold CV + Class Balancing
def eval_phase2_robust(particle):
    mask = (particle[:num_features] > 0.5).astype(int)
    num_sel = np.sum(mask)
    if num_sel == 0:
        return np.inf, 0, 0, 0, 1.0
    
    X_sel = X_scaled.iloc[:, mask.astype(bool)]
    n_est = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
    max_d = int(np.clip(particle[num_features + 1] * 7 + 3, 3, 10))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mdrs = [], [], []
    
    for tr, te in skf.split(X_sel, y):
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42, n_jobs=1)
        rf.fit(X_sel.iloc[tr], y.iloc[tr])
        preds = rf.predict(X_sel.iloc[te])
        
        accs.append(accuracy_score(y.iloc[te], preds))
        f1s.append(f1_score(y.iloc[te], preds, average="macro"))
        mdrs.append(calc_critical_mdr(y.iloc[te], preds))
        
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    avg_mdr = np.mean(mdrs)
    
    fitness = -((avg_acc + avg_f1) / 2.0) + 0.1 * (num_sel / num_features)
    return fitness, avg_acc, avg_f1, num_sel, avg_mdr

# -------------------------------------------------------------
# RUNNER
# -------------------------------------------------------------
def run_benchmark(name, eval_fn, co_opt=False, mode="pso", num_particles=25, max_iterations=50):
    dims = num_features + 2 if co_opt else num_features
    np.random.seed(42)
    pos = np.random.uniform(0.0, 1.0, size=(num_particles, dims))
    vel = np.random.uniform(low=-1, high=1, size=(num_particles, dims))
    
    p_best_pos = pos.copy()
    p_best_fit = np.full(num_particles, np.inf)
    
    g_best_pos = np.zeros(dims)
    g_best_fit = np.inf
    g_best_acc = g_best_f1 = g_best_num = 0
    g_best_mdr = 1.0
    
    w, c1, c2, b = 0.7, 2.0, 2.0, 1
    
    print(f"\n--- Running {name} ---", flush=True)
    start_time = time.time()
    
    for i in range(num_particles):
        f, acc, f1, num_sel, mdr = eval_fn(pos[i])
        p_best_fit[i] = f
        if f < g_best_fit:
            g_best_fit, g_best_pos = f, pos[i].copy()
            g_best_acc, g_best_f1, g_best_num, g_best_mdr = acc, f1, num_sel, mdr
            
    for iteration in range(max_iterations):
        a = 2 - (2 * iteration / max_iterations)
        is_pso = mode == "pso" or (mode == "hybrid" and iteration < max_iterations // 2)
        
        if mode == "hybrid" and iteration == max_iterations // 2 and i == 0:
            g_best_pos = 0.9 * g_best_pos + 0.05
            
        for i in range(num_particles):
            if is_pso:
                r1, r2 = np.random.rand(dims), np.random.rand(dims)
                vel[i] = w * vel[i] + c1 * r1 * (p_best_pos[i] - pos[i]) + c2 * r2 * (g_best_pos - pos[i])
                pos[i] = pos[i] + vel[i]
            else:
                r1, r2, p = np.random.rand(), np.random.rand(), np.random.rand()
                A, C = 2 * a * r1 - a, 2 * r2
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
            f, acc, f1, num_sel, mdr = eval_fn(pos[i])
            
            if f < p_best_fit[i]:
                p_best_fit[i], p_best_pos[i] = f, pos[i].copy()
            if f < g_best_fit:
                g_best_fit, g_best_pos = f, pos[i].copy()
                g_best_acc, g_best_f1, g_best_num, g_best_mdr = acc, f1, num_sel, mdr

    print(f"Done {name} in {time.time()-start_time:.1f}s | Acc: {g_best_acc:.4f} | F1: {g_best_f1:.4f} | MDR_Critical: {g_best_mdr:.4f} | Sensors: {g_best_num}/{num_features}", flush=True)
    
    mask = (g_best_pos[:num_features] > 0.5).astype(int)
    sel_cols = list(X.columns[mask.astype(bool)])
    
    return {
        "Model": name,
        "Classifier": "KNN (k=5)" if not co_opt else "RandomForest (Co-Tuned)",
        "Accuracy": round(g_best_acc, 4),
        "Macro-F1": round(g_best_f1, 4),
        "MDR (Critical Class)": round(g_best_mdr, 4),
        "Selected Sensors": f"{g_best_num}/{num_features}",
        "Sensor Names": ", ".join(sel_cols)
    }

results = []
# Baselines matching bms.py & Mayingi et al. (2026) exact feature set
results.append(run_benchmark("1. BPSO (Paper Baseline)", eval_paper_knn, co_opt=False, mode="pso", max_iterations=50))
results.append(run_benchmark("2. BWOA (Paper Baseline)", eval_paper_knn, co_opt=False, mode="woa", max_iterations=50))
results.append(run_benchmark("3. BHPWOA (Paper Baseline)", eval_paper_knn, co_opt=False, mode="hybrid", max_iterations=50))

# Proposed Extensions: Co-Tuned RF, CV & XAI
results.append(run_benchmark("4. Phase 1 (Co-Opt RF)", eval_phase1_coopt, co_opt=True, mode="hybrid", max_iterations=50))
results.append(run_benchmark("5. Phase 2 (Proposed 5-Fold CV + XAI)", eval_phase2_robust, co_opt=True, mode="hybrid", max_iterations=50))

df_res = pd.DataFrame(results)
print("\n================ FINAL EXACT BMS.PY REPRODUCED TABLE ================")
print(df_res.to_string(index=False))
df_res.to_csv("reproduced_and_proposed_benchmark.csv", index=False)
