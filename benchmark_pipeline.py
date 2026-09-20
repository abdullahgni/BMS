import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings
import time
warnings.filterwarnings('ignore')

# -----------------------------------
# 1. LOAD AND PREPARE DATASET
# -----------------------------------
df = pd.read_csv('sampled_50cols_dataset.csv')

# Merge fault flags into one 'System_Fault' target
flag_cols = [
    'overcurrent_flag', 'undercurrent_flag', 'over_voltage_flag', 
    'under_voltage_flag', 'over_temperature_flag', 'under_temperature_flag', 
    'short_circuit_flag'
]
df['System_Fault'] = df[flag_cols].max(axis=1)

# Drop unused columns and old flags
cols_to_drop = ['cell_id', 'timestamp', 'manufacturer', 'chemistry_type'] + flag_cols
df = df.drop(columns=cols_to_drop)

# Assuming all remaining columns except 'System_Fault' are numerical features
X = df.drop(columns=["System_Fault"])
y = df["System_Fault"]
num_features = X.shape[1]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Default split for single-eval models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Metaheuristic Settings
num_particles = 15
max_iterations = 30
w = 0.7; c1 = 2.0; c2 = 2.0; b = 1
lower_bound = 0.0; upper_bound = 1.0

# -----------------------------------
# 2. FITNESS FUNCTIONS
# -----------------------------------
def eval_standard(particle, co_optimize=False):
    feature_probs = particle[:num_features]
    binary_mask = (feature_probs > 0.5).astype(int)
    num_selected = np.sum(binary_mask)
    
    if num_selected == 0:
        return np.inf, 0, 0, 0, 1.0

    X_tr = X_train.loc[:, binary_mask.astype(bool)]
    X_te = X_test.loc[:, binary_mask.astype(bool)]

    if co_optimize:
        n_est = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
        max_d = int(np.clip(particle[num_features+1] * 7 + 3, 3, 10))
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    
    # MDR calculation (assuming fault class is 1)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    if np.sum(y_test == 1) > 0:
        mdr = 1 - (cm[1, 1] / np.sum(y_test == 1))
    else:
        mdr = 0.0

    penalty = 0.1 * (num_selected / num_features)
    fitness = -((acc + f1) / 2) + penalty
    return fitness, acc, f1, num_selected, mdr

def eval_robust(particle):
    feature_probs = particle[:num_features]
    binary_mask = (feature_probs > 0.5).astype(int)
    num_selected = np.sum(binary_mask)
    
    if num_selected == 0:
        return np.inf, 0, 0, 0, 1.0

    X_selected = X_scaled.loc[:, binary_mask.astype(bool)]
    n_est = int(np.clip(particle[num_features] * 100 + 20, 20, 120))
    max_d = int(np.clip(particle[num_features+1] * 7 + 3, 3, 10))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mdrs = [], [], []
    
    for tr_idx, te_idx in skf.split(X_selected, y):
        X_tr, X_te = X_selected.iloc[tr_idx], X_selected.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds, average="macro"))
        cm = confusion_matrix(y_te, preds, labels=[0, 1])
        if np.sum(y_te == 1) > 0:
            mdrs.append(1 - (cm[1, 1] / np.sum(y_te == 1)))
        else:
            mdrs.append(0.0)

    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    avg_mdr = np.mean(mdrs)
    
    penalty = 0.1 * (num_selected / num_features)
    fitness = -((avg_acc + avg_f1) / 2) + penalty
    return fitness, avg_acc, avg_f1, num_selected, avg_mdr

# -----------------------------------
# 3. ALGORITHMS RUNNER
# -----------------------------------
def run_metaheuristic(name, fitness_func, co_optimize=False, mode="pso"):
    dims = num_features + 2 if co_optimize else num_features
    np.random.seed(42)
    pos = np.random.uniform(lower_bound, upper_bound, size=(num_particles, dims))
    vel = np.random.uniform(low=-1, high=1, size=(num_particles, dims))
    
    p_best_pos = pos.copy()
    p_best_fit = np.full(num_particles, np.inf)
    
    g_best_pos = np.zeros(dims)
    g_best_fit, g_best_acc, g_best_f1, g_best_num, g_best_mdr = np.inf, 0, 0, 0, 1.0
    
    fit_history = []
    
    print(f"\n--- Running {name} ---")
    start_time = time.time()
    
    for i in range(num_particles):
        f, acc, f1, num, mdr = fitness_func(pos[i])
        p_best_fit[i] = f
        if f < g_best_fit:
            g_best_fit, g_best_pos, g_best_acc, g_best_f1, g_best_num, g_best_mdr = f, pos[i].copy(), acc, f1, num, mdr

    for iteration in range(max_iterations):
        a = 2 - (2 * iteration / max_iterations)
        
        for i in range(num_particles):
            is_pso = True
            if mode == "woa":
                is_pso = False
            elif mode == "hybrid" and iteration >= max_iterations // 2:
                is_pso = False
                if iteration == max_iterations // 2 and i == 0:
                    g_best_pos = 0.9 * g_best_pos + 0.05
                    
            if is_pso:
                r1, r2 = np.random.rand(dims), np.random.rand(dims)
                vel[i] = (w * vel[i] + c1 * r1 * (p_best_pos[i] - pos[i]) + c2 * r2 * (g_best_pos - pos[i]))
                pos[i] = pos[i] + vel[i]
            else:
                r1, r2 = np.random.rand(), np.random.rand()
                A, C, p = 2*a*r1 - a, 2*r2, np.random.rand()
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
                    pos[i] = (D * np.exp(b * l) * np.cos(2 * np.pi * l) + g_best_pos)
                    
            pos[i] = np.clip(pos[i], lower_bound, upper_bound)
            f, acc, f1, num, mdr = fitness_func(pos[i])
            
            if f < p_best_fit[i]:
                p_best_fit[i], p_best_pos[i] = f, pos[i].copy()
            if f < g_best_fit:
                g_best_fit, g_best_pos, g_best_acc, g_best_f1, g_best_num, g_best_mdr = f, pos[i].copy(), acc, f1, num, mdr
                
        fit_history.append(g_best_fit)
        
    print(f"Done in {time.time()-start_time:.1f}s | F1: {g_best_f1:.4f} | MDR: {g_best_mdr:.4f}")
    return {
        "Model": name,
        "Best Fitness": g_best_fit,
        "Accuracy": g_best_acc,
        "Macro-F1": g_best_f1,
        "MDR (Critical)": g_best_mdr,
        "Selected Features": g_best_num,
        "Best Position": g_best_pos,
        "History": fit_history
    }

# -----------------------------------
# 4. EXECUTE PIPELINE
# -----------------------------------
results = []
results.append(run_metaheuristic("1. BPSO", lambda p: eval_standard(p, False), False, "pso"))
results.append(run_metaheuristic("2. BWOA", lambda p: eval_standard(p, False), False, "woa"))
results.append(run_metaheuristic("3. BHPWOA", lambda p: eval_standard(p, False), False, "hybrid"))
results.append(run_metaheuristic("4. Phase 1 (Co-Opt)", lambda p: eval_standard(p, True), True, "hybrid"))
results.append(run_metaheuristic("5. Phase 2 (Robust+XAI)", eval_robust, True, "hybrid"))

# -----------------------------------
# 5. GENERATE TABLES & PLOTS
# -----------------------------------
# Save table
df_res = pd.DataFrame(results).drop(columns=["Best Position", "History"])
df_res.to_csv("benchmark_results.csv", index=False)
print("\n--- Final Results ---")
print(df_res.to_string())

# Plot Convergence
plt.figure(figsize=(10,6))
for r in results:
    plt.plot(range(1, max_iterations + 1), r["History"], label=r["Model"], linewidth=2)
plt.axvline(x=max_iterations//2, color='k', linestyle=':', label='Hybrid Phase Transfer')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (Lower is Better)")
plt.title("Comparative Convergence Analysis")
plt.legend()
plt.grid(True)
plt.savefig('combined_convergence.png', bbox_inches='tight')

# Phase 2 SHAP
phase2 = results[4]
best_pos = phase2["Best Position"]
binary_mask = (best_pos[:num_features] > 0.5).astype(int)
sel_idx = np.where(binary_mask == 1)[0]
sel_cols = X_scaled.columns[sel_idx]

X_final = X_scaled.loc[:, sel_cols]
n_est = int(np.clip(best_pos[num_features] * 100 + 20, 20, 120))
max_d = int(np.clip(best_pos[num_features+1] * 7 + 3, 3, 10))

clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X_final, y)

explainer = shap.TreeExplainer(clf)
# For a large dataset, SHAP takes forever. Sample 1000 rows for the SHAP plot.
X_shap = X_final.sample(n=min(1000, len(X_final)), random_state=42)
shap_values = explainer.shap_values(X_shap)

# For binary classification (RandomForest), shap_values is a list of [n_samples, n_features] for class 0 and class 1
try:
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        # Some versions return an array of shape (n_samples, n_features, n_classes)
        if len(shap_values.shape) == 3:
            shap_values_class1 = shap_values[:, :, 1]
        else:
            shap_values_class1 = shap_values

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values_class1, X_shap, show=False)
    plt.title("SHAP Summary Plot (Phase 2)")
    plt.savefig('shap_summary.png', bbox_inches='tight')

    # Dependence Plot
    vals = np.abs(shap_values_class1).mean(0)
    feat_imp = pd.DataFrame(list(zip(X_shap.columns, vals)), columns=['col','imp'])
    top_feat = feat_imp.sort_values(by='imp', ascending=False).iloc[0]['col']
    
    plt.figure(figsize=(8,6))
    shap.dependence_plot(top_feat, shap_values_class1, X_shap, show=False)
    plt.title(f"SHAP Dependence Plot: {top_feat}")
    plt.savefig('shap_dependence.png', bbox_inches='tight')
except Exception as e:
    print("SHAP Warning:", e)

print("\nPipeline Complete!")
