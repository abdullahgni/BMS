import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------
# 1. LOAD AND PREPARE DATASET
# -----------------------------------
df = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes.csv')

# Drop unused columns
df = df.drop(columns=["Timestamp", "ChargerID", "CellID", "Notes"])

label_encoder = LabelEncoder()
categorical_columns = ["ChargingStage", "BMS_Status", "EventFlag"]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
df["MoistureDetected"] = df["MoistureDetected"].astype(int)

X = df.drop(columns=["BMS_Status"])
y = df["BMS_Status"]
num_features = X.shape[1]  # Should be 18

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# -----------------------------------
# 2. FITNESS FUNCTION (5-Fold CV)
# -----------------------------------
def fitness_function(particle, X_data, y_data):
    # Dimensions 0 to 17: Features (18 total)
    feature_probs = particle[:num_features]
    binary_mask = (feature_probs > 0.5).astype(int)
    
    num_selected = np.sum(binary_mask)
    if num_selected == 0:
        return np.inf, 0, 0, 0

    X_selected = X_data.loc[:, binary_mask.astype(bool)]

    # Dimensions 18 to 21: Hyperparameters
    n_estimators = int(np.clip(particle[18] * 250 + 50, 50, 300))
    max_depth = int(np.clip(particle[19] * 7 + 3, 3, 10))
    min_samples_split = int(np.clip(particle[20] * 8 + 2, 2, 10))
    min_samples_leaf = int(np.clip(particle[21] * 4 + 1, 1, 5))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_macro_f1s = []
    
    for train_idx, test_idx in skf.split(X_selected, y_data):
        X_tr, X_te = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_tr, y_te = y_data.iloc[train_idx], y_data.iloc[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        
        fold_accuracies.append(accuracy_score(y_te, preds))
        fold_macro_f1s.append(f1_score(y_te, preds, average="macro"))

    avg_accuracy = np.mean(fold_accuracies)
    avg_macro_f1 = np.mean(fold_macro_f1s)

    # Penalty for number of features
    penalty = 0.1 * (num_selected / num_features)

    # Minimize: -((Acc + F1)/2) + Penalty
    fitness = -((avg_accuracy + avg_macro_f1) / 2) + penalty

    return fitness, avg_accuracy, avg_macro_f1, num_selected

# -----------------------------------
# 3. HYBRID PSO-WOA CO-OPTIMIZATION
# -----------------------------------
num_particles = 25
total_dimensions = num_features + 4  # 18 features + 4 hyperparams
max_iterations = 100

w = 0.7; c1 = 2.0; c2 = 2.0; b = 1
lower_bound = 0.0; upper_bound = 1.0

np.random.seed(42)

positions = np.random.uniform(lower_bound, upper_bound, size=(num_particles, total_dimensions))
velocities = np.random.uniform(low=-1, high=1, size=(num_particles, total_dimensions))

personal_best_positions = positions.copy()
personal_best_fitness = np.full(num_particles, np.inf)

global_best_position = np.zeros(total_dimensions)
global_best_fitness = np.inf
global_best_accuracy = 0
global_best_macro_f1 = 0
global_best_num_features = 0

print("Starting BHPWOA-RF Co-Optimization with 5-Fold CV...")

for i in range(num_particles):
    fitness, accuracy, macro_f1, num_selected = fitness_function(positions[i], X_scaled, y)
    personal_best_fitness[i] = fitness
    if fitness < global_best_fitness:
        global_best_fitness = fitness
        global_best_position = positions[i].copy()
        global_best_accuracy = accuracy
        global_best_macro_f1 = macro_f1
        global_best_num_features = num_selected

fitness_history = []

for iteration in range(max_iterations):
    a = 2 - (2 * iteration / max_iterations)

    for i in range(num_particles):
        if iteration < 50:
            # PSO Phase
            r1 = np.random.rand(total_dimensions)
            r2 = np.random.rand(total_dimensions)
            velocities[i] = (w * velocities[i] + c1 * r1 * (personal_best_positions[i] - positions[i]) + c2 * r2 * (global_best_position - positions[i]))
            positions[i] = positions[i] + velocities[i]
        else:
            # WOA Phase
            if iteration == 50 and i == 0:
                global_best_position = 0.9 * global_best_position + 0.05
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()

            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * global_best_position - positions[i])
                    positions[i] = global_best_position - A * D
                else:
                    random_index = np.random.randint(num_particles)
                    random_position = positions[random_index]
                    D = np.abs(C * random_position - positions[i])
                    positions[i] = random_position - A * D
            else:
                l = np.random.uniform(-1, 1)
                D = np.abs(global_best_position - positions[i])
                positions[i] = (D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best_position)

        positions[i] = np.clip(positions[i], lower_bound, upper_bound)

        fitness, accuracy, macro_f1, num_selected = fitness_function(positions[i], X_scaled, y)

        if fitness < personal_best_fitness[i]:
            personal_best_fitness[i] = fitness
            personal_best_positions[i] = positions[i].copy()

        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = positions[i].copy()
            global_best_accuracy = accuracy
            global_best_macro_f1 = macro_f1
            global_best_num_features = num_selected

    fitness_history.append(global_best_fitness)
    print(f"Iteration {iteration + 1}/{max_iterations} | Best Fitness = {global_best_fitness:.4f} | F1 = {global_best_macro_f1:.4f} | Features = {global_best_num_features}")

# -----------------------------------
# 4. FINAL RESULTS & SHAP ANALYSIS
# -----------------------------------
best_binary_position = (global_best_position[:num_features] > 0.5).astype(int)
selected_feature_indices = np.where(best_binary_position == 1)[0]
selected_feature_names = X_scaled.columns[selected_feature_indices]

best_n_estimators = int(np.clip(global_best_position[18] * 250 + 50, 50, 300))
best_max_depth = int(np.clip(global_best_position[19] * 7 + 3, 3, 10))
best_min_samples_split = int(np.clip(global_best_position[20] * 8 + 2, 2, 10))
best_min_samples_leaf = int(np.clip(global_best_position[21] * 4 + 1, 1, 5))

print("\n" + "=" * 50)
print("Phase 2: Robust Co-Optimization Completed")
print("=" * 50)
print(f"Best Fitness (5-Fold CV) : {global_best_fitness:.4f}")
print(f"Accuracy     (5-Fold CV) : {global_best_accuracy:.4f}")
print(f"Macro F1     (5-Fold CV) : {global_best_macro_f1:.4f}")
print(f"Selected Features        : {global_best_num_features}")

print("\nOptimal Hyperparameters:")
print(f"n_estimators      = {best_n_estimators}")
print(f"max_depth         = {best_max_depth}")
print(f"min_samples_split = {best_min_samples_split}")
print(f"min_samples_leaf  = {best_min_samples_leaf}")

print("\nSelected Feature Names:")
for feature in selected_feature_names:
    print("-", feature)

# Plot Convergence
plt.figure(figsize=(8,5))
plt.plot(range(1, max_iterations + 1), fitness_history, linewidth=2)
plt.axvline(x=50, color='r', linestyle='--', label='Phase Transfer (PSO -> WOA)')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (5-Fold CV)")
plt.title("BHPWOA-RF Convergence (Phase 2)")
plt.grid(True)
plt.legend()
plt.savefig('convergence_phase2.png')

# Train final model on FULL dataset for SHAP Analysis
X_final = X_scaled.loc[:, selected_feature_names]
final_model = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_final, y)

# Generate predictions for confusion matrix/MDR
final_preds = final_model.predict(X_final)
# MDR for Critical class (Assuming Critical is encoded as 2, Warn as 1, OK as 0)
# Based on original paper, Critical is class 2.
if 2 in y.unique():
    cm = confusion_matrix(y, final_preds)
    critical_total = np.sum(y == 2)
    critical_detected = cm[2, 2] # True positive for class 2
    mdr_critical = 1 - (critical_detected / critical_total)
    print(f"\nCritical Fault Missed Detection Rate (MDR): {mdr_critical:.4f}")

# SHAP Analysis
print("\nGenerating SHAP plots...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_final)

# SHAP Summary Plot
plt.figure(figsize=(10,6))
# For multi-class, shap_values is a list. We take class 2 (Critical) if it exists
class_index = 2 if len(shap_values) > 2 else 1
try:
    shap.summary_plot(shap_values[:,:,class_index], X_final, show=False)
except:
    shap.summary_plot(shap_values[class_index], X_final, show=False)
plt.title("SHAP Summary Plot for Critical Class")
plt.savefig('shap_summary.png', bbox_inches='tight')

# Find top feature for dependence plot
top_feature = selected_feature_names[0] # Fallback
try:
    vals = np.abs(shap_values[:,:,class_index]).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_final.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_feature = feature_importance.iloc[0]['col_name']
except:
    pass

plt.figure(figsize=(8,6))
try:
    shap.dependence_plot(top_feature, shap_values[:,:,class_index], X_final, show=False)
except:
    shap.dependence_plot(top_feature, shap_values[class_index], X_final, show=False)
plt.title(f"SHAP Dependence Plot: {top_feature}")
plt.savefig('shap_dependence.png', bbox_inches='tight')

print("SHAP plots saved successfully.")
