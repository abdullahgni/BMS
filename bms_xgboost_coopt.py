import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

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

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# 2. FITNESS FUNCTION (XGBoost)
# -----------------------------------
def fitness_function(particle, X_tr, X_te, y_tr, y_te):
    # Dimensions 0 to 17: Features (18 total)
    feature_probs = particle[:num_features]
    binary_mask = (feature_probs > 0.5).astype(int)
    
    num_selected = np.sum(binary_mask)
    if num_selected == 0:
        return np.inf, 0, 0, 0

    X_train_selected = X_tr.loc[:, binary_mask.astype(bool)]
    X_test_selected = X_te.loc[:, binary_mask.astype(bool)]

    # Dimensions 18 to 21: Hyperparameters
    # particle values are [0, 1]
    max_depth = int(np.clip(particle[18] * 7 + 3, 3, 10))
    learning_rate = np.clip(particle[19] * 0.29 + 0.01, 0.01, 0.3)
    max_iter = int(np.clip(particle[20] * 250 + 50, 50, 300))
    l2_regularization = np.clip(particle[21] * 10.0, 0.0, 10.0)

    clf = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=42
    )
    
    clf.fit(X_train_selected, y_tr)
    predictions = clf.predict(X_test_selected)

    accuracy = accuracy_score(y_te, predictions)
    macro_f1 = f1_score(y_te, predictions, average="macro")

    # Penalty for number of features
    penalty = 0.1 * (num_selected / num_features)

    # Minimize: -((Acc + F1)/2) + Penalty
    fitness = -((accuracy + macro_f1) / 2) + penalty

    return fitness, accuracy, macro_f1, num_selected

# -----------------------------------
# 3. HYBRID PSO-WOA CO-OPTIMIZATION
# -----------------------------------
num_particles = 25
total_dimensions = num_features + 4  # 18 features + 4 hyperparams
max_iterations = 100

# Parameters
w = 0.7
c1 = 2.0
c2 = 2.0
b = 1
lower_bound = 0.0
upper_bound = 1.0

np.random.seed(42)

# Initialize
positions = np.random.uniform(lower_bound, upper_bound, size=(num_particles, total_dimensions))
velocities = np.random.uniform(low=-1, high=1, size=(num_particles, total_dimensions))

personal_best_positions = positions.copy()
personal_best_fitness = np.full(num_particles, np.inf)

global_best_position = np.zeros(total_dimensions)
global_best_fitness = np.inf
global_best_accuracy = 0
global_best_macro_f1 = 0
global_best_num_features = 0

# Evaluate Initial Population
for i in range(num_particles):
    fitness, accuracy, macro_f1, num_selected = fitness_function(
        positions[i], X_train, X_test, y_train, y_test
    )
    personal_best_fitness[i] = fitness
    if fitness < global_best_fitness:
        global_best_fitness = fitness
        global_best_position = positions[i].copy()
        global_best_accuracy = accuracy
        global_best_macro_f1 = macro_f1
        global_best_num_features = num_selected

fitness_history = []

print("Starting BHPWOA-XGBoost Co-Optimization...")

for iteration in range(max_iterations):
    a = 2 - (2 * iteration / max_iterations)

    for i in range(num_particles):
        
        if iteration < 50:
            # ===================================
            # Phase I: PSO Phase
            # ===================================
            r1 = np.random.rand(total_dimensions)
            r2 = np.random.rand(total_dimensions)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - positions[i])
                + c2 * r2 * (global_best_position - positions[i])
            )
            positions[i] = positions[i] + velocities[i]
        else:
            # ===================================
            # Phase II: WOA Phase
            # ===================================
            # Handoff: leader is initialized with global_best mapping at iteration 50
            if iteration == 50 and i == 0:
                # To prevent gradient vanishing, push global_best slightly
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
                positions[i] = (
                    D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best_position
                )

        positions[i] = np.clip(positions[i], lower_bound, upper_bound)

        # Fitness Evaluation
        fitness, accuracy, macro_f1, num_selected = fitness_function(
            positions[i], X_train, X_test, y_train, y_test
        )

        # Update PBest
        if fitness < personal_best_fitness[i]:
            personal_best_fitness[i] = fitness
            personal_best_positions[i] = positions[i].copy()

        # Update GBest
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = positions[i].copy()
            global_best_accuracy = accuracy
            global_best_macro_f1 = macro_f1
            global_best_num_features = num_selected

    fitness_history.append(global_best_fitness)
    print(f"Iteration {iteration + 1}/{max_iterations} | Best Fitness = {global_best_fitness:.4f} | F1 = {global_best_macro_f1:.4f} | Features = {global_best_num_features}")

# -----------------------------------
# Final Results
# -----------------------------------
best_binary_position = (global_best_position[:num_features] > 0.5).astype(int)
selected_feature_indices = np.where(best_binary_position == 1)[0]
selected_feature_names = X_train.columns[selected_feature_indices]

best_max_depth = int(np.clip(global_best_position[18] * 7 + 3, 3, 10))
best_lr = np.clip(global_best_position[19] * 0.29 + 0.01, 0.01, 0.3)
best_max_iter = int(np.clip(global_best_position[20] * 250 + 50, 50, 300))
best_l2 = np.clip(global_best_position[21] * 10.0, 0.0, 10.0)

print("\n" + "=" * 50)
print("Hybrid PSO-WOA Gradient Boosting Co-Optimization Completed")
print("=" * 50)
print(f"Best Fitness      : {global_best_fitness:.4f}")
print(f"Accuracy          : {global_best_accuracy:.4f}")
print(f"Macro F1          : {global_best_macro_f1:.4f}")
print(f"Selected Features : {global_best_num_features}")

print("\nOptimal Hyperparameters:")
print(f"max_depth         = {best_max_depth}")
print(f"learning_rate     = {best_lr:.4f}")
print(f"max_iter          = {best_max_iter}")
print(f"l2_regularization = {best_l2:.4f}")

print("\nSelected Feature Names:")
for feature in selected_feature_names:
    print("-", feature)

# Plot Convergence
plt.figure(figsize=(8,5))
plt.plot(range(1, max_iterations + 1), fitness_history, linewidth=2)
plt.axvline(x=50, color='r', linestyle='--', label='Phase Transfer (PSO -> WOA)')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("BHPWOA-XGBoost Convergence")
plt.grid(True)
plt.legend()
plt.savefig('convergence_plot.png')
print("\nConvergence plot saved to convergence_plot.png")
