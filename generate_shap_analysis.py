"""
Publication-Quality SHAP Explainable AI (XAI) Analysis
for the Proposed HGWO-PSO Optimal 5-Sensor BMS Model.

Analyzes the physical interpretability of the selected sensors:
- DemandVoltage_V
- SOC_%
- MaxTemp_C
- AvgTemp_C
- MoistureDetected
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless publication backend
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("Starting SHAP Explainability Analysis...", flush=True)

# 1. Load dataset with ground-truth physical safety mapping
df = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes.csv')
target_map = {'Runaway': 0, 'Watch': 1, 'Alarm': 2, 'Warning': 2}
y = df['EventFlag'].map(target_map).values
class_names = ['Critical (Runaway)', 'Normal (Watch)', 'Warning (Alarm/Warning)']

# 2. Optimal 5 Sensors selected by HGWO-PSO
selected_sensors = ['DemandVoltage_V', 'SOC_%', 'MaxTemp_C', 'AvgTemp_C', 'MoistureDetected']
df['MoistureDetected'] = df['MoistureDetected'].astype(int)
X = df[selected_sensors]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=selected_sensors)

# 3. Train HGWO-PSO Co-Tuned Model
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    class_weight={0: 2.0, 1: 1.0, 2: 1.0},
    random_state=42
)
clf.fit(X_scaled, y)
print("Trained Random Forest model on 5 optimal sensors successfully.", flush=True)

# 4. Compute SHAP Values using TreeExplainer
explainer = shap.TreeExplainer(clf)
shap_exp = explainer(X_scaled) # Shape: (500, 5, 3)

# Extract Class 0 (Critical Thermal Runaway) Explanation object
shap_crit = shap_exp[:, :, 0]

# 5. Figure 1: Feature Importance Bar Plot (Critical Class)
plt.figure(figsize=(9, 5), dpi=300)
shap.plots.bar(shap_crit, show=False)
plt.title("Figure 1: Mean |SHAP| Importance for Critical Thermal Runaway", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved shap_summary_bar.png", flush=True)

# 6. Figure 2: Beeswarm Plot specifically for Class 0: Critical (Thermal Runaway)
plt.figure(figsize=(9, 5), dpi=300)
shap.plots.beeswarm(shap_crit, show=False)
plt.title("Figure 2: SHAP Value Distribution for Critical Thermal Runaway (Class 0)", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("shap_beeswarm_critical.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved shap_beeswarm_critical.png", flush=True)

# 7. Figure 3: SHAP Dependence Plot (MaxTemp_C vs SOC_%)
plt.figure(figsize=(8, 5), dpi=300)
shap.plots.scatter(shap_crit[:, "MaxTemp_C"], color=shap_crit[:, "SOC_%"], show=False)
plt.title("Figure 3: Thermal Runaway Critical Impact: MaxTemp_C vs SOC_%", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("MaxTemp_C (Standardized)", fontsize=11)
plt.ylabel("SHAP Value (Impact on Critical Runaway)", fontsize=11)
plt.tight_layout()
plt.savefig("shap_dependence_temp_soc.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved shap_dependence_temp_soc.png", flush=True)

# 8. Copy to Artifact Directory for UI Rendering
artifact_dir = "/Users/abdullah/.gemini/antigravity-ide/brain/8911f183-f911-44ab-8ceb-076d231ad09f"
for fig in ["shap_summary_bar.png", "shap_beeswarm_critical.png", "shap_dependence_temp_soc.png"]:
    dest = os.path.join(artifact_dir, fig)
    shutil.copy(fig, dest)
    print(f"Copied {fig} to artifact directory.", flush=True)

# 9. Compute Numerical Mean |SHAP| values for Critical Class
mean_abs_shap = np.mean(np.abs(shap_crit.values), axis=0)
shap_df = pd.DataFrame({
    "Sensor": selected_sensors,
    "Mean_|SHAP|_Critical": mean_abs_shap
}).sort_values(by="Mean_|SHAP|_Critical", ascending=False)

print("\n" + "="*50)
print("MEAN ABSOLUTE SHAP IMPORTANCE FOR CRITICAL RUNAWAY:")
print("="*50)
print(shap_df.to_string(index=False))

shap_df.to_csv("shap_importance_critical.csv", index=False)
print("\nSHAP analysis complete!", flush=True)
