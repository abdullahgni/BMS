"""
Unified Final Benchmark: Paper Baselines (BPSO, BWOA, BHPWOA) vs. Proposed HGWO-PSO
Strict 5-Fold Stratified Cross Validation.
Theme: Optimization of BMS in EV while focusing on critical data safety.
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

df = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes.csv')
df = df.drop(columns=["Timestamp", "ChargerID", "CellID", "Notes"], errors='ignore')

le = LabelEncoder()
for col in ["ChargingStage", "BMS_Status", "EventFlag"]:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
if 'MoistureDetected' in df.columns:
    df['MoistureDetected'] = df['MoistureDetected'].astype(int)

# 0: Critical, 1: OK, 2: Warning
y = df["BMS_Status"].values
X = df.drop(columns=["BMS_Status"])
feature_names = X.columns.tolist()
num_features = len(feature_names)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def calc_critical_mdr(y_true, y_pred):
    true_crit = (y_true == 0)
    pred_non_crit = (y_pred != 0)
    missed = np.sum(true_crit & pred_non_crit)
    total_crit = np.sum(true_crit)
    return missed / total_crit if total_crit > 0 else 0.0

def evaluate_model_cv(X_data, y_data, clf_builder):
    accs, f1s, mdrs = [], [], []
    for tr, te in skf.split(X_data, y_data):
        clf = clf_builder()
        clf.fit(X_data.iloc[tr], y_data[tr])
        preds = clf.predict(X_data.iloc[te])
        accs.append(accuracy_score(y_data[te], preds))
        f1s.append(f1_score(y_data[te], preds, average="macro"))
        mdrs.append(calc_critical_mdr(y_data[te], preds))
    return np.mean(accs), np.mean(f1s), np.mean(mdrs)

# 1. Baseline 1: BPSO (Paper Baseline, KNN k=5)
bpso_sensors = ["ChargingStage", "DemandCurrent_A", "AvgTemp_C", "AmbientTemp_C", "StateOfHealth_%", "VibrationLevel_mg", "MoistureDetected", "ChargePower_kW", "TR_Probability"]
acc_bpso, f1_bpso, mdr_bpso = evaluate_model_cv(X_scaled[bpso_sensors], y, lambda: KNeighborsClassifier(n_neighbors=5))

# 2. Baseline 2: BWOA (Paper Baseline, KNN k=5)
bwoa_sensors = ["DemandCurrent_A", "StateOfHealth_%", "VibrationLevel_mg", "MoistureDetected"]
acc_bwoa, f1_bwoa, mdr_bwoa = evaluate_model_cv(X_scaled[bwoa_sensors], y, lambda: KNeighborsClassifier(n_neighbors=5))

# 3. Baseline 3: BHPWOA (Paper Baseline, KNN k=5)
bhpwoa_sensors = ["ChargingStage", "DemandCurrent_A", "AvgTemp_C", "AmbientTemp_C", "StateOfHealth_%", "VibrationLevel_mg", "MoistureDetected", "ChargePower_kW", "TR_Probability"]
acc_bhpwoa, f1_bhpwoa, mdr_bhpwoa = evaluate_model_cv(X_scaled[bhpwoa_sensors], y, lambda: KNeighborsClassifier(n_neighbors=5))

# 4. Phase 1: Co-Opt RF
p1_sensors = ["ChargingStage", "ChargeCurrent_A", "SOC_%", "StateOfHealth_%", "MoistureDetected", "Pressure_kPa"]
acc_p1, f1_p1, mdr_p1 = evaluate_model_cv(X_scaled[p1_sensors], y, lambda: RandomForestClassifier(n_estimators=100, max_depth=8, class_weight={0: 3.0, 1: 1.0, 2: 1.0}, random_state=42))

# 5. Proposed HGWO-PSO (Safety Co-Tuned)
hgwo_sensors = ["CellVoltage_V", "ChargeCurrent_A", "DemandCurrent_A", "MinTemp_C", "AvgTemp_C", "TR_Probability"]
acc_hgwo, f1_hgwo, mdr_hgwo = evaluate_model_cv(X_scaled[hgwo_sensors], y, lambda: RandomForestClassifier(n_estimators=40, max_depth=4, class_weight={0: 6.5, 1: 1.0, 2: 1.2}, random_state=42))

# 6. Proposed HGWO-PSO (Balanced Safety Operating Point)
acc_hgwo_bal, f1_hgwo_bal, mdr_hgwo_bal = evaluate_model_cv(X_scaled[hgwo_sensors], y, lambda: RandomForestClassifier(n_estimators=60, max_depth=5, class_weight={0: 4.5, 1: 1.0, 2: 1.2}, random_state=42))

results = [
    {
        "Phase / Method": "Baseline 1 (BPSO - Paper)",
        "Classifier": "KNN (k=5)",
        "Accuracy": f"{acc_bpso*100:.1f}%",
        "Macro-F1": f"{f1_bpso:.4f}",
        "Critical MDR": f"{mdr_bpso*100:.2f}%",
        "Sensors": f"{len(bpso_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(bpso_sensors)
    },
    {
        "Phase / Method": "Baseline 2 (BWOA - Paper)",
        "Classifier": "KNN (k=5)",
        "Accuracy": f"{acc_bwoa*100:.1f}%",
        "Macro-F1": f"{f1_bwoa:.4f}",
        "Critical MDR": f"{mdr_bwoa*100:.2f}%",
        "Sensors": f"{len(bwoa_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(bwoa_sensors)
    },
    {
        "Phase / Method": "Baseline 3 (BHPWOA - Paper)",
        "Classifier": "KNN (k=5)",
        "Accuracy": f"{acc_bhpwoa*100:.1f}%",
        "Macro-F1": f"{f1_bhpwoa:.4f}",
        "Critical MDR": f"{mdr_bhpwoa*100:.2f}%",
        "Sensors": f"{len(bhpwoa_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(bhpwoa_sensors)
    },
    {
        "Phase / Method": "Phase 1 (Co-Opt RF)",
        "Classifier": "RandomForest (Co-Tuned)",
        "Accuracy": f"{acc_p1*100:.1f}%",
        "Macro-F1": f"{f1_p1:.4f}",
        "Critical MDR": f"{mdr_p1*100:.2f}%",
        "Sensors": f"{len(p1_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(p1_sensors)
    },
    {
        "Phase / Method": "Proposed HGWO-PSO (Safety-Prioritized)",
        "Classifier": "RF (HGWO-PSO Co-Tuned)",
        "Accuracy": f"{acc_hgwo*100:.1f}%",
        "Macro-F1": f"{f1_hgwo:.4f}",
        "Critical MDR": f"{mdr_hgwo*100:.2f}%",
        "Sensors": f"{len(hgwo_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(hgwo_sensors)
    },
    {
        "Phase / Method": "Proposed HGWO-PSO (Balanced-Safety)",
        "Classifier": "RF (HGWO-PSO Co-Tuned)",
        "Accuracy": f"{acc_hgwo_bal*100:.1f}%",
        "Macro-F1": f"{f1_hgwo_bal:.4f}",
        "Critical MDR": f"{mdr_hgwo_bal*100:.2f}%",
        "Sensors": f"{len(hgwo_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(hgwo_sensors)
    }
]

df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("FINAL REPRODUCIBLE BENCHMARK TABLE (STRICT 5-FOLD CV)")
print("="*80)
print(df_res.to_string(index=False))
df_res.to_csv("final_hgwo_bms_comparison.csv", index=False)
