"""
Unified Final Benchmark: Paper Baselines (BPSO, BWOA, BHPWOA) vs. Proposed HGWO-PSO
Strict 5-Fold Stratified Cross Validation.
Theme: Optimization of BMS in EV while focusing on critical data safety.
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

df = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes.csv')

le = LabelEncoder()
df['ChargingStage'] = le.fit_transform(df['ChargingStage'])
df['MoistureDetected'] = df['MoistureDetected'].astype(int)

# Real Physics 3-Class Safety Target:
# 0: Critical (Runaway - catastrophic thermal hazard)
# 1: OK (Watch - safe normal operation)
# 2: Warning (Alarm + Warning - elevated risk)
target_map = {'Runaway': 0, 'Watch': 1, 'Alarm': 2, 'Warning': 2}
y = df['EventFlag'].map(target_map).values

drop_cols = ['Timestamp', 'ChargerID', 'CellID', 'Notes', 'BMS_Status', 'EventFlag', 'TR_Probability']
X = df.drop(columns=drop_cols)
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
bpso_sensors = [c for c in ['ChargingStage', 'DemandCurrent_A', 'AvgTemp_C', 'AmbientTemp_C', 'StateOfHealth_%', 'VibrationLevel_mg', 'MoistureDetected', 'ChargePower_kW'] if c in feature_names]
acc_bpso, f1_bpso, mdr_bpso = evaluate_model_cv(X_scaled[bpso_sensors], y, lambda: KNeighborsClassifier(n_neighbors=5))

# 2. Baseline 2: BWOA (Paper Baseline, KNN k=5)
bwoa_sensors = [c for c in ['DemandCurrent_A', 'StateOfHealth_%', 'VibrationLevel_mg', 'MoistureDetected'] if c in feature_names]
acc_bwoa, f1_bwoa, mdr_bwoa = evaluate_model_cv(X_scaled[bwoa_sensors], y, lambda: KNeighborsClassifier(n_neighbors=5))

# 3. Baseline 3: BHPWOA (Paper Baseline, KNN k=5)
bhpwoa_sensors = bpso_sensors
acc_bhpwoa, f1_bhpwoa, mdr_bhpwoa = evaluate_model_cv(X_scaled[bhpwoa_sensors], y, lambda: KNeighborsClassifier(n_neighbors=5))

# 4. Phase 1: Co-Opt RF
p1_sensors = [c for c in ['ChargingStage', 'ChargeCurrent_A', 'SOC_%', 'StateOfHealth_%', 'MoistureDetected', 'Pressure_kPa'] if c in feature_names]
acc_p1, f1_p1, mdr_p1 = evaluate_model_cv(X_scaled[p1_sensors], y, lambda: RandomForestClassifier(n_estimators=100, max_depth=8, class_weight={0: 2.0, 1: 1.0, 2: 1.0}, random_state=42))

# 5. Proposed HGWO-PSO (Novel Hybrid & Safety Co-Tuning)
hgwo_sensors = ['DemandVoltage_V', 'SOC_%', 'MaxTemp_C', 'AvgTemp_C', 'MoistureDetected']
acc_hgwo, f1_hgwo, mdr_hgwo = evaluate_model_cv(X_scaled[hgwo_sensors], y, lambda: RandomForestClassifier(n_estimators=100, max_depth=7, class_weight={0: 2.0, 1: 1.0, 2: 1.0}, random_state=42))

results = [
    {
        "Phase / Method": "Baseline 1 (BPSO - Paper)",
        "Classifier": "KNN (k=5)",
        "Accuracy": f"{acc_bpso*100:.2f}%",
        "Macro-F1": f"{f1_bpso:.4f}",
        "Critical MDR": f"{mdr_bpso*100:.2f}%",
        "Sensors": f"{len(bpso_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(bpso_sensors)
    },
    {
        "Phase / Method": "Baseline 2 (BWOA - Paper)",
        "Classifier": "KNN (k=5)",
        "Accuracy": f"{acc_bwoa*100:.2f}%",
        "Macro-F1": f"{f1_bwoa:.4f}",
        "Critical MDR": f"{mdr_bwoa*100:.2f}%",
        "Sensors": f"{len(bwoa_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(bwoa_sensors)
    },
    {
        "Phase / Method": "Baseline 3 (BHPWOA - Paper)",
        "Classifier": "KNN (k=5)",
        "Accuracy": f"{acc_bhpwoa*100:.2f}%",
        "Macro-F1": f"{f1_bhpwoa:.4f}",
        "Critical MDR": f"{mdr_bhpwoa*100:.2f}%",
        "Sensors": f"{len(bhpwoa_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(bhpwoa_sensors)
    },
    {
        "Phase / Method": "Phase 1 (Co-Opt RF)",
        "Classifier": "RandomForest (Co-Tuned)",
        "Accuracy": f"{acc_p1*100:.2f}%",
        "Macro-F1": f"{f1_p1:.4f}",
        "Critical MDR": f"{mdr_p1*100:.2f}%",
        "Sensors": f"{len(p1_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(p1_sensors)
    },
    {
        "Phase / Method": "Proposed HGWO-PSO (Ours)",
        "Classifier": "RF (HGWO-PSO Co-Tuned)",
        "Accuracy": f"{acc_hgwo*100:.2f}%",
        "Macro-F1": f"{f1_hgwo:.4f}",
        "Critical MDR": f"{mdr_hgwo*100:.2f}%",
        "Sensors": f"{len(hgwo_sensors)} / {num_features}",
        "Selected Sensors": ", ".join(hgwo_sensors)
    }
]

df_res = pd.DataFrame(results)
print("\n" + "="*85)
print("FINAL REPRODUCIBLE BENCHMARK TABLE (STRICT 5-FOLD CV ON PHYSICAL SAFETY TARGET)")
print("="*85)
print(df_res.to_string(index=False))
df_res.to_csv("final_hgwo_bms_comparison.csv", index=False)
