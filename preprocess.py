import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(csv_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv"):
    df = pd.read_csv(csv_path)

    if "EventFlag" in df.columns:
        le = LabelEncoder()
        if "ChargingStage" in df.columns:
            df["ChargingStage"] = le.fit_transform(df["ChargingStage"])
        if "MoistureDetected" in df.columns:
            df["MoistureDetected"] = df["MoistureDetected"].astype(int)

        target_map = {"Runaway": 0, "Watch": 1, "Alarm": 2, "Warning": 2}
        y = df["EventFlag"].map(target_map).values
        drop_cols = ["Timestamp", "ChargerID", "CellID", "Notes", "BMS_Status", "EventFlag", "TR_Probability"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    elif "Target_Label" in df.columns:
        y = df["Target_Label"].values
        drop_cols = ["battery_id", "cycle_file", "Capacity_Ah", "Health_State", "Target_Label"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    else:
        raise ValueError("Unsupported dataset schema. Expected EventFlag or Target_Label.")

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    return X_scaled, y, feature_names

def calc_critical_mdr(y_true, y_pred, crit_class=0):
    true_crit = (y_true == crit_class)
    pred_non_crit = (y_pred != crit_class)
    missed = np.sum(true_crit & pred_non_crit)
    total_crit = np.sum(true_crit)
    return float(missed / total_crit) if total_crit > 0 else 0.0

if __name__ == "__main__":
    X, y, feats = load_and_preprocess()
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features: {feats}")
