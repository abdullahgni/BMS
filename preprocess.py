import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_ev(csv_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv"):
    df = pd.read_csv(csv_path)
    
    le = LabelEncoder()
    df["ChargingStage"] = le.fit_transform(df["ChargingStage"])
    df["MoistureDetected"] = df["MoistureDetected"].astype(int)
    
    target_map = {"Runaway": 0, "Watch": 1, "Alarm": 2, "Warning": 2}
    y = df["EventFlag"].map(target_map).values
    
    drop_cols = ["Timestamp", "ChargerID", "CellID", "Notes", "BMS_Status", "EventFlag", "TR_Probability"]
    X = df.drop(columns=drop_cols)
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    return X_scaled, y, feature_names

def load_and_preprocess_nasa(csv_path="nasa_battery_benchmark_classic4.csv"):
    df = pd.read_csv(csv_path)
    y = df["Target_Label"].values
    
    non_features = ["battery_id", "cycle_file", "Capacity_Ah", "Health_State", "Target_Label"]
    feature_names = [c for c in df.columns if c not in non_features]
    
    X = df[feature_names].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    return X_scaled, y, feature_names

def load_and_preprocess_nasa_global(csv_path="nasa_battery_benchmark_global_2750.csv"):
    return load_and_preprocess_nasa(csv_path)

def calc_critical_mdr(y_true, y_pred, crit_class=0):
    true_crit = (y_true == crit_class)
    pred_non_crit = (y_pred != crit_class)
    missed = np.sum(true_crit & pred_non_crit)
    total_crit = np.sum(true_crit)
    return float(missed / total_crit) if total_crit > 0 else 0.0

if __name__ == "__main__":
    X, y, feats = load_and_preprocess_ev()
    print(f"EV Data Processed: {X.shape[0]} samples, {len(feats)} features")
    print(f"Classes: Critical={np.sum(y==0)}, OK={np.sum(y==1)}, Warning={np.sum(y==2)}")
