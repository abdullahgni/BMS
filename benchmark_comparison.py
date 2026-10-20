import argparse
import pandas as pd
from preprocess import load_and_preprocess
from bpso import evaluate_selected as eval_bpso
from bwoa import evaluate_selected as eval_bwoa
from bhpwoa import evaluate_selected as eval_bhpwoa
from hgwo_pso import evaluate_selected as eval_hgwo

def run_benchmark(data_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv"):
    X, y, feature_names = load_and_preprocess(data_path)
    print("=" * 95)
    print(f"EVALUATING UNIFIED BENCHMARK COMPARISON ON: {data_path}")
    print(f"Dataset Size: {X.shape[0]} samples, {X.shape[1]} physical channels (5-Fold Stratified CV)")
    print("=" * 95 + "\n")

    is_nasa = ("Temp_Rise_C" in feature_names or "Discharge_Duration_s" in feature_names)

    if is_nasa:
        bpso_feats = ['Temp_Rise_C', 'Final_Voltage_V', 'Avg_Current_A', 'Discharge_Duration_s']
        bwoa_feats = ['Max_Temp_C', 'Avg_Voltage_V', 'Energy_Delivered_Wh']
        bhpwoa_feats = ['Temp_Rise_C', 'Final_Voltage_V', 'Avg_Current_A', 'Discharge_Duration_s']
        hgwo_feats = ['Discharge_Duration_s', 'Energy_Delivered_Wh', 'Max_Load_Current_A']
        hgwo_hp = {"n_est": 100, "max_d": 6, "w_crit": 2.0}
    else:
        bpso_feats = [
            'ChargingStage', 'DemandCurrent_A', 'AvgTemp_C', 'AmbientTemp_C',
            'StateOfHealth_%', 'VibrationLevel_mg', 'MoistureDetected', 'ChargePower_kW'
        ]
        bwoa_feats = ['DemandCurrent_A', 'StateOfHealth_%', 'VibrationLevel_mg', 'MoistureDetected']
        bhpwoa_feats = bpso_feats
        hgwo_feats = ['DemandVoltage_V', 'SOC_%', 'MaxTemp_C', 'AvgTemp_C', 'MoistureDetected']
        hgwo_hp = {"n_est": 100, "max_d": 7, "w_crit": 2.0}

    print("[1/4] Evaluating Baseline 1: BPSO (KNN k=5)...")
    acc1, f11, mdr1, sel1 = eval_bpso(X, y, bpso_feats)

    print("[2/4] Evaluating Baseline 2: BWOA (KNN k=5)...")
    acc2, f12, mdr2, sel2 = eval_bwoa(X, y, bwoa_feats)

    print("[3/4] Evaluating Baseline 3: BHPWOA (KNN k=5)...")
    acc3, f13, mdr3, sel3 = eval_bhpwoa(X, y, bhpwoa_feats)

    print("[4/4] Evaluating Proposed HGWO-PSO (Co-Tuned RF)...")
    acc4, f14, mdr4, sel4 = eval_hgwo(X, y, hgwo_feats, n_est=hgwo_hp["n_est"], max_d=hgwo_hp["max_d"], w_crit=hgwo_hp["w_crit"])

    results = [
        {
            "Method": "Baseline 1 (BPSO - Paper)",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{acc1*100:.2f}%",
            "Macro-F1": f"{f11:.4f}",
            "Critical MDR": f"{mdr1*100:.2f}%",
            "Sensors": f"{len(sel1)} / {len(feature_names)}",
            "Selected Channels": ", ".join(sel1)
        },
        {
            "Method": "Baseline 2 (BWOA - Paper)",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{acc2*100:.2f}%",
            "Macro-F1": f"{f12:.4f}",
            "Critical MDR": f"{mdr2*100:.2f}%",
            "Sensors": f"{len(sel2)} / {len(feature_names)}",
            "Selected Channels": ", ".join(sel2)
        },
        {
            "Method": "Baseline 3 (BHPWOA - Paper)",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{acc3*100:.2f}%",
            "Macro-F1": f"{f13:.4f}",
            "Critical MDR": f"{mdr3*100:.2f}%",
            "Sensors": f"{len(sel3)} / {len(feature_names)}",
            "Selected Channels": ", ".join(sel3)
        },
        {
            "Method": "Proposed HGWO-PSO (Ours)",
            "Classifier": "RF (Co-Tuned)",
            "Accuracy": f"{acc4*100:.2f}%",
            "Macro-F1": f"{f14:.4f}",
            "Critical MDR": f"{mdr4*100:.2f}%",
            "Sensors": f"{len(sel4)} / {len(feature_names)}",
            "Selected Channels": ", ".join(sel4)
        }
    ]

    df_comp = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print(f"BENCHMARK COMPARISON TABLE ({data_path})")
    print("=" * 95)
    print(df_comp[["Method", "Classifier", "Accuracy", "Macro-F1", "Critical MDR", "Sensors"]].to_string(index=False))
    print("=" * 95 + "\n")
    return df_comp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="EV_Battery_Charging_TR_Dataset_with_Notes.csv")
    args = parser.parse_args()
    run_benchmark(args.data)
