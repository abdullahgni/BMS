import argparse
import pandas as pd
from preprocess import load_and_preprocess
from bpso import BPSO
from bwoa import BWOA
from bhpwoa import BHPWOA
from hgwo_pso import HGWO_PSO

def run_benchmark(data_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv", iter_count=5):
    X, y, feature_names = load_and_preprocess(data_path)
    print("=" * 80)
    print(f"EVALUATING COMPLETE BENCHMARK ON: {data_path}")
    print(f"Dataset Size: {X.shape[0]} samples, {X.shape[1]} physical channels")
    print("=" * 80 + "\n")

    print("[1/4] Running BPSO...")
    bpso = BPSO(n_particles=15, max_iter=iter_count)
    res_bpso = bpso.fit(X, y)

    print("\n[2/4] Running BWOA...")
    bwoa = BWOA(n_whales=15, max_iter=iter_count)
    res_bwoa = bwoa.fit(X, y)

    print("\n[3/4] Running BHPWOA...")
    bhpwoa = BHPWOA(n_agents=15, max_iter=iter_count)
    res_bhpwoa = bhpwoa.fit(X, y)

    print("\n[4/4] Running Proposed HGWO-PSO...")
    hgwo = HGWO_PSO(n_wolves=15, max_iter=iter_count)
    res_hgwo = hgwo.fit(X, y)

    results = [
        {
            "Method": "Baseline 1: BPSO",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{res_bpso['accuracy']*100:.2f}%",
            "Macro-F1": f"{res_bpso['macro_f1']:.4f}",
            "Critical MDR": f"{res_bpso['critical_mdr']*100:.2f}%",
            "Sensors": f"{len(res_bpso['selected_features'])} / {len(feature_names)}",
            "Selected Channels": ", ".join(res_bpso['selected_features'])
        },
        {
            "Method": "Baseline 2: BWOA",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{res_bwoa['accuracy']*100:.2f}%",
            "Macro-F1": f"{res_bwoa['macro_f1']:.4f}",
            "Critical MDR": f"{res_bwoa['critical_mdr']*100:.2f}%",
            "Sensors": f"{len(res_bwoa['selected_features'])} / {len(feature_names)}",
            "Selected Channels": ", ".join(res_bwoa['selected_features'])
        },
        {
            "Method": "Baseline 3: BHPWOA",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{res_bhpwoa['accuracy']*100:.2f}%",
            "Macro-F1": f"{res_bhpwoa['macro_f1']:.4f}",
            "Critical MDR": f"{res_bhpwoa['critical_mdr']*100:.2f}%",
            "Sensors": f"{len(res_bhpwoa['selected_features'])} / {len(feature_names)}",
            "Selected Channels": ", ".join(res_bhpwoa['selected_features'])
        },
        {
            "Method": "Proposed HGWO-PSO",
            "Classifier": "Co-Tuned RF",
            "Accuracy": f"{res_hgwo['accuracy']*100:.2f}%",
            "Macro-F1": f"{res_hgwo['macro_f1']:.4f}",
            "Critical MDR": f"{res_hgwo['critical_mdr']*100:.2f}%",
            "Sensors": f"{len(res_hgwo['selected_features'])} / {len(feature_names)}",
            "Selected Channels": ", ".join(res_hgwo['selected_features'])
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
    parser.add_argument("--iter", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(args.data, iter_count=args.iter)
