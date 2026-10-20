import argparse
import pandas as pd
from bpso import run_bpso
from bwoa import run_bwoa
from bhpwoa import run_bhpwoa
from hgwo_pso import run_hgwo_pso

def generate_comparison_table(dataset="ev"):
    print(f"\nEvaluating complete benchmark suite on dataset: {dataset.upper()}...\n")
    m1 = run_bpso(dataset)
    m2 = run_bwoa(dataset)
    m3 = run_bhpwoa(dataset)
    m4 = run_hgwo_pso(dataset)

    results = [
        {
            "Method": "Baseline 1: BPSO",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{m1['accuracy']*100:.2f}%",
            "Macro-F1": f"{m1['macro_f1']:.4f}",
            "Critical MDR": f"{m1['critical_mdr']*100:.2f}%",
            "Sensors": len(m1["selected"]),
            "Selected Channels": ", ".join(m1["selected"])
        },
        {
            "Method": "Baseline 2: BWOA",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{m2['accuracy']*100:.2f}%",
            "Macro-F1": f"{m2['macro_f1']:.4f}",
            "Critical MDR": f"{m2['critical_mdr']*100:.2f}%",
            "Sensors": len(m2["selected"]),
            "Selected Channels": ", ".join(m2["selected"])
        },
        {
            "Method": "Baseline 3: BHPWOA",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{m3['accuracy']*100:.2f}%",
            "Macro-F1": f"{m3['macro_f1']:.4f}",
            "Critical MDR": f"{m3['critical_mdr']*100:.2f}%",
            "Sensors": len(m3["selected"]),
            "Selected Channels": ", ".join(m3["selected"])
        },
        {
            "Method": "Proposed HGWO-PSO",
            "Classifier": "Co-Tuned RF",
            "Accuracy": f"{m4['accuracy']*100:.2f}%",
            "Macro-F1": f"{m4['macro_f1']:.4f}",
            "Critical MDR": f"{m4['critical_mdr']*100:.2f}%",
            "Sensors": len(m4["selected"]),
            "Selected Channels": ", ".join(m4["selected"])
        }
    ]

    df_comp = pd.DataFrame(results)
    print("\n" + "="*100)
    print(f"BENCHMARK COMPARISON TABLE ({dataset.upper()})")
    print("="*100)
    print(df_comp[["Method", "Classifier", "Accuracy", "Macro-F1", "Critical MDR", "Sensors"]].to_string(index=False))
    print("="*100 + "\n")
    return df_comp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ev", "nasa", "nasa_global"], default="ev")
    args = parser.parse_args()
    generate_comparison_table(args.dataset)
