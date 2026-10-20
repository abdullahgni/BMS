import argparse
import pandas as pd
from bpso import run_bpso
from bwoa import run_bwoa
from bhpwoa import run_bhpwoa
from hgwo_pso import run_hgwo_pso

def generate_comparison_table(dataset="ev", iter_count=5):
    print(f"\nEvaluating complete benchmark suite on dataset: {dataset.upper()}...\n")
    m1 = run_bpso(dataset, n_particles=15, max_iter=iter_count)
    m2 = run_bwoa(dataset, n_whales=15, max_iter=iter_count)
    m3 = run_bhpwoa(dataset, n_agents=15, max_iter=iter_count)
    m4 = run_hgwo_pso(dataset, n_wolves=15, max_iter=iter_count)

    def get_feats(m):
        return m.get("selected_features", m.get("selected", []))

    f1 = get_feats(m1)
    f2 = get_feats(m2)
    f3 = get_feats(m3)
    f4 = get_feats(m4)

    results = [
        {
            "Method": "Baseline 1: BPSO",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{m1['accuracy']*100:.2f}%",
            "Macro-F1": f"{m1['macro_f1']:.4f}",
            "Critical MDR": f"{m1['critical_mdr']*100:.2f}%",
            "Sensors": len(f1),
            "Selected Channels": ", ".join(f1)
        },
        {
            "Method": "Baseline 2: BWOA",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{m2['accuracy']*100:.2f}%",
            "Macro-F1": f"{m2['macro_f1']:.4f}",
            "Critical MDR": f"{m2['critical_mdr']*100:.2f}%",
            "Sensors": len(f2),
            "Selected Channels": ", ".join(f2)
        },
        {
            "Method": "Baseline 3: BHPWOA",
            "Classifier": "KNN (k=5)",
            "Accuracy": f"{m3['accuracy']*100:.2f}%",
            "Macro-F1": f"{m3['macro_f1']:.4f}",
            "Critical MDR": f"{m3['critical_mdr']*100:.2f}%",
            "Sensors": len(f3),
            "Selected Channels": ", ".join(f3)
        },
        {
            "Method": "Proposed HGWO-PSO",
            "Classifier": "Co-Tuned RF",
            "Accuracy": f"{m4['accuracy']*100:.2f}%",
            "Macro-F1": f"{m4['macro_f1']:.4f}",
            "Critical MDR": f"{m4['critical_mdr']*100:.2f}%",
            "Sensors": len(f4),
            "Selected Channels": ", ".join(f4)
        }
    ]

    df_comp = pd.DataFrame(results)
    print("\n" + "="*95)
    print(f"BENCHMARK COMPARISON TABLE ({dataset.upper()})")
    print("="*95)
    print(df_comp[["Method", "Classifier", "Accuracy", "Macro-F1", "Critical MDR", "Sensors"]].to_string(index=False))
    print("="*95 + "\n")
    return df_comp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ev", "nasa", "nasa_global"], default="ev")
    parser.add_argument("--iter", type=int, default=5)
    args = parser.parse_args()
    generate_comparison_table(args.dataset, iter_count=args.iter)
