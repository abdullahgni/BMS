import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess_ev

def run_shap_analysis():
    X_scaled, y, _ = load_and_preprocess_ev()
    selected_sensors = ["DemandVoltage_V", "SOC_%", "MaxTemp_C", "AvgTemp_C", "MoistureDetected"]
    X_sub = X_scaled[selected_sensors]

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=7,
        class_weight={0: 2.0, 1: 1.0, 2: 1.0},
        random_state=42
    )
    clf.fit(X_sub, y)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_sub)
    shap_critical = shap_values[:, :, 0]

    plt.figure(figsize=(9, 4.5), dpi=150)
    shap.plots.bar(shap_critical, show=False)
    plt.title("SHAP Global Feature Importance (Critical Class)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png")
    plt.show()

    plt.figure(figsize=(9, 4.5), dpi=150)
    shap.plots.beeswarm(shap_critical, show=False)
    plt.title("SHAP Beeswarm Distribution for Critical Thermal Runaway", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_beeswarm_critical.png")
    plt.show()
    print("SHAP explainability plots generated successfully.")

if __name__ == "__main__":
    run_shap_analysis()
