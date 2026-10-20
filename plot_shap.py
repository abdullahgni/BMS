import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

def run_shap_analysis(data_path="EV_Battery_Charging_TR_Dataset_with_Notes.csv"):
    X_scaled, y, feature_names = load_and_preprocess(data_path)

    # Fast demonstration model on physical features
    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        class_weight={0: 2.5, 1: 1.0, 2: 1.0},
        random_state=42
    )
    clf.fit(X_scaled, y)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_scaled)
    shap_critical = shap_values[:, :, 0]

    plt.figure(figsize=(9, 4.5), dpi=120)
    shap.plots.bar(shap_critical, show=False)
    plt.title("SHAP Global Feature Importance (Critical Class)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4.5), dpi=120)
    shap.plots.beeswarm(shap_critical, show=False)
    plt.title("SHAP Beeswarm Distribution for Critical Thermal Runaway", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="EV_Battery_Charging_TR_Dataset_with_Notes.csv")
    args = parser.parse_args()
    run_shap_analysis(args.data)
