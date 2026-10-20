import numpy as np
import matplotlib.pyplot as plt

def plot_cross_dataset():
    methods = ["BPSO", "BWOA", "HGWO-PSO"]
    ev_acc = [64.8, 57.0, 94.0]
    nasa_acc = [97.48, 97.64, 98.74]

    ev_mdr = [59.14, 73.01, 7.18]
    nasa_mdr = [2.31, 1.54, 2.31]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    x = np.arange(len(methods))
    w = 0.35

    axes[0].bar(x - w/2, ev_acc, w, label="EV Thermal Runaway", color="#4361EE", edgecolor="black")
    axes[0].bar(x + w/2, nasa_acc, w, label="NASA Li-ion Classic 4", color="#4CC9F0", edgecolor="black")
    axes[0].set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Accuracy Comparison across Datasets", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=10, fontweight="bold")
    axes[0].set_ylim(40, 105)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(x - w/2, ev_mdr, w, label="EV Thermal Runaway", color="#E63946", edgecolor="black")
    axes[1].bar(x + w/2, nasa_mdr, w, label="NASA Li-ion Classic 4", color="#F77F00", edgecolor="black")
    axes[1].set_ylabel("Critical Missed Detection Rate (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Safety-Critical MDR Comparison (Lower is Safer)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 85)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    datasets = ["EV Fast-Charging", "NASA Classic 4", "NASA Global Archive"]
    orig_sensors = [17, 15, 15]
    opt_sensors = [5, 3, 3]
    x_d = np.arange(len(datasets))

    axes[2].bar(x_d - w/2, orig_sensors, w, label="Full Candidate Suite", color="#adb5bd", edgecolor="black")
    axes[2].bar(x_d + w/2, opt_sensors, w, label="HGWO-PSO Selected", color="#2b8a3e", edgecolor="black")
    axes[2].set_ylabel("Active Physical Sensors", fontsize=11, fontweight="bold")
    axes[2].set_title("Hardware Sensor Footprint Compression", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x_d)
    axes[2].set_xticklabels(datasets, fontsize=10, fontweight="bold")
    axes[2].legend()
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("cross_dataset_comparison.png")
    plt.show()
    print("Cross-dataset comparison plots saved successfully.")

if __name__ == "__main__":
    plot_cross_dataset()
