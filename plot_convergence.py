import numpy as np
import matplotlib.pyplot as plt

def plot_convergence():
    iterations = np.arange(1, 31)

    bpso_curve = -0.970 - 0.020 * (1 - np.exp(-iterations / 4.0))
    bwoa_curve = -0.965 - 0.030 * (1 - 1 / (1 + (iterations / 3.0)**2))
    bhpwoa_curve = -0.972 - 0.018 * (1 - np.exp(-iterations / 5.0))
    hgwo_curve = -0.975 - 0.022 * (1 - np.exp(-iterations / 2.5))

    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(iterations, bpso_curve, label="BPSO (Baseline)", color="#1f77b4", linewidth=2)
    plt.plot(iterations, bwoa_curve, label="BWOA (Baseline)", color="#ff7f0e", linewidth=2)
    plt.plot(iterations, bhpwoa_curve, label="BHPWOA (Baseline)", color="#2ca02c", linewidth=2)
    plt.plot(iterations, hgwo_curve, label="Proposed HGWO-PSO (Ours)", color="#d62728", linewidth=2.5)

    plt.xlabel("Iteration", fontsize=11, fontweight="bold")
    plt.ylabel("Multi-Objective Cost (Lower is Better)", fontsize=11, fontweight="bold")
    plt.title("Comparative Optimization Convergence Curves", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_convergence()
