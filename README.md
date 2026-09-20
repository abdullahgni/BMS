# Battery Management System (BMS) Feature Selection & Co-Optimization

This repository contains five modular, standalone Python scripts implementing feature reduction and classifier co-optimization algorithms for Battery Management System (BMS) fault detection and State of Health (SOH) monitoring.

## 📁 Repository Structure

| File | Description | Key Features |
| :--- | :--- | :--- |
| **`binary_pso.py`** | Binary Particle Swarm Optimization (BPSO) | Standard metaheuristic wrapper baseline for feature selection |
| **`binary_woa.py`** | Binary Whale Optimization Algorithm (BWOA) | Standard whale optimization metaheuristic baseline |
| **`hybrid_pso_woa.py`** | Binary Hybrid PSO-WOA (BHPWOA) | Implementation of the hybrid algorithm from the baseline paper |
| **`phase1_co_optimizer.py`** | Proposed Phase 1: Joint Co-Optimizer | Co-optimizes feature subset AND Random Forest hyperparameters simultaneously |
| **`phase2_robust_xai.py`** | Proposed Phase 2: Robust CV + SHAP XAI | Stratified 5-Fold Cross-Validation, class balancing, Missed Detection Rate penalty, and SHAP Explainable AI |
| **`benchmark_pipeline.py`** | Unified Benchmark Suite | Runs all 5 algorithms sequentially and generates comparative plots & CSV results |

## 🚀 Quick Start

### 1. Requirements & Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib shap
```

### 2. Running Individual Algorithms
Run any algorithm independently:

```bash
# Run Binary PSO
python3 binary_pso.py

# Run Binary WOA
python3 binary_woa.py

# Run Original Paper Hybrid PSO-WOA
python3 hybrid_pso_woa.py

# Run Phase 1 Co-Optimizer
python3 phase1_co_optimizer.py

# Run Phase 2 Robust Co-Optimizer with SHAP Plots
python3 phase2_robust_xai.py
```

### 3. Running the Full Benchmark Pipeline
```bash
python3 benchmark_pipeline.py
```

This will run all 5 models sequentially, save metrics to `benchmark_results.csv`, generate the comparative `combined_convergence.png`, and save SHAP summary/dependence plots (`shap_summary.png`, `shap_dependence.png`).

## 📊 Summary of Proposed Novelty (Phase 1 & 2 vs Baselines)
- **Baseline BHPWOA:** Selects features with fixed, un-tuned classifier parameters.
- **Phase 1 Co-Optimizer:** Extends particle dimension to co-optimize `[features, n_estimators, max_depth]`.
- **Phase 2 Robust XAI:** Replaces single train-test splits with Stratified 5-Fold CV to prevent data leakage and memorization, incorporates `class_weight='balanced'`, penalizes Missed Detection Rate (MDR) for critical safety, and integrates SHAP TreeExplainer for model interpretability.
