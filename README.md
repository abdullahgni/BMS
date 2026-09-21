# Battery Management System (BMS) Feature Selection & Co-Optimization

This repository contains clean, modular Python scripts for Electric Vehicle (EV) Battery Management System (BMS) feature reduction and classifier co-optimization for 3-class fault detection (`OK`, `Warning`, `Critical`).

## 📁 Clean Repository Structure

| File | Description | Execution Role |
| :--- | :--- | :--- |
| **`binary_pso.py`** | Binary Particle Swarm Optimization (BPSO) | Baseline metaheuristic feature selection wrapper (KNN $k=5$) |
| **`binary_woa.py`** | Binary Whale Optimization Algorithm (BWOA) | Baseline whale optimization metaheuristic wrapper (KNN $k=5$) |
| **`hybrid_pso_woa.py`** | Binary Hybrid PSO-WOA (BHPWOA) | Original paper (Mayingi et al., 2026) hybrid feature selector |
| **`phase1_co_optimizer.py`** | Proposed Phase 1: Joint Co-Optimizer | Co-optimizes feature mask AND Random Forest hyperparameters (`n_estimators`, `max_depth`) |
| **`phase2_robust_xai.py`** | Proposed Phase 2: Robust CV + SHAP XAI | Stratified 5-Fold Cross-Validation, class balancing, MDR optimization, and SHAP Explainable AI |
| **`run_og_benchmark.py`** | Unified Benchmark Runner | Runs all 5 models sequentially on `EV_Battery_Charging_TR_Dataset_with_Notes.csv` and outputs `reproduced_and_proposed_benchmark.csv` |
| **`EV_Battery_Charging_TR_Dataset_with_Notes.csv`** | Primary 18-Feature Dataset | Real-world 500-sample EV charging telemetry dataset |
| **`presentation_review2.md`** | Review-2 Slide Deck | Academic presentation content for Review-2 grading rubrics |

---

## 💻 Google Colab Quick-Start Instructions

You can clone and run this entire repository directly inside Google Colab in **one click**:

```python
# Cell 1: Clone Repository & Install Dependencies
!git clone https://github.com/abdullahgni/BMS.git
%cd BMS
!pip install numpy pandas scikit-learn matplotlib shap

# Cell 2: Run Full 5-Model Benchmark Suite
!python run_og_benchmark.py

# Cell 3: Display Benchmark Results Table
import pandas as pd
df = pd.read_csv('reproduced_and_proposed_benchmark.csv')
display(df)
```

### Running Individual Models in Colab:
```python
# Run BPSO Baseline
!python binary_pso.py

# Run Original Paper Hybrid PSO-WOA
!python hybrid_pso_woa.py

# Run Phase 1 Co-Optimizer
!python phase1_co_optimizer.py

# Run Phase 2 Robust CV + SHAP XAI
!python phase2_robust_xai.py
```

