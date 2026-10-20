# Optimization of Battery Management Systems (BMS) in Electric Vehicles

An intelligent machine learning framework for safety-critical fault detection and physical sensor optimization in Electric Vehicle (EV) Battery Management Systems. 

This project implements a novel **Hybrid Grey Wolf Optimizer – Particle Swarm Optimization (HGWO-PSO)** algorithm to simultaneously identify the minimal set of physical battery sensors and co-tune classifier hyperparameters, prioritizing the reduction of the **Critical Missed Detection Rate (MDR)** for battery thermal runaway and aging hazards.

---

## Project Workflow

```text
Telemetry Data Ingestion  ──>  Data Analytics & Preprocessing  ──>  Feature Selection & Hyperparameter Co-Tuning  ──>  Explainable AI (SHAP)  ──>  Cross-Dataset Validation
(EV Charging & NASA Li-ion)     (Z-Score Scaling & 3-Class Target)   (BPSO, BWOA, BHPWOA vs. Proposed HGWO-PSO)         (Electrochemical Auditing)    (Generalization on 34 Cells)
```

1. **Data Ingestion & Analytics**: Loads multivariate battery telemetry streams and evaluates statistical distributions and operational class imbalances.
2. **Preprocessing**: Cleans non-informative metadata, applies Z-score normalization, and encodes safety states (`Critical`, `OK`, `Warning`).
3. **Comparative Optimization**:
   - Evaluates literature baselines: **BPSO**, **BWOA**, and **BHPWOA** with K-Nearest Neighbors ($k=5$).
   - Executes proposed **HGWO-PSO** with Random Forest hyperparameter co-tuning under a safety-prioritized multi-objective fitness function.
4. **Explainable AI (SHAP)**: Uses TreeExplainer to audit feature attributions, verifying that retained sensors align with electrochemical principles.
5. **Cross-Dataset Validation**: Benchmarks the identical pipeline on the NASA Li-ion prognostic aging archive to confirm transferability.

---

## Repository Structure

```text
├── EV_Battery_Charging_TR_Dataset_with_Notes.csv   # EV fast-charging telemetry dataset (1,000 samples)
├── nasa_battery_benchmark_classic4.csv              # NASA Li-ion classic 4-cell benchmark dataset (636 cycles)
├── nasa_battery_benchmark_global_2750.csv          # NASA global multi-temperature dataset (2,750 cycles)
│
├── preprocess.py                                    # Centralized telemetry cleaning & scaling routines
├── bpso.py                                          # Binary Particle Swarm Optimization
├── bwoa.py                                          # Binary Whale Optimization Algorithm
├── bhpwoa.py                                        # Binary Hybrid PSO-WOA
├── hgwo_pso.py                                      # Proposed HGWO-PSO co-optimization algorithm
│
├── plot_convergence.py                             # Generates multi-objective convergence curves
├── plot_shap.py                                     # Generates SHAP importance and beeswarm plots
├── plot_cross_dataset.py                            # Generates cross-dataset comparison bar charts
├── benchmark_comparison.py                          # Compiles summary performance tables
│
├── bms_presentation_demo.ipynb                      # Google Colab notebook for step-by-step presentation
├── requirements.txt                                 # Python dependencies
└── README.md                                        # Project overview and workflow documentation
```

---

## Quickstart (Google Colab / Local)

Open [`bms_presentation_demo.ipynb`](https://github.com/abdullahgni/BMS/blob/main/bms_presentation_demo.ipynb) directly in Google Colab, or run sequentially:

```bash
# 1. Clone repository
git clone https://github.com/abdullahgni/BMS.git
cd BMS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run individual algorithms
python bpso.py --dataset ev
python bwoa.py --dataset ev
python bhpwoa.py --dataset ev
python hgwo_pso.py --dataset ev

# 4. View visual analysis & summary comparison table
python plot_convergence.py
python plot_shap.py
python benchmark_comparison.py --dataset ev
```
