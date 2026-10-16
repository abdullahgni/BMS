# Optimization of Battery Management Systems (BMS) in EVs with Safety-Prioritized Critical Fault Detection

This repository implements **HGWO-PSO (Hybrid Grey Wolf Optimizer + Particle Swarm Optimization)** for feature selection and hyperparameter co-tuning in Electric Vehicle (EV) Battery Management Systems (BMS). The framework focuses on **critical functional safety**, simultaneously maximizing classification accuracy and slashing the **Critical Miss Detection Rate (MDR)** for battery thermal runaway hazards while significantly reducing hardware sensor requirements.

---

## 1. Problem Formulation & Motivation

In electric vehicle batteries, undetected thermal runaway events can lead to catastrophic cell rupture, toxic gas emission, and pack fire. A robust BMS must satisfy three critical engineering goals:
1. **Minimize Critical Miss Detection Rate (MDR)**: Prevent life-threatening false negatives on thermal runaway onset.
2. **High Generalization Accuracy & Macro-F1**: Accurately differentiate between normal charging, elevated stress, and critical runaway across balanced multi-class evaluation.
3. **Sensor Footprint Compression**: Minimize the active sensor suite to cut vehicle wiring complexity, ECU bus overhead, and sensor hardware cost.

---

## 2. Proposed Framework: HGWO-PSO

The proposed **HGWO-PSO** framework introduces:
1. **Hybrid Metaheuristic Search**: Integrates the social leadership hierarchy ($\alpha, \beta, \delta$ wolves) of Grey Wolf Optimization (GWO) with the dynamic velocity and inertia exploration of Particle Swarm Optimization (PSO).
2. **Safety Co-Optimization**: Co-tunes the 17-channel physical sensor mask alongside Random Forest hyperparameters (`n_estimators`, `max_depth`, and critical class penalty) under a multi-objective safety fitness function.
3. **Hardware Reduction**: Compresses the active sensor footprint from 17 channels down to **5 essential channels** (a 70.6% reduction in physical hardware).

---

## 3. Benchmark Results (Strict 5-Fold Stratified Cross-Validation)

| Phase / Method | Optimization Framework | Classifier | Accuracy | Macro-F1 | Critical MDR (Thermal Runaway) | Active Sensors | Selected Sensor Channels |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1** | BPSO (Paper Baseline) | KNN ($k=5$) | 64.80% | 0.4209 | **59.14%** | 8 / 17 | ChargingStage, DemandCurrent_A, AvgTemp_C, AmbientTemp_C, StateOfHealth_%, VibrationLevel_mg, MoistureDetected, ChargePower_kW |
| **Baseline 2** | BWOA (Paper Baseline) | KNN ($k=5$) | 57.00% | 0.3543 | **73.01%** | 4 / 17 | DemandCurrent_A, StateOfHealth_%, VibrationLevel_mg, MoistureDetected |
| **Baseline 3** | BHPWOA (Paper Baseline) | KNN ($k=5$) | 64.80% | 0.4209 | **59.14%** | 8 / 17 | ChargingStage, DemandCurrent_A, AvgTemp_C, AmbientTemp_C, StateOfHealth_%, VibrationLevel_mg, MoistureDetected, ChargePower_kW |
| **Phase 1** | Co-Opt RF | RF (Co-Tuned) | 67.80% | 0.4466 | **34.22%** | 6 / 17 | ChargingStage, ChargeCurrent_A, SOC_%, StateOfHealth_%, MoistureDetected, Pressure_kPa |
| **Proposed HGWO-PSO (Ours)** | **HGWO-PSO (Novel Hybrid)** | **RF (HGWO-PSO Co-Tuned)** | **94.20%** | **0.8803** | **5.89%** | **5 / 17** | **DemandVoltage_V, SOC_%, MaxTemp_C, AvgTemp_C, MoistureDetected** |

---

## 4. Key Findings

* **Critical Hazard Detection**: Proposed HGWO-PSO slashes the Critical Miss Detection Rate to **5.89%**, successfully flagging **94.11% of all thermal runaway hazards**, whereas paper baselines miss 59% to 73% of critical events.
* **Accuracy & Macro-F1 Dominance**: HGWO-PSO achieves **94.20% accuracy** and a **0.8803 Macro-F1 score**, significantly outperforming all paper baselines.
* **Optimal 5-Sensor Suite**: The framework converged on 5 sensors (`DemandVoltage_V`, `SOC_%`, `MaxTemp_C`, `AvgTemp_C`, and `MoistureDetected`), demonstrating that thermal and charging electrical dynamics provide near-complete observability of thermal runaway onset.

---

## 5. How to Run on Google Colab / Local

```bash
# 1. Clone repository
git clone https://github.com/abdullahgni/BMS.git
cd BMS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full benchmark table (Takes ~5 seconds)
python run_final_hgwo_benchmark.py

# 4. (Optional) Run HGWO-PSO evolutionary search from scratch
python hgwo_pso_bms.py
```
