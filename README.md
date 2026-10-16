# Optimization of Battery Management Systems (BMS) in EVs with Safety-Prioritized Critical Fault Detection

This repository implements **HGWO-PSO (Hybrid Grey Wolf Optimizer + Particle Swarm Optimization)** for feature selection and hyperparameter co-tuning in Electric Vehicle (EV) Battery Management Systems (BMS). The framework specifically tackles the problem of **deceptive nominal accuracy** in imbalanced safety-critical data, prioritizing the reduction of **Critical Miss Detection Rate (MDR)** for battery thermal runaway hazards.

---

## 1. Motivation & The Safety Blindspot

In standard battery datasets, benign operational conditions represent ~70% of charging cycles. When baseline optimizers (BPSO, BWOA, BHPWOA) optimize standard loss metrics:
- They achieve nominal accuracy of **64%–71%**.
- However, they exhibit an alarming **94%–97% Miss Detection Rate (MDR) on the Critical class**.
- In practical EV applications, missing 19 out of 20 thermal runaway events constitutes a catastrophic failure mode.

---

## 2. Proposed Framework: HGWO-PSO

The proposed **HGWO-PSO** framework introduces:
1. **Hybrid Metaheuristic Search**: Combines the hunting pack hierarchy of Grey Wolf Optimization ($\alpha, \beta, \delta$ wolves) with the dynamic velocity/inertia exploration of Particle Swarm Optimization (PSO).
2. **Safety Co-Optimization**: Co-tunes the 19-dimensional sensor selection mask alongside Random Forest hyperparameters (`n_estimators`, `max_depth`, and critical class cost penalty) under a multi-objective safety fitness function.
3. **Hardware Reduction**: Compresses the active sensor footprint from 19 channels down to **6 key channels** (a 68.4% reduction in physical hardware).

---

## 3. Benchmark Results (Strict 5-Fold Stratified Cross-Validation)

| Phase / Method | Optimization Framework | Classifier | Accuracy | Macro-F1 | Critical MDR (Thermal Runaway) | Active Sensors | Selected Sensor Channels |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1** | BPSO (Paper Baseline) | KNN ($k=5$) | 64.0% | 0.3087 | **94.70%** | 9 / 19 | ChargingStage, DemandCurrent_A, AvgTemp_C, AmbientTemp_C, StateOfHealth_%, VibrationLevel_mg, MoistureDetected, ChargePower_kW, TR_Probability |
| **Baseline 2** | BWOA (Paper Baseline) | KNN ($k=5$) | 64.2% | 0.2846 | **96.67%** | 4 / 19 | DemandCurrent_A, StateOfHealth_%, VibrationLevel_mg, MoistureDetected |
| **Baseline 3** | BHPWOA (Paper Baseline) | KNN ($k=5$) | 64.0% | 0.3087 | **94.70%** | 9 / 19 | ChargingStage, DemandCurrent_A, AvgTemp_C, AmbientTemp_C, StateOfHealth_%, VibrationLevel_mg, MoistureDetected, ChargePower_kW, TR_Probability |
| **Phase 1** | Co-Opt RF | RF (Co-Tuned) | 67.6% | 0.2857 | **96.52%** | 6 / 19 | ChargingStage, ChargeCurrent_A, SOC_%, StateOfHealth_%, MoistureDetected, Pressure_kPa |
| **Proposed HGWO-PSO (Safety-Prioritized)** | **HGWO-PSO (Novel Hybrid)** | **RF (HGWO-PSO Co-Tuned)** | **50.2%** | **0.3141** | **42.88%** *(down to 35.9%)* | **6 / 19** | **CellVoltage_V, ChargeCurrent_A, DemandCurrent_A, MinTemp_C, AvgTemp_C, TR_Probability** |
| **Proposed HGWO-PSO (Balanced-Safety)** | **HGWO-PSO (Novel Hybrid)** | **RF (HGWO-PSO Co-Tuned)** | **61.4%** | **0.3123** | **79.24%** | **6 / 19** | **CellVoltage_V, ChargeCurrent_A, DemandCurrent_A, MinTemp_C, AvgTemp_C, TR_Probability** |

---

## 4. Key Findings

- **Tradeoff Frontier**: High accuracy on standard models is obtained solely by predicting the majority non-critical class. Our safety-focused HGWO-PSO reduces critical hazard misses by over 50 percentage points.
- **Optimal Sensor Suite**: Thermal and electrical sensors (`CellVoltage_V`, `ChargeCurrent_A`, `DemandCurrent_A`, `MinTemp_C`, `AvgTemp_C`, and `TR_Probability`) contain sufficient diagnostic signals to detect onset hazards without mechanical noise from vibration or moisture sensors.

---

## 5. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run HGWO-PSO Feature & Parameter Co-Optimization
```bash
python hgwo_pso_bms.py
```

### Run Unified Final Benchmark Across All Baselines
```bash
python run_final_hgwo_benchmark.py
```
