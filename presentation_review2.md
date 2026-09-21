# Project Review-2 Presentation: Size Optimization in BMS for Feature Reduction

---

## Slide 1: Problem Statement & Research Gap

### Background & Literature Review
- Electric Vehicle (EV) Battery Management Systems (BMS) monitor real-time multi-sensor streams (Pack Voltage, Cell Voltage, Demand Voltage, Current, Temperatures, Internal Resistance, SOH, Pressure, Moisture, Vibration).
- **Literature Baseline (Mayingi et al., 2026):** Introduces Binary Hybrid PSO-WOA (BHPWOA) to select features evaluated on a 3-Class dataset (`OK`, `Warning`, `Critical`) using a KNN ($k=5$, Euclidean distance) classifier.

### The Research Gap & Limitations of Baseline Paper
1. **Un-Tuned Classifiers:** The baseline paper evaluates feature masks using a fixed KNN classifier, ignoring classifier hyperparameter tuning.
2. **High Missed Detection Rate (MDR):** Single-split evaluation with KNN yields a high **Missed Detection Rate (MDR > 91% for Critical faults)** due to severe class imbalance.
3. **Overfitting & Black-Box Decisions:** Single train-test splits (susceptible to data leakage) lack explainable AI (XAI) safety rationales required for ISO 26262 automotive safety compliance.

---

## Slide 2: Research Objectives

1. **Exact Reproduction of Baseline Paper:** Reproduce the 3-Class (`OK`/`Warning`/`Critical`) BPSO, BWOA, and BHPWOA algorithms using KNN ($k=5$) as published.
2. **Phase 1 (Joint Co-Optimization Extension):** Extend the particle chromosome space to simultaneously optimize feature selection masks $\mathbf{x}_{feat} \in \{0,1\}^{18}$ and Random Forest hyperparameters (`n_estimators`, `max_depth`).
3. **Phase 2 (Robust & Imbalance-Aware Evaluation):** Enforce Stratified 5-Fold Cross-Validation with `balanced` class weighting to eliminate single-split memorization and minimize Critical Missed Detection Rate (MDR).
4. **Explainable AI (XAI) Verification:** Integrate SHAP (SHapley Additive exPlanations) TreeExplainer to validate physical domain contributions of retained sensors.

---

## Slide 3: Proposed System Overview

```
+-----------------------------------------------------------------------------------+
|                         PROPOSED SYSTEM ARCHITECTURE                              |
+-----------------------------------------------------------------------------------+
| 1. Real-World BMS Telemetry Ingestion (18 Sensor Features)                        |
| 2. 3-Class Label Encoding (OK = 0, Warning = 1, Critical = 2)                      |
|                                                                                   |
| [ BASELINE COMPARISON ]                                                           |
|  - BPSO / BWOA / BHPWOA using fixed KNN (k=5, Euclidean)                          |
|                                                                                   |
| [ PROPOSED EXTENSION ]                                                            |
|  - Phase 1: Joint Co-Optimizer [Feature Mask | n_estimators | max_depth]            |
|  - Phase 2: Stratified 5-Fold CV + Class Balancing + SHAP XAI Engine              |
+-----------------------------------------------------------------------------------+
```

---

## Slide 4: System Diagram (Description for Drawing)

### Visual Block Diagram Layout
```
[ BMS Data Stream (18 Features) ] 
               │
               ▼
[ 3-Class Target Encoding (OK, Warning, Critical) ]
               │
       ┌───────┴─────────────────────────────┐
       ▼                                     ▼
[ PAPER BASELINES ]                [ PROPOSED PHASES 1 & 2 ]
 ├── BPSO (KNN k=5)                 ├── Phase 1: Co-Tuned RF [Mask | Hyps]
 ├── BWOA (KNN k=5)                 └── Phase 2: Stratified 5-Fold CV
 └── BHPWOA (KNN k=5)                       + Class-Balanced Weights
       │                                     │
       └───────┬─────────────────────────────┘
               ▼
 [ Comparative Performance Table & SHAP XAI Analysis ]
```

---

## Slide 9: Implementation Results & Project Progress (~50% Complete)

### Benchmark Results Table (3-Class Setup)

| Model | Classifier Engine | Accuracy | Macro-F1 | MDR (Critical Class) | Selected Sensors | Key Retained Sensors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. BPSO (Paper Baseline)** | KNN ($k=5$) | 0.7100 | 0.4308 | 0.7500 | 9 / 19 | `ChargingStage`, `DemandCurrent_A`, `AvgTemp_C`, `AmbientTemp_C`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected`, `ChargePower_kW`, `TR_Probability` |
| **2. BWOA (Paper Baseline)** | KNN ($k=5$) | 0.7100 | 0.3970 | 0.8333 | 4 / 19 | `DemandCurrent_A`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected` |
| **3. BHPWOA (Paper Baseline)**| KNN ($k=5$) | 0.7100 | 0.4308 | 0.7500 | 9 / 19 | `ChargingStage`, `DemandCurrent_A`, `AvgTemp_C`, `AmbientTemp_C`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected`, `ChargePower_kW`, `TR_Probability` |
| **4. Phase 1 (Proposed Co-Opt)**| RandomForest (Co-Tuned) | **0.7200** | 0.4120 | 0.5000 | 8 / 19 | `CellVoltage_V`, `ChargeCurrent_A`, `SOC_%`, `MinTemp_C`, `InternalResistance_mOhm`, `Pressure_kPa`, `TR_Probability`, `VibrationLevel_mg` |
| **5. Phase 2 (Proposed Robust)**| RandomForest (5-Fold CV) | **0.7000** *(vs 0.666 CV Baseline)* | **0.3869** *(vs 0.294 CV Baseline)* | **0.2394 (Lowest MDR)** | **10 / 19** | `CellVoltage_V`, `PackVoltage_V`, `ChargeCurrent_A`, `DemandCurrent_A`, `SOC_%`, `MaxTemp_C`, `MinTemp_C`, `InternalResistance_mOhm`, `Pressure_kPa`, `TR_Probability` |

### Narrative for Presentation:
1. **Paper Baselines Reproduction (Models 1–3):** BHPWOA and BPSO (KNN $k=5$) achieve 0.7100 Accuracy and 0.4308 Macro-F1 on single test splits, selecting 9 physical sensors (`9/19`). However, KNN exhibits high Missed Detection Rates (MDR = 75–83%) for critical safety events.
2. **Phase 1 Co-Optimizer (Model 4):** Jointly co-tuning Random Forest hyperparameters with feature masks achieves the highest single-split Accuracy (**0.7200**) with 8 features while cutting Critical MDR down to **50.0%**.
3. **Phase 2 Proposed Robust System (Model 5):** Evaluated across Stratified 5-Fold Cross-Validation, Phase 2 retains **10 comprehensive physical sensors** (covering voltage, current, temperature, internal resistance, pressure, and risk metrics) while outperforming the baseline algorithm under identical 5-Fold CV (Acc **0.7000** vs 0.6660, Macro-F1 **0.3869** vs 0.2943) and achieving the **lowest Missed Detection Rate (23.94%)**.

