# Optimization of Battery Management Systems (BMS) in EVs with Safety-Prioritized Critical Fault Detection

This repository contains the official, fully reproducible implementation of **HGWO-PSO (Hybrid Grey Wolf Optimizer + Particle Swarm Optimization)** for feature selection and hyperparameter co-tuning in Electric Vehicle (EV) Battery Management Systems (BMS).

The framework focuses on **critical functional safety**, simultaneously maximizing classification accuracy, eliminating the **Critical Missed Detection Rate (MDR)** for catastrophic faults (thermal runaway and end-of-life battery retirement), and drastically compressing physical hardware sensor requirements.

---

## 1. Key Research Highlights

1. **Multi-Dataset Validation**:
   - **Dataset 1 (EV Fast-Charging Thermal Runaway)**: 1,000 samples, 17 candidate physical sensor channels. Slashes Critical MDR from **73.01% down to 5.89%** while reducing sensors from 17 down to **5 channels**.
   - **Dataset 2 (NASA Classic 4-Cell Li-ion Benchmark)**: 636 cycles (B0005, B0006, B0007, B0018) at 24°C. Slashes Critical MDR to **0.00%** with **99.21% accuracy** using only **3 physical sensors** (`Discharge_Duration_s`, `Energy_Delivered_Wh`, `Max_Load_Current_A`).
   - **Dataset 3 (NASA Global Multi-Temperature Archive)**: 2,750 cycles across 34 battery cells under extreme environmental thermal conditions (4°C, 22°C, 24°C, 43°C, 44°C). Achieves **98.36% accuracy** and **0.47% Critical MDR** using only **3 physical sensors**.

2. **Strict Cross-Validation**:
   - Evaluated under **5-Fold Stratified Cross-Validation** with fixed random seeds for 100% mathematical reproducibility.

3. **Multi-Objective Safety-Critical Fitness**:
   $$\text{Fitness} = -0.50 \cdot \text{Acc} - 0.50 \cdot \text{F1}_{\text{macro}} + 0.35 \cdot \text{MDR}_{\text{crit}} + 0.15 \cdot \left(\frac{d_{\text{sel}}}{D}\right)$$

---

## 2. Experimental Benchmark Results

### Table 7.1: EV Fast-Charging Thermal Runaway Benchmark (1,000 Samples, 5-Fold CV)
| Phase / Method | Classifier | Accuracy | Macro-F1 | Critical MDR | Sensors | Selected Sensor Channels |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1: BPSO** | KNN ($k=5$) | 64.80% | 0.4209 | 59.14% | 8 / 17 | ChargingStage, DemandCurrent_A, AvgTemp_C, AmbientTemp_C, StateOfHealth_%, VibrationLevel_mg, MoistureDetected, ChargePower_kW |
| **Baseline 2: BWOA** | KNN ($k=5$) | 57.00% | 0.3543 | 73.01% | 4 / 17 | DemandCurrent_A, StateOfHealth_%, VibrationLevel_mg, MoistureDetected |
| **Baseline 3: BHPWOA** | KNN ($k=5$) | 64.80% | 0.4209 | 59.14% | 8 / 17 | ChargingStage, DemandCurrent_A, AvgTemp_C, AmbientTemp_C, StateOfHealth_%, VibrationLevel_mg, MoistureDetected, ChargePower_kW |
| **Phase 1: Co-Opt RF** | RF (Co-Tuned) | 67.80% | 0.4466 | 34.22% | 6 / 17 | ChargingStage, ChargeCurrent_A, SOC_%, StateOfHealth_%, MoistureDetected, Pressure_kPa |
| **Proposed HGWO-PSO (Ours)** | **RF (HGWO-PSO)** | **94.20%** | **0.8803** | **5.89%** | **5 / 17** | **DemandVoltage_V, SOC_%, MaxTemp_C, AvgTemp_C, MoistureDetected** |

### Table 7.2: NASA Classic 4-Cell Li-ion Benchmark (636 Cycles, 5-Fold CV)
| Method / Framework | Classifier | Accuracy | Macro-F1 | Critical MDR | Sensors | Selected Physical Sensors |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1: BPSO** | KNN ($k=5$) | 97.48% | 0.9724 | 2.31% | 4 / 15 | Temp_Rise_C, Final_Voltage_V, Avg_Current_A, Discharge_Duration_s |
| **Baseline 2: BWOA** | KNN ($k=5$) | 97.64% | 0.9756 | 1.54% | 3 / 15 | Max_Temp_C, Avg_Voltage_V, Energy_Delivered_Wh |
| **Standard Full-Feature RF** | RF (Default) | 99.21% | 0.9920 | 0.00% | 15 / 15 | All Features |
| **Proposed HGWO-PSO (Ours)** | **RF (Co-Tuned)** | **99.21%** | **0.9913** | **0.00%** | **3 / 15** | **Discharge_Duration_s, Energy_Delivered_Wh, Max_Load_Current_A** |

### Section 7.3.2: NASA Global Multi-Temperature Archive (34 Cells, 2,750 Cycles, 4°C-44°C)
| Method / Framework | Classifier | Accuracy | Macro-F1 | Critical MDR | Sensors | Selected Physical Sensors |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1: BPSO** | KNN ($k=5$) | 97.45% | 0.9676 | 1.02% | 4 / 15 | Avg_Temp_C, Avg_Current_A, Energy_Delivered_Wh, Max_Load_Current_A |
| **Baseline 2: BWOA** | KNN ($k=5$) | 97.78% | 0.9713 | 0.86% | 2 / 15 | Discharge_Duration_s, Energy_Delivered_Wh |
| **Standard Full-Feature RF** | RF (Default) | 98.29% | 0.9781 | 0.55% | 15 / 15 | All Features |
| **Proposed HGWO-PSO (Ours)** | **RF (Co-Tuned)** | **98.36%** | **0.9789** | **0.47%** | **3 / 15** | **Discharge_Duration_s, Energy_Delivered_Wh, Avg_Load_Voltage_V** |

---

## 3. How to Run and Verify All Final Values

### Option A: 1-Click Verification Script (~5 seconds)
To verify all tables across both datasets (EV and NASA benchmarks):

```bash
# 1. Clone repository
git clone https://github.com/abdullahgni/BMS.git
cd BMS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete verification for all datasets
python verify_all_final_values.py
```

To run individual dataset verification:
```bash
python verify_all_final_values.py --dataset ev
python verify_all_final_values.py --dataset nasa_classic
python verify_all_final_values.py --dataset nasa_global
```

### Option B: Google Colab
Open `verify_bms_benchmarks.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells to verify the tables and view cross-dataset comparison plots.

### Option C: Run Full Evolutionary Optimization From Scratch
To run the evolutionary metaheuristic search from scratch:
- **EV Dataset**: `python hgwo_pso_bms.py`
- **NASA Classic 4 Dataset**: `python run_nasa_hgwo_benchmark.py`
- **NASA Global 2,750 Dataset**: `python run_nasa_global_benchmark.py`

---

## 4. Repository Structure

```text
├── EV_Battery_Charging_TR_Dataset_with_Notes.csv   # 1,000-sample EV thermal runaway dataset
├── nasa_battery_benchmark_classic4.csv              # 636-cycle NASA classic 4-cell dataset
├── nasa_battery_benchmark_global_2750.csv          # 2,750-cycle NASA global multi-temp archive
├── nasa_battery_benchmark_all24c.csv                # 1,384-cycle NASA 24°C multi-cell dataset
├── verify_all_final_values.py                       # Unified 1-click verification script
├── verify_bms_benchmarks.ipynb                      # Colab notebook for interactive reproduction
├── run_final_hgwo_benchmark.py                      # EV dataset benchmark runner
├── run_nasa_hgwo_benchmark.py                       # NASA classic 4 benchmark runner
├── run_nasa_global_benchmark.py                     # NASA global 2,750 benchmark runner
├── hgwo_pso.py                                      # Universal HGWO-PSO algorithm implementation
├── hgwo_pso_bms.py                                  # EV BMS-specific HGWO-PSO optimizer
├── generate_shap_analysis.py                        # SHAP explainability analysis generator
├── plot_cross_dataset.py                            # Cross-dataset visualization plotting
├── requirements.txt                                 # Python dependencies
└── README.md                                        # Documentation and guide
```
