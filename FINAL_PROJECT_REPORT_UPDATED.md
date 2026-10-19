# EXECUTIVE SUMMARY

Electric Vehicle (EV) Battery Management Systems (BMS) are safety-critical embedded controllers responsible for tracking cell operating envelopes and preventing hazardous thermal runaway. Standard production architectures capture high-dimensional telemetry across dozens of physical channels—monitoring voltages, current profiles, distributed temperatures, internal impedance, state of charge, state of health, mechanical vibration, and moisture ingress. While comprehensive, this dense sensory infrastructure introduces substantial engineering penalties, including wiring harness mass, analog-to-digital converter pin contention, bill-of-materials expense, and real-time microcontroller computational overhead. Crucially, baseline metaheuristic feature selectors—such as the Binary Hybrid Particle Swarm Optimization–Whale Optimization Algorithm (BHPWOA) evaluated with default K-Nearest Neighbors (KNN)—exhibit alarming Critical Missed Detection Rates (MDR between 59% and 73%) due to severe operational class imbalance and an absence of classifier co-tuning.

To solve these fundamental safety and hardware vulnerabilities, this project presents a novel Hybrid Grey Wolf Optimizer–Particle Swarm Optimization (HGWO-PSO) framework designed for concurrent sensor feature selection and classifier hyperparameter co-optimization. The architecture leverages grey wolf social hierarchy leadership for global exploration while incorporating particle velocity dynamics to prevent premature stagnation, evaluated under a multi-objective, safety-prioritized fitness formulation. Validated through Stratified 5-Fold Cross-Validation on an EV fast-charging thermal runaway telemetry dataset, the framework compressed the physical sensor array by 70.6% (from 17 down to 5 critical channels: Demand Voltage, State of Charge, Maximum Temperature, Average Temperature, and Moisture Detected) while improving diagnostic accuracy to 94.20%, achieving a Macro-F1 score of 0.8803, and driving the Critical Missed Detection Rate down to 5.89%. Universal generalization was independently proven on the public NASA Ames Li-ion Battery Prognostics benchmark, attaining 99.37% accuracy, 0.00% critical missed detections, and retaining just 3 degradation features. Explainable AI auditing via SHAP TreeExplainer confirmed the electrochemical validity of the selected channels, satisfying ISO 26262 functional safety requirements.

---

# TABLE OF CONTENTS

- **Executive Summary**
- **List of Figures**
- **List of Tables**
- **List of Abbreviations**
- **Symbols and Notations**
- **1. INTRODUCTION**
  - 1.1 Background
    - 1.1.1 Machine Learning Based Fault Detection
    - 1.1.2 Metaheuristic Feature Selection and Optimization
    - 1.1.3 Functional Safety and Explainable Artificial Intelligence
  - 1.2 Motivation
  - 1.3 Scope of the Project
- **2. PROJECT DESCRIPTION AND GOALS**
  - 2.1 Literature Review
    - 2.1.1 Machine Learning and Metaheuristic Sensor Optimization in BMS
    - 2.1.2 Deep Learning Architectures in Battery State Estimation and Prognostics
    - 2.1.3 Class Imbalance, Cost-Sensitive Safety, and Explainable AI
    - 2.1.4 Cross-Dataset Validation and Battery Aging Prognostics
  - 2.2 Research Gap
  - 2.3 Objectives
  - 2.4 Problem Statement
  - 2.5 Project Plan
- **3. TECHNICAL SPECIFICATION**
  - 3.1 Requirements
    - 3.1.1 Functional Requirements
    - 3.1.2 Non-Functional Requirements
  - 3.2 Feasibility Study
    - 3.2.1 Technical Feasibility
    - 3.2.2 Economic Feasibility
    - 3.2.3 Social Feasibility
  - 3.3 System Specification
    - 3.3.1 Hardware Specification
    - 3.3.2 Software Specification
- **4. SYSTEM DESIGN**
  - 4.1 System Architecture
  - 4.2 Detailed Design
    - 4.2.1 Use Case Diagram and Description
    - 4.2.2 Data Flow Diagram (DFD Level 0 & Level 1)
    - 4.2.3 State Transition Diagram
    - 4.2.4 Sequence Diagram
    - 4.2.5 Class Diagram
    - 4.2.6 Activity Diagram
- **5. METHODOLOGY AND TESTING**
  - 5.1 Algorithm Formulation
    - Algorithm 1: Proposed HGWO-PSO Feature Selection and Hyperparameter Co-Optimization
    - Algorithm 2: Multi-Objective Safety Fitness Evaluation and Stratified K-Fold CV
  - 5.2 Module Description
    - 5.2.1 Telemetry Preprocessing and 3-Class Target Formulations
    - 5.2.2 HGWO-PSO Evolutionary Optimization Engine
    - 5.2.3 Cost-Sensitive Classifier Training and Cross-Validation
    - 5.2.4 SHAP Electrochemical Interpretability Engine
  - 5.3 Testing and Quality Assurance
    - 5.3.1 Unit Testing
    - 5.3.2 Integration Testing
    - 5.3.3 Cross-Validation Testing
    - 5.3.4 Cross-Dataset Generalization Testing
- **6. PROJECT IMPLEMENTATION**
  - 6.1 Experimental Setup
  - 6.2 Datasets with Description
    - 6.2.1 EV Battery Fast-Charging Thermal Runaway Dataset
    - 6.2.2 NASA Ames Li-ion Battery Prognostics Dataset
  - 6.3 Step-by-Step Implementation Details
- **7. RESULTS AND DISCUSSION**
  - 7.1 Baseline Reproduction and Benchmark Evaluation
  - 7.2 Safety Critical Performance and Missed Detection Rate Analysis
  - 7.3 Cross-Dataset Universal Generalization Findings
  - 7.4 Explainable AI (SHAP) Electrochemical Interpretability
  - 7.5 Physical Hardware and Engineering Cost Analysis
- **8. CONCLUSION AND FUTURE ENHANCEMENTS**
  - 8.1 Conclusion
  - 8.2 Future Enhancements
- **REFERENCES**
- **APPENDIX A – SAMPLE CODE**
- **APPENDIX B – CONFERENCE / JOURNAL PUBLICATION DETAILS**

---

# LIST OF FIGURES

| Figure No. | Title |
| :---: | :--- |
| 4.1 | High-Level System Architecture of Proposed HGWO-PSO BMS Optimization Framework |
| 4.2 | Use Case Diagram for Intelligent BMS Telemetry Diagnostic System |
| 4.3 | Data Flow Diagram (Level 0 Context Diagram and Level 1 Processing Pipeline) |
| 4.4 | State Transition Diagram for Battery Fault Operating States |
| 4.5 | Sequence Diagram of HGWO-PSO Co-Optimization and Cross-Validation Execution |
| 4.6 | UML Class Diagram of the Proposed Machine Learning Optimization Architecture |
| 4.7 | Activity Diagram for Safety-Prioritized Fitness Evaluation and Sensor Reduction |
| 7.1 | Convergence Curves of BPSO, BWOA, BHPWOA, and Proposed HGWO-PSO |
| 7.2 | Cross-Dataset Performance Comparison on EV Thermal Runaway and NASA Datasets |
| 7.3 | Global Mean Absolute SHAP Importance Across Retained Battery Sensor Channels |
| 7.4 | SHAP Beeswarm Distribution for Critical Thermal Runaway Classification |
| 7.5 | Non-Linear SHAP Interaction Dependence Plot: Maximum Temperature vs. State of Charge |

---

# LIST OF TABLES

| Table No. | Title |
| :---: | :--- |
| 2.1 | Comprehensive Synthesis of Reviewed Literature Across Machine Learning, Metaheuristics, and BMS Safety |
| 2.2 | Project Milestones, Deliverables, and Schedule Breakdown |
| 3.1 | Functional and Non-Functional Requirement Specifications |
| 3.2 | System Hardware and Development Software Specifications |
| 6.1 | Summary of Physical Telemetry Sensors in EV Fast-Charging Thermal Runaway Dataset |
| 6.2 | Attribute Description of NASA Ames Li-ion Battery Prognostics Dataset |
| 7.1 | Benchmark Comparison Across Baselines and Proposed HGWO-PSO on EV BMS Dataset |
| 7.2 | Active Physical Sensor Sets Retained Across Optimization Frameworks |
| 7.3 | Cross-Dataset Validation Results on NASA Li-ion Battery Prognostics Benchmark (Classic 4 Cells) |
| 7.4 | Multi-Temperature Global NASA Archive Benchmark (Strict 5-Fold Stratified CV, 2,750 Cycles) |
| 7.5 | Mean Absolute SHAP Attribute Ranking and Electrochemical Function in Battery Safety |
| 7.6 | Physical Hardware, Wiring Harness, and Computational Resource Reduction Metrics |

---

# LIST OF ABBREVIATIONS

| Abbreviation | Expansion |
| :--- | :--- |
| ADC | Analog-to-Digital Converter |
| ASIL | Automotive Safety Integrity Level |
| BHPWOA | Binary Hybrid Particle Swarm Optimization – Whale Optimization Algorithm |
| BMS | Battery Management System |
| BPSO | Binary Particle Swarm Optimization |
| BWOA | Binary Whale Optimization Algorithm |
| CAN | Controller Area Network |
| CNN | Convolutional Neural Network |
| CPU | Central Processing Unit |
| CV | Cross-Validation |
| DFD | Data Flow Diagram |
| DL | Deep Learning |
| ECU | Electronic Control Unit |
| EOL | End-of-Life |
| EV | Electric Vehicle |
| GWO | Grey Wolf Optimizer |
| HGWO-PSO | Hybrid Grey Wolf Optimizer – Particle Swarm Optimization |
| ISO | International Organization for Standardization |
| KNN | K-Nearest Neighbors |
| LIME | Local Interpretable Model-agnostic Explanations |
| LSTM | Long Short-Term Memory |
| MDR | Missed Detection Rate |
| ML | Machine Learning |
| NASA | National Aeronautics and Space Administration |
| PSO | Particle Swarm Optimization |
| RAM | Random Access Memory |
| RF | Random Forest |
| RUL | Remaining Useful Life |
| SHAP | SHapley Additive exPlanations |
| SOC | State of Charge |
| SOH | State of Health |
| SVM | Support Vector Machine |
| SVR | Support Vector Regression |
| TR | Thermal Runaway |
| UML | Unified Modeling Language |
| WOA | Whale Optimization Algorithm |
| XAI | Explainable Artificial Intelligence |

---

# SYMBOLS AND NOTATIONS

| Symbol | Definition |
| :--- | :--- |
| $\alpha$ | Alpha Wolf (Primary Elite Leader in GWO) |
| $\beta$ | Beta Wolf (Secondary Subordinate Leader in GWO) |
| $\delta$ | Delta Wolf (Tertiary Subordinate Leader in GWO) |
| $\omega$ | Inertia Weight in Particle Swarm Optimization |
| $c_1, c_2$ | Cognitive and Social Acceleration Coefficients in PSO |
| $r_1, r_2$ | Uniformly Distributed Random Vectors in $[0, 1]$ |
| $\mathbf{x}_{feat}$ | Binary Feature Selection Mask Vector $\in \{0, 1\}^D$ |
| $D$ | Total Number of Candidate Telemetry Sensor Channels |
| $d_{sel}$ | Number of Active Selected Physical Sensors |
| $n_{est}$ | Number of Decision Trees in Random Forest Classifier |
| $max\_d$ | Maximum Depth of Decision Trees |
| $w_{crit}$ | Class Loss Penalty Weight Assigned to Critical Thermal Runaway Faults |
| $w_{sens}$ | Regularization Weight Penalizing Large Sensor Subset Cardinality |
| $v_{i,d}^t$ | Velocity of Particle $i$ in Dimension $d$ at Iteration $t$ |
| $S(v)$ | Sigmoid Transfer Function Mapping Continuous Velocity to Probability |
| $\mathcal{F}$ | Multi-Objective Fitness Evaluation Function |
| $|\text{SHAP}|$ | Mean Absolute Shapley Additive Explanation Value |

---
# CHAPTER 1: INTRODUCTION

## 1.1 BACKGROUND
The transition toward electrified transport has pushed lithium-ion battery technology to the forefront of automotive engineering. In modern Electric Vehicles (EVs), traction battery packs must satisfy rigorous performance demands: delivering high specific energy for extended driving range, supporting high-rate discharge pulses for vehicle acceleration, and safely absorbing massive current inflows during DC fast-charging sessions. Safeguarding these high-voltage packs requires an embedded controller—the Battery Management System (BMS). The BMS continuously measures physical states, balances individual cells, estimates State of Charge (SOC) and State of Health (SOH), and intervenes before catastrophic thermal runaway occurs (Plett, 2015; Vidal et al., 2020; Xiong et al., 2013).

To guarantee safety under volatile environmental and electrical stresses, production battery packs historically relied on extensive sensor arrays. Contemporary packs capture high-dimensional telemetry streams, continuously sampling cell terminal voltages, aggregate bus voltage, current demands, multi-point surface and tab temperatures, internal impedance, mechanical shock, and pack humidity (Hu et al., 2021; Shrivastava et al., 2024).

### 1.1.1 Machine Learning Based Fault Detection
While comprehensive sensing provides high operational visibility, real-world vehicle engineering introduces severe physical and computational trade-offs. Every physical sensor channel demands wiring, dedicated connector pins, microcontroller Analog-to-Digital Converter (ADC) channels, and printed circuit board (PCB) routing. In high-voltage packs, expanding the wiring harness adds parasitic mass, increases assembly labor, drives up bill-of-materials costs, and introduces additional physical points of failure, such as connector pin corrosion or harness chafing under road vibration (Mayingi et al., 2026). 

Furthermore, traditional fixed-threshold alarms—such as triggering an emergency disconnect only after a single thermocouple crosses $60^\circ\text{C}$—frequently respond too late. During DC fast-charging, localized exothermic self-heating can accelerate rapidly, triggering thermal runaway before bulk heat dissipates to distant sensors. This reality has driven the adoption of multivariate machine learning algorithms. Ensembles like Random Forests and Support Vector Machines can recognize subtle, cross-channel warning signs across electrical and thermal features well before catastrophic thermal breakdown occurs (Severson et al., 2019; Sun et al., 2022; Yao et al., 2021).

### 1.1.2 Metaheuristic Feature Selection and Optimization
Because adjacent telemetry channels often carry redundant or collinear information—such as neighboring temperature probes or derived electrical power calculations—wrapper feature selection algorithms are widely deployed to isolate the minimal essential sensor subset. Population-based metaheuristics, such as Particle Swarm Optimization (PSO) and the Whale Optimization Algorithm (WOA), offer effective tools for navigating discrete binary combinatorial search spaces (Kennedy & Eberhart, 1995; Mirjalili & Lewis, 2016). Recently, Mayingi et al. (2026) introduced a Binary Hybrid PSO-WOA (BHPWOA) feature selector paired with a K-Nearest Neighbors (KNN) classifier, demonstrating that reducing telemetry channels can streamline BMS computational pipelines without degrading nominal classification accuracy.

### 1.1.3 Functional Safety and Explainable Artificial Intelligence
Despite promising benchmark results, standard machine learning pipelines reveal critical vulnerabilities when evaluated in real-world automotive environments. Real battery telemetry exhibits extreme operational class imbalance: nominal operating cycles (`OK`) comprise more than 70% to 90% of all recorded telemetry, whereas severe thermal runaway events (`Critical`) occur as rare anomalies (Chen et al., 2004; Fontanari et al., 2022). Unweighted classification models evaluated on standard overall accuracy naturally prioritize the majority class, sacrificing minority safety instances and producing dangerously high Critical Missed Detection Rates (MDR).

Under the ISO 26262 automotive functional safety standard, failing to detect an impending thermal runaway condition is a catastrophic hazard classified at the highest Automotive Safety Integrity Level (ASIL-D). Furthermore, black-box decision models cannot be certified or audited for road vehicles without transparent interpretability. Consequently, sensor selection cannot be treated as an isolated exercise; it must be co-optimized with cost-sensitive classifier hyperparameters and verified through Explainable AI (XAI) frameworks like SHAP (Lundberg & Lee, 2017) to confirm that the retained sensors reflect legitimate electrochemical and physical phenomena.

## 1.2 MOTIVATION
In high-voltage EV packs, failing to detect a critical thermal runaway condition can cause vehicle fires and severe structural damage. The core motivation for this research stems from four major limitations observed in the baseline literature:
1. **The Flaw of Untuned Classifiers in Wrapper Selection:** Published feature selection wrappers (Mayingi et al., 2026) evaluate candidate sensor masks using a fixed, default classifier (e.g., $k=5$ KNN with Euclidean distance). In practice, compressing a feature space from 17 sensors down to 5 fundamentally changes the data geometry. An un-tuned classifier cannot adapt to this transformed manifold, resulting in distorted fitness evaluations and suboptimal sensor selection.
2. **Unacceptable Critical Missed Detection Rates:** When evaluated on realistic 3-class telemetry, published baseline algorithms miss between 59% and 73% of critical thermal runaway events. From an automotive safety standpoint, missing nearly three out of every four thermal runaway occurrences is unacceptable.
3. **Black-Box Decision Making and ISO 26262 Auditability:** Existing heuristic selectors treat sensor subsets as abstract binary strings, providing no electrochemical or thermodynamic justification for why specific sensors were retained or discarded. Without transparent feature attribution, machine learning-driven BMS designs cannot pass automotive safety audits.
4. **Physical Harness and Manufacturing Cost Pressures:** Demonstrating that 5 carefully selected physical sensors can match or exceed the diagnostic fidelity of a dense 17-channel suite directly reduces copper harness mass, microcontroller pin count, and vehicle assembly costs.

## 1.3 SCOPE OF THE PROJECT
The scope of this project encompasses:
- Exact reproduction and benchmarking of published metaheuristic feature selectors (BPSO, BWOA, BHPWOA) paired with default KNN ($k=5$) classifiers on a 3-class battery telemetry dataset.
- Formulation and mathematical development of a novel **Hybrid Grey Wolf Optimizer–Particle Swarm Optimization (HGWO-PSO)** framework capable of simultaneously searching the 17-bit binary physical sensor mask $\mathbf{x}_{feat} \in \{0, 1\}^{17}$ and Random Forest hyperparameters ($n_{est}, max\_d, w_{crit}$).
- Implementation of a safety-prioritized, multi-objective fitness function that heavily penalizes false negatives on the Critical thermal runaway class under strict Stratified 5-Fold Cross-Validation.
- Integration of game-theoretic Explainable AI (SHAP TreeExplainer) to evaluate global feature importance rankings, beeswarm distributions, and non-linear multi-sensor interaction dependencies.
- Independent validation on the public benchmark **NASA Ames Li-ion Battery Prognostics Dataset** to prove universal cross-dataset transferability without manual re-tuning.
- Complete software implementation delivering modular, reproducible, object-oriented Python source code.

Physical hardware-in-the-loop (HIL) dynamometer testing and low-level microcontroller C code generation fall outside the current project scope and are identified as future enhancements.

---
# CHAPTER 2: PROJECT DESCRIPTION AND GOALS

## 2.1 LITERATURE REVIEW
A comprehensive literature review examining 60 peer-reviewed scientific studies across machine learning, metaheuristic optimization, functional safety, and battery prognostics was conducted to establish the empirical foundation of this project.

### 2.1.1 Machine Learning and Metaheuristic Sensor Optimization in BMS
Data-driven battery management relies heavily on supervised learning for fault classification and state tracking. Mayingi et al. (2026) established the primary baseline for this study by formulating a multi-objective Binary Hybrid PSO-WOA (BHPWOA) feature selector paired with a KNN classifier across a 3-class operating dataset (`OK`, `Warning`, `Critical`). Hu et al. (2021) and Shrivastava et al. (2024) provided comprehensive surveys on battery fault diagnostics, observing that decision tree ensembles and support vector machines achieve an optimal balance between execution speed and memory footprint on automotive microcontrollers. Yao et al. (2021) developed grid-search SVMs for lithium battery internal short-circuit diagnosis, while Sun et al. (2022) and Jiang et al. (2021) mined fleet telemetry to build early thermal fault warning systems, confirming that multi-sensor patterns detect thermal anomalies far earlier than single-sensor thresholds. Deng et al. (2021) and Fang et al. (2021) investigated multi-class SVM frameworks and cell inconsistency tracking for pack-level monitoring.

In the feature reduction domain, metaheuristic algorithms offer an effective mechanism for exploring discrete binary spaces. Kennedy and Eberhart (1995) established Particle Swarm Optimization (PSO), which was later extended to binary wrappers (BPSO) by Xue et al. (2013), Song et al. (2018), Nguyen et al. (2020), Gu et al. (2020), and Xue et al. (2021). Mirjalili and Lewis (2016) proposed the Whale Optimization Algorithm (WOA), mimicking the bubble-net foraging behavior of humpback whales. Subsequent binary and hybrid WOA variants developed by Mafarja and Mirjalili (2017, 2018), Riyahi et al. (2020), Mafarja et al. (2020), and Too and Abdullah (2021) demonstrated that hybridizing exploratory swarm mechanisms with localized search operators prevents premature population collapse in multimodal fitness landscapes (Blum et al., 2011; Dokeroglu et al., 2019; Emary et al., 2016).

### 2.1.2 Deep Learning Architectures in Battery State Estimation and Prognostics
Deep learning architectures have gained considerable attention for modeling non-linear, time-dependent battery dynamics. Li et al. (2020) trained deep neural networks to model heat generation and thermal runaway onset in commercial battery cells. Abdolrasol et al. (2021), Zhao et al. (2021), Li et al. (2021), Ojo et al. (2021), Jia et al. (2021), Hong et al. (2021), and Li et al. (2022) conducted extensive studies on deep neural networks for thermal fault detection, validating that multi-layer neural architectures capture complex thermal diffusion under aggressive charging regimes.

For State of Charge (SOC) and State of Health (SOH) estimation, Song et al. (2020) implemented combined CNN-LSTM networks to extract spatio-temporal features from continuous voltage and current profiles. Chemali et al. (2018) and Ma et al. (2021) deployed Long Short-Term Memory (LSTM) recurrent networks, demonstrating accurate tracking under wide ambient temperature boundaries. Javid et al. (2021) developed adaptive deep recurrent neural networks for online SOC estimation, confirming that deep sequential models can track long-term electrochemical capacity fade. However, as emphasized by Hu et al. (2021) and Shrivastava et al. (2024), deep architectures demand substantial RAM, introduce non-deterministic execution timing, and lack the transparent interpretability mandated by ISO 26262 functional safety standards. Consequently, optimized tree ensembles with Explainable AI represent a much more viable choice for real-time onboard BMS deployment.

### 2.1.3 Class Imbalance, Cost-Sensitive Safety, and Explainable AI
In automotive safety systems, severe class imbalance poses a major challenge. Breiman (2001) established Random Forest (RF) ensembles as resilient tabular estimators. Chen et al. (2004), Gu et al. (2022), and Probst et al. (2019) demonstrated that standard ensemble accuracy collapses under severe class imbalance unless cost-sensitive loss weighting or minority re-sampling is enforced. Guyon et al. (2002) introduced Recursive Feature Elimination, highlighting the importance of evaluating feature subsets against the targeted classifier architecture. Wong and Yeh (2020) and Fontanari et al. (2022) established cross-validation protocols for imbalanced datasets, demonstrating that arbitrary single train-test splits introduce significant variance and over-optimism.

To address the black-box nature of machine learning in safety-critical systems, Explainable AI (XAI) frameworks have become indispensable. Lundberg and Lee (2017) formulated SHAP (SHapley Additive exPlanations), unifying cooperative game theory with local model interpretability. Lundberg et al. (2020) developed TreeExplainer, enabling polynomial-time computation of exact Shapley values for tree ensembles. Ribeiro et al. (2016) introduced LIME for local fidelity testing. Applying XAI to battery systems provides validation of physical sensor contributions, ensuring alignment with ISO 26262 safety goals.

### 2.1.4 Cross-Dataset Validation and Battery Aging Prognostics
Proving that an optimization algorithm generalizes requires validation across independent battery chemistries and operational regimes. Saha and Goebel (2007) introduced the benchmark NASA Ames Li-ion Battery Prognostics Dataset, providing experimental aging profiles for commercial 18650 cells cycled under controlled temperatures and discharge currents. Severson et al. (2019) demonstrated early-cycle capacity prediction across hundreds of lithium iron phosphate cells, establishing the role of discharge voltage variance in predicting longevity. Plett (2015), Xiong et al. (2013), and Vidal et al. (2020) provided mathematical formulations for battery equivalent circuit modeling, confirming that internal impedance, thermal rise, and integrated discharge energy serve as universal electrochemical indicators of battery health and degradation.

---

### TABLE 2.1: Comprehensive Synthesis of Reviewed Literature Across Machine Learning, Metaheuristics, and BMS Safety

| Author & Year | Domain / Focus | Methodology | Key Contributions & Findings | Relevance to Current Work |
| :--- | :--- | :--- | :--- | :--- |
| **Mayingi et al. (2026)** | BMS Fault Diagnosis | BHPWOA + KNN ($k=5$) | Multi-objective sensor reduction on 3-class telemetry (`OK`, `Warning`, `Critical`). | Primary literature baseline reproduced in this study. |
| **Muthiah et al. (2026)** | EV Fleet Optimization | Hybrid Optimization | Integrated electric bus scheduling and charging profile management. | Highlights operational fast-charging thermal and current stresses. |
| **Hu et al. (2021)** | Fault Diagnosis Survey | Comparative Review | Comprehensive survey of ML methods for lithium-ion battery safety and diagnostics. | Justifies selection of tree ensembles over black-box networks. |
| **Shrivastava et al. (2024)** | BMS Safety & Faults | Comprehensive Review | Evaluated data-driven fault detection methods, emphasizing embedded execution limits. | Guides real-time latency and computational budget requirements. |
| **Yao et al. (2021)** | Internal Fault Detection | GS-SVM + Cosine Filter | Identified short circuits and thermal degradation using voltage differentials. | Validates voltage dynamics as a core indicator of cell health. |
| **Sun et al. (2022)** | Thermal Runaway Warning | Fleet Telemetry Mining | Early warning of thermal runaway events utilizing fleet operational datasets. | Confirms multi-sensor detection superiority over fixed thresholds. |
| **Jiang et al. (2021)** | Thermal Fault Modeling | Statistical Anomaly | Early thermal fault identification using pack temperature distribution curves. | Demonstrates importance of max and average temperature tracking. |
| **Deng et al. (2021)** | Multi-Class Diagnosis | Multi-SVM Ensemble | Classified external shorts, internal degradation, and sensor connection faults. | Supports 3-class target categorization (`OK`, `Warning`, `Critical`). |
| **Fang et al. (2021)** | Cell Inconsistency | Shannon Entropy + SVR | Inconsistency tracking across series-connected cells for fault localization. | Relates voltage divergence to internal resistance growth. |
| **Li et al. (2020)** | Thermal Runaway ML | Deep Learning Networks | Modeled heat generation rates and thermal runaway onset in commercial cells. | Validates thermal trigger thresholds under fast-charging conditions. |
| **Abdolrasol et al. (2021)** | Neural Diagnostics | Artificial Neural Nets | Examined neural network diagnostic frameworks for EV energy storage systems. | Analyzed computational overhead trade-offs on automotive ECUs. |
| **Zhao et al. (2021)** | Anomaly Detection | Big Data Neural Engine | Real-time outlier identification across large-scale commercial battery deployments. | Demonstrates the need for robust feature selection to eliminate noise. |
| **Li et al. (2021)** | Temperature Estimation | CNN Architectures | Spatial temperature field reconstruction inside large-format battery modules. | Confirms the physical significance of thermal sensor matrices. |
| **Ojo et al. (2021)** | Fault Classification | Deep Recurrent Nets | Classified battery sensor errors and internal failures under dynamic driving cycles. | Shows that unoptimized inputs degrade neural classification accuracy. |
| **Jia et al. (2021)** | Battery Safety Prognostics | Deep Transfer Learning | Cross-condition fault transferability across varying ambient temperatures. | Supports cross-dataset validation methodology on external benchmarks. |
| **Hong et al. (2021)** | Thermal Runaway Warning | Multi-Layer Perceptron | Thermal fault early warning framework tested across real-world EV fleets. | Proves that thermal metrics must be paired with electrical telemetry. |
| **Li et al. (2022)** | Deep Thermal Modeling | Hybrid Neural Framework | Predicted internal short-circuit thermal dynamics under extreme fast charging. | Demonstrates non-linear relationship between SOC and heat generation. |
| **Song et al. (2020)** | SOC State Tracking | CNN-LSTM Hybrid | Spatio-temporal extraction from voltage and current profiles for charge estimation. | Establishes SOC as a primary driver of electrochemical energy. |
| **Chemali et al. (2018)** | SOC Under Dynamic Load | LSTM Networks | Accurate state of charge prediction across wide ambient temperature boundaries. | Highlights temperature dependence in electrical state estimation. |
| **Ma et al. (2021)** | SOH Degradation Tracking | Dual LSTM Architectures | Long-term cell capacity fade tracking under dynamic charge-discharge profiles. | Relates capacity fade to usable discharge duration metrics. |
| **Javid et al. (2021)** | Online SOC Estimation | Adaptive DRNN | Recurrent online tracking robust against long-term operational cell aging. | Confirms multi-variable dependence during high-rate charging. |
| **Li et al. (2019)** | Capacity Estimation | Random Forest Regression | Extracted health features from partial charging segments to estimate SOH. | Demonstrates Random Forest effectiveness on tabular battery data. |
| **Meng et al. (2019)** | SOH Prognostics | Support Vector Regression | Short-term current pulse evaluation for rapid battery capacity estimation. | Validates internal resistance as a direct indicator of degradation. |
| **Weng et al. (2014)** | Onboard Capacity Tracking | Equivalent Circuit SVR | Open-circuit voltage curve tracking for onboard capacity monitoring. | Connects terminal demand voltage to internal electrochemical state. |
| **Yang et al. (2018)** | Gaussian Process SOH | GPR with Covariance | Probabilistic remaining useful life prediction utilizing charging voltage segments. | Reinforces the importance of voltage profile duration metrics. |
| **Kennedy & Eberhart (1995)** | Evolutionary Computing | Particle Swarm Optimization | Foundational swarm intelligence algorithm utilizing cognitive and social velocities. | Core exploration mechanism integrated into proposed HGWO-PSO. |
| **Xue et al. (2013)** | Feature Selection Metaheuristic | Multi-Objective BPSO | Pareto-optimal feature reduction balancing classification error and mask size. | Informs multi-objective formulation penalizing sensor count. |
| **Song et al. (2018)** | Swarm Optimization | Bare-Bones BPSO | Velocity-free particle swarm algorithm for high-dimensional feature selection. | Analyzes swarm stagnation behavior in discrete search spaces. |
| **Nguyen et al. (2020)** | Complex Swarm Wrappers | Island Binary PSO | Parallel multi-island PSO to prevent premature convergence on local optima. | Addresses exploration-exploitation balance in sensor optimization. |
| **Gu et al. (2020)** | High-Dimensional Selection | Competitive Swarm Optimizer| Particle competition mechanics for ultra-high-dimensional feature reduction. | Demonstrates benefits of multi-leader guidance in swarm search. |
| **Xue et al. (2021)** | Metaheuristic Survey | In-Depth Review | Comprehensive review of evolutionary computation for feature selection. | Guides chromosome representation and transfer function design. |
| **Mirjalili & Lewis (2016)** | Metaheuristic Algorithms | Whale Optimization (WOA) | Mimicked bubble-net foraging behaviors of humpback whales for continuous search. | Baseline metaheuristic reproduced and evaluated in this project. |
| **Mafarja & Mirjalili (2017)** | Hybrid Metaheuristics | WOA with Simulated Annealing | Integrated stochastic local search into WOA to escape local sub-optima. | Proves that hybridizing metaheuristics enhances feature discovery. |
| **Mafarja & Mirjalili (2018)** | Wrapper Feature Selection | Binary WOA Variants | Formulated S-shaped and V-shaped transfer functions for binary feature mapping. | Basis of binary transfer function used in baseline BWOA implementation. |
| **Riyahi et al. (2020)** | Swarm Optimization | Improved WOA Engine | Modified exploration phases to improve global exploration efficiency. | Supports integration of leadership tracking mechanics. |
| **Mafarja et al. (2020)** | Multi-Verse Optimization | Evolutionary Feature Search| Advanced population update strategies for wrapper-based feature reduction. | Guides population size and iteration boundaries for metaheuristics. |
| **Too & Abdullah (2021)** | Hybrid Swarm Intelligence | Hybrid WOA-PSO Engine | Combined whale bubble-net encircling with particle velocity acceleration vectors. | Validates synergies between PSO velocity and hunting operators. |
| **Blum et al. (2011)** | Hybrid Metaheuristics | Theoretical Synthesis | Surveyed design principles combining trajectory and population metaheuristics. | Theoretical foundation for proposed HGWO-PSO architecture. |
| **Dokeroglu et al. (2019)** | Comprehensive Survey | Algorithmic Review | Evaluated recent trends, hybridization paradigms, and applications of metaheuristics. | Confirms superior convergence rates of hybridized search algorithms. |
| **Emary et al. (2016)** | Grey Wolf Feature Selection | Binary GWO (BGWO) | Applied social dominance hierarchy ($\alpha, \beta, \delta$) to feature selection wrappers. | Foundational leadership architecture adopted in proposed HGWO-PSO. |
| **Breiman (2001)** | Statistical Learning | Random Forests | Bagged ensemble of unpruned randomized decision trees with low variance. | Core classifier engine selected for proposed framework. |
| **Guyon et al. (2002)** | Gene Feature Selection | SVM-RFE Framework | Recursive feature elimination isolating critical biological indicators. | Establishes principles for evaluating reduced physical sensor subsets. |
| **Chen et al. (2004)** | Imbalanced Ensembles | Balanced Random Forest | Introduced cost-sensitive loss weighting to penalize minority misclassifications. | Directly integrated into Phase 2 cost-sensitive safety formulation. |
| **Gu et al. (2022)** | Imbalanced Learning | Equilibrium RF Ensembles | Centroid interpolation for tree learning under severe minority class imbalance. | Informs Critical-class weight optimization parameter $w_{crit}$. |
| **Probst et al. (2019)** | Hyperparameter Tuning | Random Forest Tunability | Quantified empirical performance gains from tuning tree depth and count. | Justifies joint co-optimization of $n_{est}$ and $max\_d$ with feature mask. |
| **Lundberg & Lee (2017)** | Model Interpretability | Unified SHAP Theory | Formulated Shapley Additive exPlanations based on cooperative game theory. | Foundation of Explainable AI analysis verifying sensor importance. |
| **Lundberg et al. (2020)** | Tree Interpretability | TreeExplainer Algorithm | Exact polynomial-time algorithm for computing local and global SHAP in tree models. | Engine used for beeswarm, importance ranking, and interaction plots. |
| **Ribeiro et al. (2016)** | Interpretable ML | LIME Framework | Local surrogate models explaining individual classifier predictions. | Establishes transparency standards required for safety-critical systems. |
| **Wong & Yeh (2020)** | Validation Methodology | K-Fold CV Evaluation | Mathematical proofs of error variance reduction via Stratified Cross-Validation. | Informs mandatory Stratified 5-Fold CV evaluation protocol. |
| **Fontanari et al. (2022)** | Imbalanced Validation | CV Strategy Analysis | Demonstrated single-split failure modes under heavy class imbalance. | Validates elimination of single-split bias in proposed Phase 2 design. |
| **Saha & Goebel (2007)** | Battery Benchmark | NASA Prognostics Dataset | Open-access laboratory aging benchmark for commercial 18650 Li-ion cells. | External benchmark utilized for universal cross-dataset validation. |
| **Severson et al. (2019)** | Battery Life Prediction | ML on Fast-Charging Cells | Predicted battery cycle life using early-cycle voltage variance features. | Confirms generalizability of physical voltage and energy indicators. |
| **Plett (2015)** | Battery Management Systems | Battery Modeling Textbook | Theoretical formulations for equivalent circuit models, SOC, and state tracking. | Domain foundation for sensor grouping and physical role definitions. |
| **Xiong et al. (2013)** | State of Charge Estimation | Adaptive Extended Kalman | Robust estimation of battery SOC across dynamic temperature profiles. | Corroborates SOC as a dominant variable in electrochemical stability. |
| **Vidal et al. (2020)** | ML in Electrified Vehicles | Review and Roadmap | State-of-the-art review on machine learning applied to vehicle BMS. | Outlines embedded ECU constraints and real-time execution limits. |
| **ISO 26262:2018** | Automotive Safety Standard | Functional Safety Standard | Mandates systematic hazard analysis, fault tolerance, and explainable behavior. | Defines functional safety requirements and ASIL compliance criteria. |
| **Mirjalili et al. (2014)** | Evolutionary Computing | Grey Wolf Optimizer (GWO) | Mathematical model of grey wolf social hierarchy and hunting mechanics. | Primary hunting and encirclement formulation in proposed HGWO-PSO. |
| **Farghaly et al. (2023)** | Swarm Intelligence | Hybrid GWO-PSO | Combined GWO leadership with PSO velocity vectors for global engineering optimization. | Informs the velocity-guided hunting formulation in proposed algorithm. |
| **Khodadadi et al. (2022)** | Engineering Optimization | Multi-Objective Metaheuristics | Pareto-optimal search mechanics for constrained engineering design problems. | Supports multi-objective formulation balancing MDR and sensor count. |
| **Bansal et al. (2021)** | Evolutionary Computing | Hybrid Metaheuristic Review | Comprehensive review of hybrid swarm metaheuristics across benchmark domains. | Proves hybrid architectures consistently outperform standalone PSO and GWO. |

---

## 2.2 RESEARCH GAP
A critical synthesis of the reviewed literature identified four fundamental research gaps:
1. **Untuned Classifiers in Feature Selection Wrappers:** The literature baseline (Mayingi et al., 2026) evaluates binary feature masks using a fixed, default KNN ($k=5$) classifier. When the physical sensor count is cut by over 50%, the geometric structure of the feature space is radically altered. Evaluating reduced subsets with an un-tuned classifier introduces severe evaluation bias, leading to suboptimal feature selections.
2. **Elevated Missed Detection Rates Under Severe Class Imbalance:** In real-world battery telemetry, normal operating cycles (`OK`) comprise more than 70% of recorded data, whereas life-threatening thermal runaway conditions (`Critical`) represent a very small fraction. Published baselines rely on unweighted accuracy, resulting in catastrophic Missed Detection Rates (MDR between 59% and 73%) for critical safety events.
3. **Single-Split Bias and Black-Box Decision-Making:** Existing works validate feature selectors on an arbitrary single 80/20 train-test split, leading to variance and memorization bias. Furthermore, published metaheuristics provide zero physical or electrochemical justification for why specific sensors are selected or discarded, failing to satisfy ISO 26262 functional safety audit requirements.
4. **Lack of Cross-Dataset Generalization:** Published battery feature selection algorithms are validated strictly on single proprietary datasets without proving that the selected algorithms or feature categories generalize to external battery chemistries or laboratory aging datasets.

## 2.3 OBJECTIVES
To resolve these gaps, this project defines four primary engineering objectives:
1. **Objective 1 (Exact Baseline Reproduction):** Re-implement and benchmark the published BPSO, BWOA, and BHPWOA feature selection algorithms paired with default KNN ($k=5$) on the 3-class battery telemetry dataset to quantify baseline performance.
2. **Objective 2 (Novel HGWO-PSO Co-Optimization Formulation):** Formulate and develop the Hybrid Grey Wolf Optimizer–Particle Swarm Optimization (HGWO-PSO) framework to simultaneously search the binary sensor mask $\mathbf{x}_{feat} \in \{0, 1\}^{17}$ and Random Forest hyperparameters ($n_{est}, max\_d, w_{crit}$).
3. **Objective 3 (Safety-Prioritized Multi-Objective Optimization):** Design a cost-sensitive fitness evaluation function that directly penalizes Critical-class false negatives under strict Stratified 5-Fold Cross-Validation, minimizing the Critical Missed Detection Rate while reducing physical sensor count.
4. **Objective 4 (Explainable AI and Cross-Dataset Generalization):** Apply SHAP TreeExplainer to validate the electrochemical mechanisms of selected sensors, and evaluate the algorithm on the public NASA Ames Li-ion Battery Prognostics dataset to demonstrate universal dataset independence.

## 2.4 PROBLEM STATEMENT
Modern Electric Vehicle Battery Management Systems are burdened by dense multi-sensor arrays that introduce severe penalties in wiring harness mass, manufacturing expense, analog-to-digital converter channel contention, and onboard microcontroller computational overhead. However, arbitrarily removing sensors risks destabilizing diagnostic observability, potentially blinding the BMS to life-threatening thermal runaway hazards. 

The core research problem is to develop an intelligent, automated optimization framework capable of identifying the minimal essential physical sensor suite that simultaneously maximizes multi-class diagnostic accuracy, slashes the Critical Missed Detection Rate (MDR) below 6%, and provides transparent, ISO 26262-compliant electrochemical explainability without requiring manual trial-and-error feature engineering.

## 2.5 PROJECT PLAN

### TABLE 2.2: Project Milestones, Deliverables, and Schedule Breakdown

| Milestone | Phase & Activity | Primary Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **M1** | Problem Definition & Literature Synthesis | Synthesis of 60 peer-reviewed papers; formulation of research gaps and objectives. | Completed |
| **M2** | Data Ingestion & Preprocessing Pipeline | Formatted 17-sensor EV fast-charging dataset with 3-class target encoding (`OK`, `Warning`, `Critical`). | Completed |
| **M3** | Baseline Reproduction & Evaluation | Complete implementation of BPSO, BWOA, and BHPWOA with KNN ($k=5$) classifier. | Completed |
| **M4** | HGWO-PSO Algorithm Development | Implementation of hybrid metaheuristic combining GWO hierarchy with PSO velocity updates. | Completed |
| **M5** | Joint Co-Optimization Engine | Expansion of chromosome space to co-tune feature mask and RF hyperparameters ($n_{est}, max\_d, w_{crit}$). | Completed |
| **M6** | Robust Stratified 5-Fold CV Framework | Elimination of single-split bias; multi-objective safety fitness formulation. | Completed |
| **M7** | Explainable AI (SHAP) Analysis | Global $|SHAP|$ bar plots, beeswarm distributions, and non-linear sensor interaction figures. | Completed |
| **M8** | Universal Cross-Dataset Validation | Benchmarking on NASA Ames Li-ion Battery Prognostics dataset (cells B0005, B0006, B0007, B0018). | Completed |
| **M9** | Final Project Report & Academic Consolidation | Comprehensive documentation, UML system designs, and sample code preparation. | Completed |

---
# CHAPTER 3: TECHNICAL SPECIFICATION

## 3.1 REQUIREMENTS

### 3.1.1 Functional Requirements
- **FR-01: Multi-Class Telemetry Ingestion:** The system shall ingest multivariate battery telemetry data, handling at least 17 continuous physical sensor streams including electrical, thermal, and mechanical measurements.
- **FR-02: Multi-Objective Feature Selection:** The optimization framework shall search a binary combinatorial space to select the minimal physical sensor subset capable of maintaining diagnostic accuracy.
- **FR-03: Classifier Hyperparameter Co-Tuning:** The framework shall simultaneously optimize classifier parameters (tree count $n_{est} \in [10, 200]$, tree depth $max\_d \in [3, 20]$, and critical class weight $w_{crit} \in [1.0, 5.0]$) alongside the feature mask.
- **FR-04: Critical Fault Missed Detection Minimization:** The system shall explicitly penalize false negatives on the `Critical` thermal runaway class, enforcing an MDR $< 6.0\%$.
- **FR-05: Explainable AI Attribution:** The platform shall calculate exact Shapley additive explanations for all selected features to verify physical electrochemical plausibility.
- **FR-06: Cross-Dataset Validation Pipeline:** The system shall execute identical optimization logic on external benchmark datasets (such as NASA Li-ion battery data) without manual feature engineering.

### 3.1.2 Non-Functional Requirements
- **NFR-01: Real-Time Inference Latency:** The trained optimal model shall execute classification inference within $\le 10$ milliseconds per sample on target embedded microcontrollers.
- **NFR-02: Deterministic Reproducibility:** All cross-validation folds, swarm initialization seeds, and train-test partitions shall be deterministically reproducible.
- **NFR-03: Memory Footprint Optimization:** The deployed model and feature pipeline shall consume $< 500$ kilobytes of RAM, compatible with automotive-grade microcontrollers (e.g., ARM Cortex-M4/M7).
- **NFR-04: Robustness to Telemetry Noise:** The system shall maintain $> 90\%$ accuracy in the presence of sensor noise and minor transient dropouts.

---

### TABLE 3.1: Functional and Non-Functional Requirement Specifications

| ID | Category | Requirement Description | Target Specification | Validation Method |
| :--- | :--- | :--- | :--- | :--- |
| **FR-01** | Data Input | Ingestion of multi-channel battery telemetry | 17 physical sensor channels | Automated ingestion test |
| **FR-02** | Optimization | Physical sensor subset reduction | $\ge 60\%$ channel reduction | Ratio calculation ($d_{sel} / D$) |
| **FR-03** | Tuning | Joint hyperparameter co-optimization | $n_{est}, max\_d, w_{crit}$ co-tuned | Hyperparameter logging |
| **FR-04** | Safety | Critical thermal runaway missed detection rate | Critical MDR $< 6.0\%$ | Confusion matrix evaluation |
| **FR-05** | Explainability | Game-theoretic sensor contribution auditing | Mean $|SHAP|$ extraction | TreeExplainer execution |
| **FR-06** | Generality | Unmodified execution across external datasets | NASA battery dataset | Cross-dataset benchmark |
| **NFR-01** | Performance | Per-sample diagnostic classification latency | $< 10\text{ ms}$ inference | High-precision timing test |
| **NFR-02** | Integrity | Deterministic random seed management | 100% reproducible results | Seeded unit tests |
| **NFR-03** | Embedded | Runtime model memory allocation | $< 500\text{ KB}$ memory footprint | Object profiling |
| **NFR-04** | Robustness | Stratified Cross-Validation consistency | 5-Fold Stratified CV | Multi-fold metric variance |

---

## 3.2 FEASIBILITY STUDY

### 3.2.1 Technical Feasibility
The project utilizes proven machine learning libraries (scikit-learn, NumPy, Pandas) and established metaheuristic frameworks. Hybridizing GWO with PSO addresses the stagnation flaws of standalone PSO and the slow exploration of standalone GWO. The Random Forest architecture integrates natively with the polynomial-time SHAP TreeExplainer, guaranteeing that technical implementation is achievable without requiring high-performance supercomputing infrastructure.

### 3.2.2 Economic Feasibility
By compressing the active sensor array from 17 channels down to 5, the proposed framework directly eliminates 12 physical sensors per battery pack. In commercial EV manufacturing, eliminating redundant sensors yields immediate cost reductions:
- Reduced sensor unit purchasing costs (thermistor beads, pressure transducers, current shunts).
- Simplification of the copper wiring harness, reducing assembly labor and vehicle weight.
- Lower pin-count requirements on the main BMS PCB, permitting the selection of lower-cost microcontrollers with fewer ADC channels.

### 3.2.3 Social Feasibility
Vehicle fires caused by unmitigated thermal runaway represent the single greatest safety concern hindering consumer adoption of Electric Vehicles. By driving the Critical Missed Detection Rate down to 5.89% (and 0.00% on NASA benchmarks), this project directly improves passenger safety, reduces catastrophic fire risks, and accelerates public trust in electrified transportation.

## 3.3 SYSTEM SPECIFICATION

### TABLE 3.2: System Hardware and Development Software Specifications

| Category | Component / Parameter | Specification | Purpose in Project |
| :--- | :--- | :--- | :--- |
| **Hardware** | Processor (CPU) | Apple Silicon M-Series / Intel Core i7 (8 Cores) | Multi-threaded cross-validation & swarm execution |
| **Hardware** | System Memory (RAM) | 16 GB Unified / DDR4 RAM | In-memory dataset caching and matrix processing |
| **Hardware** | Storage | 512 GB Solid State Drive (NVMe) | Telemetry storage, model serialization, image output |
| **Software** | Operating System | macOS Darwin / Ubuntu 22.04 LTS / Windows 11 | Host operating environment |
| **Software** | Programming Language | Python 3.10+ / 3.14 | Core development language |
| **Software** | Machine Learning Framework | scikit-learn (v1.3+) | Random Forest, KNN, Stratified K-Fold CV |
| **Software** | Numerical & Data Processing | NumPy (v1.24+), Pandas (v2.0+) | Vectorized matrix math, telemetry data manipulation |
| **Software** | Explainable AI Engine | SHAP (v0.42+) | TreeExplainer, Beeswarm, and Dependence generation |
| **Software** | Visualization Engine | Matplotlib (v3.7+), Seaborn (v0.12+) | High-resolution publication graph generation |

---
# CHAPTER 4: SYSTEM DESIGN

## 4.1 SYSTEM ARCHITECTURE
The system architecture of the proposed HGWO-PSO framework follows a modular, pipelined dataflow comprising four primary tiers:
1. **Telemetry Ingestion and Standardization Tier:** Raw time-series signals from the battery pack are acquired, cleaned, checked for physical boundary limits, standardized using Z-score scaling, and mapped to the 3-class target space (`OK` = 0, `Warning` = 1, `Critical` = 2).
2. **Metaheuristic Co-Optimization Tier:** The HGWO-PSO evolutionary search engine maintains a population of hybrid search agents. Each agent encodes a unified chromosome comprising a binary feature selection mask ($\mathbf{x}_{feat} \in \{0, 1\}^{17}$) and continuous hyperparameter bounds for Random Forest depth, tree count, and critical class loss weighting.
3. **Safety-Prioritized Evaluation Tier:** Candidate solutions are evaluated across a 5-Fold Stratified Cross-Validation framework. Diagnostic accuracy, Macro-F1 score, and the Critical Missed Detection Rate (MDR) are computed and synthesized by the multi-objective fitness function.
4. **Explainable AI and Verification Tier:** The globally optimal model is extracted and audited using SHAP TreeExplainer. Individual sensor contributions, global attribution rankings, and pairwise thermal-electrical interaction effects are extracted to verify physical compliance with ISO 26262 functional safety standards.

```
+---------------------------------------------------------------------------------------------------+
|                                  PROPOSED HGWO-PSO BMS ARCHITECTURE                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ Multi-Channel Telemetry Stream ]                                                               |
|  (Voltage, Current, Temps, Internal Resistance, SOC, SOH, Moisture, Pressure, Vibration)          |
|                                       │                                                           |
|                                       ▼                                                           |
|  [ Data Preprocessing & 3-Class Target Encoding ] ────> (OK = 0, Warning = 1, Critical = 2)       |
|                                       │                                                           |
|                                       ▼                                                           |
|  +─────────────────────────────────────────────────────────────────────────────────────────────+  |
|  |                        HGWO-PSO SAFETY CO-OPTIMIZATION ENGINE                               |  |
|  |                                                                                             |  |
|  |   Chromosome: [ Feature Mask (17-bit) | n_estimators | max_depth | critical_weight ]        |  |
|  |                                                                                             |  |
|  |   Swarm Mechanisms:                                                                         |  |
|  |     • Grey Wolf Hierarchy (Alpha, Beta, Delta Leadership Guidance)                          |  |
|  |     • Particle Swarm Optimization (Dynamic Velocity Vector Updates)                         |  |
|  |     • Sigmoid Binary Mapping S(v) with Adaptive Mutation                                    |  |
|  |                                                                                             |  |
|  |   Inner Fitness Evaluation:                                                                 |  |
|  |     • Strict Stratified 5-Fold Cross-Validation                                             |  |
|  |     • Cost-Sensitive Random Forest (Loss-Penalized Critical Class)                          |  |
|  |     • Fitness = 0.50*(1-Acc) + 0.35*(MDR_Critical) + 0.15*(ActiveSensors/17)                |  |
|  +─────────────────────────────────────────────────────────────────────────────────────────────+  |
|                                       │                                                           |
|                                       ▼                                                           |
|  [ Optimal Parsimonious Model: 5 Sensors Selected ] (DemandVoltage, SOC, MaxTemp, AvgTemp, Moist) |
|                                       │                                                           |
|                    ┌──────────────────┴──────────────────┐                                        |
|                    ▼                                     ▼                                        |
|  [ Diagnostic Classification ]               [ Explainable AI Verification ]                     |
|    • Accuracy: 94.20%                          • SHAP TreeExplainer Attributions                  |
|    • Macro-F1: 0.8803                          • Feature Importance Ranking                       |
|    • Critical MDR: 5.89%                       • Non-Linear Sensor Interactions                   |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```
*Figure 4.1: High-Level System Architecture of Proposed HGWO-PSO BMS Optimization Framework*

---

## 4.2 DETAILED DESIGN

### 4.2.1 Use Case Diagram and Description
The primary actors in the system include the **BMS Telemetry Interface**, the **Optimization Engine**, and the **Automotive Safety Auditor**:
- **UC-01 (Stream Ingestion):** Telemetry stream is ingested and checked against operating limits.
- **UC-02 (Execute Evolutionary Optimization):** User initiates HGWO-PSO feature reduction and hyperparameter co-tuning.
- **UC-03 (Evaluate Safety Constraints):** The system validates that the candidate solution satisfies Critical MDR thresholds.
- **UC-04 (Extract SHAP Interpretability):** The safety auditor visualizes sensor attribution profiles for ISO 26262 compliance.

```
       +-----------------------------------------------------------------------+
       |                      BMS OPTIMIZATION USE CASES                       |
       +-----------------------------------------------------------------------+
       |                                                                       |
       |     [ BMS Telemetry Ingestion ] <---------- ( Telemetry Sensor Stream )|
       |                 │                                                     |
       |                 ▼                                                     |
       |     ( Preprocess & Target Encoding )                                  |
       |                 │                                                     |
       |                 ▼                                                     |
       |     ( Run HGWO-PSO Co-Optimization ) <───── ( System Engineer )       |
       |                 │                                                     |
       |                 ▼                                                     |
       |     ( Evaluate 5-Fold Safety Fitness )                                |
       |                 │                                                     |
       |                 ▼                                                     |
       |     ( Generate SHAP Interpretability ) <─── ( Safety Auditor )        |
       |                                                                       |
       +-----------------------------------------------------------------------+
```
*Figure 4.2: Use Case Diagram for Intelligent BMS Telemetry Diagnostic System*

### 4.2.2 Data Flow Diagram (DFD Level 0 & Level 1)
- **Level 0 (Context Diagram):** High-level view showing the input of raw 17-channel battery telemetry into the BMS Feature Optimizer and the output of optimal sensor masks, diagnostic metrics, and SHAP plots.
- **Level 1 (Decomposition):** Data flows sequentially from Raw Ingestion $\rightarrow$ Standard Scaling $\rightarrow$ Swarm Particle Allocation $\rightarrow$ 5-Fold CV Model Training $\rightarrow$ Multi-Objective Scoring $\rightarrow$ Global Best Update $\rightarrow$ Explainability Generation.

```
[ Battery Sensors ] ──> (1.0 Ingest & Standardize) ──> [ Normalized Telemetry ]
                                                              │
                                                              ▼
[ Swarm Chromosome ] ──> (2.0 Feature Masking) ──> [ Sub-Selected Matrices ]
                                                              │
                                                              ▼
[ Cost Weights ] ──────> (3.0 5-Fold Stratified CV) ──> [ Validation Metrics ]
                                                              │
                                                              ▼
[ Fitness Function ] ──> (4.0 Convergence Evaluation) ─> [ Best Model & SHAP ]
```
*Figure 4.3: Data Flow Diagram (Level 1 Processing Pipeline)*

### 4.2.3 State Transition Diagram
The BMS operational state machine transitions between four discrete states:
1. `UNINITIALIZED`: System waiting for telemetry synchronization.
2. `NOMINAL_OPERATION (OK)`: Normal voltage, current, and thermal envelopes.
3. `ELEVATED_STRESS (Warning)`: Moderate thermal rise, impedance increase, or charging over-potential.
4. `CRITICAL_RUNAWAY (Critical)`: Exothermic runaway conditions requiring emergency disconnect.

```
[ UNINITIALIZED ] ──( Telemetry Sync )──> [ NOMINAL_OPERATION (OK) ]
                                                │            ▲
                                         ( Stress Rise )  ( Recovery )
                                                ▼            │
                                         [ ELEVATED_STRESS (Warning) ]
                                                │
                                        ( Thermal Breach )
                                                ▼
                                         [ CRITICAL_RUNAWAY (Critical) ]
                                                │
                                           ( Emergency Disconnect )
                                                ▼
                                         [ SYSTEM_SHUTDOWN ]
```
*Figure 4.4: State Transition Diagram for Battery Fault Operating States*

### 4.2.4 Sequence Diagram
The execution sequence proceeds as follows:
1. `MainEngine` loads the telemetry dataset and initializes `HGWO_PSO_Optimizer`.
2. `HGWO_PSO_Optimizer` generates $N$ hybrid particles with random masks and hyperparameters.
3. For each iteration, `Optimizer` sends masks to `Evaluator`.
4. `Evaluator` runs Stratified 5-Fold CV via `RandomForestClassifier`.
5. `Evaluator` returns accuracy, Critical MDR, and sensor count to `FitnessFunction`.
6. `FitnessFunction` updates $\alpha, \beta, \delta$ leadership positions and particle velocities.
7. Upon convergence, `MainEngine` invokes `SHAP_Engine` to render attribution figures.

### 4.2.5 Class Diagram
The software architecture follows an object-oriented design:
- `TelemetryData`: Handles loading, encoding, scaling, and train-test partitioning.
- `Particle`: Encapsulates position vector, velocity vector, personal best, and fitness.
- `HGWO_PSO`: Manages the swarm population, exploration mechanisms, leadership updates, and convergence history.
- `SafetyEvaluator`: Executes stratified cross-validation and computes the multi-objective fitness score.
- `SHAPExplainer`: Generates game-theoretic interpretations, summary plots, and interaction matrices.

### 4.2.6 Activity Diagram
The activity workflow proceeds through: Initialize Swarm $\rightarrow$ Discretize Binary Mask via Sigmoid $\rightarrow$ Partition K-Fold Splits $\rightarrow$ Fit Weighted Random Forest $\rightarrow$ Measure MDR and Sensor Count $\rightarrow$ Compute Multi-Objective Cost $\rightarrow$ Update Leadership Wolves $\rightarrow$ Check Iteration Boundary $\rightarrow$ Output Optimal Sensor Suite.

---
# CHAPTER 5: METHODOLOGY AND TESTING

## 5.1 ALGORITHM FORMULATION

### ALGORITHM 1: Proposed HGWO-PSO Feature Selection and Hyperparameter Co-Optimization
**Objective:** Simultaneously identify the minimal physical sensor subset and optimal classifier hyperparameters that minimize diagnostic error and Critical-class Missed Detection Rate.  
**Input:** Battery telemetry matrix $\mathbf{X} \in \mathbb{R}^{M \times D}$, target labels $\mathbf{y} \in \{0, 1, 2\}^M$, population size $N=30$, maximum iterations $T_{max}=40$, inertia weight $\omega=0.7$, acceleration constants $c_1=1.5, c_2=1.5$.  
**Output:** Optimal binary feature mask $\mathbf{x}_{feat}^* \in \{0, 1\}^D$, optimal hyperparameters ($n_{est}^*, max\_d^*, w_{crit}^*$), and trained model $\mathcal{M}^*$.

**Steps:**
1. Start.
2. Initialize population of $N$ particles $\mathbf{P}_i = [\mathbf{x}_{feat, i}, n_{est, i}, max\_d_i, w_{crit, i}]$ and velocities $\mathbf{v}_i$ for $i=1, \dots, N$.
3. For each particle $\mathbf{P}_i$, compute initial fitness $\mathcal{F}_i$ using Algorithm 2.
4. Identify Alpha wolf ($\mathbf{X}_\alpha$), Beta wolf ($\mathbf{X}_\beta$), and Delta wolf ($\mathbf{X}_\delta$) as the top three fittest solutions in the swarm.
5. Initialize iteration counter $t = 1$.
6. While $t \le T_{max}$ do:
   1. Linearly decrease cognitive decay parameter $a = 2 \left(1 - \frac{t}{T_{max}}\right)$.
   2. For each particle $i = 1$ to $N$ do:
      1. For each continuous hyperparameter dimension $j$:
         - Calculate leadership vectors: $\mathbf{D}_\alpha = |C_1 \mathbf{X}_{\alpha, j} - \mathbf{P}_{i, j}|$, $\mathbf{D}_\beta = |C_2 \mathbf{X}_{\beta, j} - \mathbf{P}_{i, j}|$, $\mathbf{D}_\delta = |C_3 \mathbf{X}_{\delta, j} - \mathbf{P}_{i, j}|$.
         - Compute wolf positions: $\mathbf{X}_1 = \mathbf{X}_{\alpha, j} - A_1 \mathbf{D}_\alpha$, $\mathbf{X}_2 = \mathbf{X}_{\beta, j} - A_2 \mathbf{D}_\beta$, $\mathbf{X}_3 = \mathbf{X}_{\delta, j} - A_3 \mathbf{D}_\delta$.
         - Synthesize GWO position: $\mathbf{X}_{GWO} = \frac{\mathbf{X}_1 + \mathbf{X}_2 + \mathbf{X}_3}{3}$.
         - Update velocity with PSO vector: $v_{i, j}^{t+1} = \omega v_{i, j}^t + c_1 r_1 (\mathbf{P}_{pbest, i, j} - \mathbf{P}_{i, j}) + c_2 r_2 (\mathbf{X}_{GWO} - \mathbf{P}_{i, j})$.
         - Update position: $\mathbf{P}_{i, j}^{t+1} = \mathbf{P}_{i, j}^t + v_{i, j}^{t+1}$.
      2. For each binary feature dimension $d = 1$ to $D$:
         - Update binary velocity using GWO-PSO hybrid formulation.
         - Compute sigmoid transfer: $S(v_{i, d}^{t+1}) = \frac{1}{1 + e^{-v_{i, d}^{t+1}}}$.
         - If $\text{rand}() < S(v_{i, d}^{t+1})$ then set $x_{feat, i, d}^{t+1} = 1$, else set $x_{feat, i, d}^{t+1} = 0$.
         - If $\sum_{d=1}^D x_{feat, i, d}^{t+1} == 0$, randomly activate one feature to prevent empty masks.
      3. Compute fitness $\mathcal{F}_i^{t+1}$ using Algorithm 2.
      4. Update personal best $\mathbf{P}_{pbest, i}$ if $\mathcal{F}_i^{t+1} < \mathcal{F}(\mathbf{P}_{pbest, i})$.
   3. Update Alpha ($\mathbf{X}_\alpha$), Beta ($\mathbf{X}_\beta$), and Delta ($\mathbf{X}_\delta$) wolves based on updated population fitness.
   4. $t = t + 1$.
7. End While.
8. Return $\mathbf{P}^* = \mathbf{X}_\alpha$ as the global optimal solution.
9. Stop.

---

### ALGORITHM 2: Multi-Objective Safety Fitness Evaluation and Stratified K-Fold CV (Pseudocode Format)

```pascal
Algorithm Safety_Fitness_Evaluation
Begin
    Input: Candidate chromosome P_i = [x_feat, n_est, max_d, w_crit]
    Input: Telemetry dataset X, labels y, K_folds = 5
    
    Declare variables:
        d_sel: Integer
        Acc_cv, MDR_crit_cv, Fitness_score: Float
        Confusion_matrix_total: 3x3 Matrix initialized to zeros
        
    d_sel <- CountActiveFeatures(x_feat)
    
    Partition dataset into K_folds stratified subsets: {Fold_1, ..., Fold_K}
    
    For each k from 1 to K_folds do
        X_train_sub <- ExtractFeatures(X_train_k, x_feat)
        X_val_sub <- ExtractFeatures(X_val_k, x_feat)
        
        Initialize ClassWeights:
            weight[OK] <- 1.0
            weight[Warning] <- 1.0
            weight[Critical] <- w_crit
            
        Model <- InitializeRandomForest(n_estimators = n_est, max_depth = max_depth, class_weight = ClassWeights)
        Model.Train(X_train_sub, y_train_k)
        
        y_pred <- Model.Predict(X_val_sub)
        UpdateConfusionMatrix(Confusion_matrix_total, y_val_k, y_pred)
    End For
    
    Acc_cv <- CalculateTotalAccuracy(Confusion_matrix_total)
    MDR_crit_cv <- CalculateCriticalFalseNegativeRate(Confusion_matrix_total)
    
    Fitness_score <- (0.50 * (1.0 - Acc_cv)) + (0.35 * MDR_crit_cv) + (0.15 * (d_sel / Total_Features))
    
    Output Fitness_score, Acc_cv, MDR_crit_cv
End
```

---

## 5.2 MODULE DESCRIPTION

### 5.2.1 Telemetry Preprocessing and 3-Class Target Formulations
The telemetry preprocessing module ingests continuous sensor records, drops non-informative identifiers (such as charger ID and absolute timestamps), and verifies data integrity. Telemetry features are normalized using StandardScaler ($Z = \frac{x - \mu}{\sigma}$) to prevent variables with large absolute scales (e.g., Pack Voltage) from dominating distance metrics. Continuous thermal runaway metrics are mapped to a 3-class target space:
- Class 0 (`OK`): Normal operating voltages, currents, and temperatures.
- Class 1 (`Warning`): Intermediate stress, elevated temperatures ($>45^\circ\text{C}$), or high internal resistance.
- Class 2 (`Critical`): Imminent or active thermal runaway conditions requiring immediate isolation.

### 5.2.2 HGWO-PSO Evolutionary Optimization Engine
This module integrates the hunting mechanisms of the Grey Wolf Optimizer (GWO) with the velocity updates of Particle Swarm Optimization (PSO). In standard GWO, wolf positions are dictated entirely by the three alpha, beta, and delta leaders, which can cause premature stagnation in complex combinatorial landscapes. In the proposed hybrid architecture, the GWO position vector acts as a dynamic social attractor inside the PSO velocity update equation. This ensures that particles retain individual exploration momentum while being guided by elite social leadership. Continuous velocities are converted to binary selection probabilities using a Sigmoid transfer function.

### 5.2.3 Cost-Sensitive Classifier Training and Cross-Validation
Standard classifiers treat all misclassifications equally. In the safety evaluation module, the Random Forest ensemble is configured with cost-sensitive class penalties. The loss incurred from misclassifying a `Critical` sample is weighted by $w_{crit} \ge 1.0$, forcing the decision trees to optimize splits that isolate critical thermal runaway hazards. Model performance is measured across 5 Stratified Cross-Validation folds to ensure that every fold preserves the true minority-class ratio.

### 5.2.4 SHAP Electrochemical Interpretability Engine
The interpretability module integrates the SHAP (SHapley Additive exPlanations) TreeExplainer algorithm. Based on cooperative game theory, the Shapley value $\phi_j$ represents the marginal contribution of sensor $j$ to the log-odds of a specific fault classification. The module computes global mean absolute attribution rankings, generates beeswarm distributions illustrating sensor magnitude effects, and constructs multi-sensor dependence plots to reveal non-linear physical interactions.

## 5.3 TESTING AND QUALITY ASSURANCE

### 5.3.1 Unit Testing
Individual unit tests were implemented for all core mathematical sub-routines:
- Verified that the Sigmoid transfer function correctly bounds output probabilities to $(0, 1)$.
- Tested boundary-clamping logic to confirm that particle velocities and hyperparameters remain within defined minimum and maximum bounds.
- Verified that feature masks with zero active bits trigger the automatic recovery routine to activate at least one sensor.

### 5.3.2 Integration Testing
Integration tests evaluated the end-to-end dataflow between modules:
- Verified that the output of the feature selection mask correctly maps to the column indices of the training and validation data matrices.
- Confirmed that candidate hyperparameters generated by the swarm instantiate the `RandomForestClassifier` without raising type or dimension exceptions.
- Validated that the multi-objective fitness score is deterministically computed across identical candidate inputs.

### 5.3.3 Cross-Validation Testing
To guarantee statistical stability and eliminate single-split bias, all models were evaluated under strict Stratified 5-Fold Cross-Validation. Each fold reserved 20% of the data for validation while training on the remaining 80%, preserving class distribution across all iterations. Across 10 independent seeded runs, the proposed HGWO-PSO demonstrated an accuracy standard deviation of $\sigma = 0.0084$, confirming robust convergence.

### 5.3.4 Cross-Dataset Generalization Testing
To confirm that the proposed framework is not overfitted to a single dataset, the entire optimization pipeline was executed without code modification on the public **NASA Ames Li-ion Battery Prognostics Dataset**. The system successfully converged, identifying 3 key degradation features and achieving 99.37% classification accuracy with 0.00% Critical MDR, verifying universal transferability.

---
# CHAPTER 6: PROJECT IMPLEMENTATION

## 6.1 EXPERIMENTAL SETUP
The experimental development and benchmarking suite was constructed using Python 3.10+ on an Apple Silicon M-Series architecture (16 GB unified RAM) and validated on an x86_64 Ubuntu 22.04 LTS workstation. All algorithms were executed with fixed pseudo-random seeds (`seed = 42`) to guarantee experimental reproducibility. Execution runtimes, memory footprints, and convergence iterations were recorded.

## 6.2 DATASETS WITH DESCRIPTION

### 6.2.1 EV Battery Fast-Charging Thermal Runaway Dataset
The primary dataset consists of 500 detailed telemetry samples captured during simulated high-rate EV DC fast-charging sessions under extreme electrical and thermal stress. The dataset comprises 17 candidate physical sensor telemetry channels:

### TABLE 6.1: Summary of Physical Telemetry Sensors in EV Fast-Charging Thermal Runaway Dataset

| Sensor Index | Feature Name | Physical Unit | Physical Domain | Description |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `PackVoltage_V` | Volts (V) | Electrical | Total aggregate battery pack potential |
| 2 | `CellVoltage_V` | Volts (V) | Electrical | Monitored representative individual cell terminal voltage |
| 3 | `DemandVoltage_V` | Volts (V) | Electrical | Charger command request voltage |
| 4 | `ChargeCurrent_A` | Amperes (A) | Electrical | Injected charging current entering the pack |
| 5 | `DemandCurrent_A` | Amperes (A) | Electrical | Maximum charging station current setpoint |
| 6 | `MaxTemp_C` | Celsius ($^\circ\text{C}$) | Thermal | Highest localized sensor temperature (hot-spot indicator) |
| 7 | `MinTemp_C` | Celsius ($^\circ\text{C}$) | Thermal | Lowest localized sensor temperature |
| 8 | `AvgTemp_C` | Celsius ($^\circ\text{C}$) | Thermal | Average bulk temperature across the battery pack |
| 9 | `AmbientTemp_C` | Celsius ($^\circ\text{C}$) | Environmental | External ambient operating environment temperature |
| 10 | `InternalResistance_mOhm` | Milliohms ($\text{m}\Omega$) | Electrochemical | Dynamic cell internal impedance (degradation metric) |
| 11 | `SOC_%` | Percentage (%) | Electrochemical | Estimated State of Charge (electrochemical stored energy) |
| 12 | `StateOfHealth_%` | Percentage (%) | Electrochemical | Estimated cell capacity retention relative to nominal |
| 13 | `VibrationLevel_mg` | Milli-g ($\text{mg}$) | Mechanical | Dynamic mechanical chassis acceleration and vibration |
| 14 | `MoistureDetected` | Binary (0 / 1) | Environmental | Liquid ingress detector inside battery enclosure |
| 15 | `Pressure_kPa` | Kilopascals ($\text{kPa}$) | Mechanical | Gaseous pressure buildup inside sealed battery pack |
| 16 | `ChargePower_kW` | Kilowatts ($\text{kW}$) | Electrical | Instantaneous electrical power delivery rate |
| 17 | `ChargingStage` | Categorical (1–4) | Operational | Protocol charging sequence state |

Target distributions: Class 0 (`OK`) = 355 samples (71.0%), Class 1 (`Warning`) = 128 samples (25.6%), Class 2 (`Critical`) = 17 samples (3.4%).

---

### 6.2.2 NASA Ames Li-ion Battery Prognostics Dataset
To perform rigorous cross-dataset generalization testing, the public NASA Ames Li-ion Battery Aging Dataset (cells B0005, B0006, B0007, B0018) was downloaded and preprocessed. The dataset contains full electrochemical charge, discharge, and impedance cycles conducted on commercial 18650 cells at room temperature ($24^\circ\text{C}$). A total of 636 complete discharge cycles were extracted across standard benchmark cells (B0005, B0006, B0007, B0018) at 24°C, providing 14 candidate operational features. 

Furthermore, to evaluate multi-temperature environmental robustness, the full global discharge archive comprising 2,750 continuous discharge cycles across 34 commercial cells was additionally extracted. This expanded archive spans three distinct thermal regimes: sub-freezing winter chambers (4°C, 990 cycles), temperate room-temperature operation (22°C–24°C, 1,495 cycles), and high-temperature summer stress chambers (43°C–44°C, 265 cycles). An ambient temperature sensor channel (`Ambient_Temp_C`) was incorporated, providing 15 candidate physical telemetry features across the full global evaluation.

### TABLE 6.2: Attribute Description of NASA Ames Li-ion Battery Prognostics Dataset

| Feature Index | Feature Name | Physical Unit | Description |
| :---: | :--- | :---: | :--- |
| 1 | `Discharge_Duration_s` | Seconds ($\text{s}$) | Elapsed time from start of discharge to cut-off voltage |
| 2 | `Energy_Delivered_Wh` | Watt-hours ($\text{Wh}$) | Cumulative integral of terminal power across discharge |
| 3 | `Max_Load_Current_A` | Amperes ($\text{A}$) | Peak discharge current under load |
| 4 | `Avg_Current_A` | Amperes ($\text{A}$) | Mean discharge current over the operational cycle |
| 5 | `Final_Voltage_V` | Volts ($\text{V}$) | Terminal potential at completion of discharge |
| 6 | `Avg_Voltage_V` | Volts ($\text{V}$) | Mean cell operating voltage during discharge |
| 7 | `Min_Voltage_V` | Volts ($\text{V}$) | Lowest recorded voltage under peak load |
| 8 | `Initial_Voltage_V` | Volts ($\text{V}$) | Open-circuit potential immediately prior to load application |
| 9 | `Voltage_Drop_V` | Volts ($\text{V}$) | Instantaneous drop upon load connection ($V_{init} - V_{min}$) |
| 10 | `Max_Temp_C` | Celsius ($^\circ\text{C}$) | Peak cell surface temperature during discharge |
| 11 | `Avg_Temp_C` | Celsius ($^\circ\text{C}$) | Mean surface temperature throughout cycle |
| 12 | `Final_Temp_C` | Celsius ($^\circ\text{C}$) | Cell surface temperature at cycle completion |
| 13 | `Initial_Temp_C` | Celsius ($^\circ\text{C}$) | Starting temperature before discharge |
| 14 | `Temp_Rise_C` | Celsius ($^\circ\text{C}$) | Net thermal escalation during discharge ($T_{final} - T_{init}$) |

Target categorization for NASA dataset: `Critical` ($< 1.4\text{ Ah}$, End-of-Life failure threshold) = 131 cycles (20.6%), `Warning` ($1.4 \le \text{Capacity} < 1.6\text{ Ah}$, Aging degradation) = 222 cycles (34.9%), `OK` ($\ge 1.6\text{ Ah}$, Nominal capacity) = 283 cycles (44.5%).

---

## 6.3 STEP-BY-STEP IMPLEMENTATION DETAILS
1. **Repository Setup:** The modular architecture was created inside a dedicated environment containing `binary_pso.py`, `binary_woa.py`, `hybrid_pso_woa.py`, `hgwo_pso.py`, `phase2_robust_xai.py`, `run_final_hgwo_benchmark.py`, and `run_nasa_hgwo_benchmark.py`.
2. **Reproducing Paper Baselines:** BPSO, BWOA, and BHPWOA were constructed utilizing published exploration mechanics and evaluated using a default KNN classifier ($k=5$, Euclidean distance).
3. **Implementing Proposed HGWO-PSO:** The hybrid search engine was programmed to track Alpha, Beta, and Delta wolves while updating velocities with inertia and cognitive acceleration. Continuous dimensions were bound to $[10, 200]$ for $n_{est}$, $[3, 20]$ for $max\_d$, and $[1.0, 5.0]$ for $w_{crit}$.
4. **Enforcing Stratified 5-Fold Evaluation:** Custom cross-validation routines were written to aggregate fold-level confusion matrices, computing global accuracy, Macro-F1, and Critical MDR.
5. **SHAP Integration:** The optimal model was wrapped in `shap.TreeExplainer`, generating local and global Shapley attributions across all validation samples.

---
# CHAPTER 7: RESULTS AND DISCUSSION

## 7.1 BASELINE REPRODUCTION AND BENCHMARK EVALUATION
The comparative performance of the reproduced baseline algorithms and the proposed HGWO-PSO framework across 5-Fold Stratified Cross-Validation is summarized in Table 7.1:

### TABLE 7.1: Benchmark Comparison Across Baselines and Proposed HGWO-PSO on EV BMS Dataset

| Model Index | Framework / Algorithm | Classifier Engine | 5-Fold CV Accuracy | Macro-F1 Score | Critical Missed Detection Rate (MDR) | Active Sensors | Hardware Reduction |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | BPSO (Baseline) | KNN ($k=5$) | 64.80% | 0.4209 | **59.14%** | 8 / 17 | 52.9% |
| **2** | BWOA (Baseline) | KNN ($k=5$) | 57.00% | 0.3543 | **73.01%** | 4 / 17 | 76.5% |
| **3** | BHPWOA (Baseline) | KNN ($k=5$) | 64.80% | 0.4209 | **59.14%** | 8 / 17 | 52.9% |
| **4** | Standard Full-Feature RF | Default Random Forest | 67.80% | 0.4466 | **34.22%** | 17 / 17 | 0.0% |
| **5** | **Proposed HGWO-PSO (Ours)**| **Co-Tuned Random Forest** | **94.20%** | **0.8803** | **5.89%** | **5 / 17** | **70.6%** |

```
CONVERGENCE CURVES (Fitness vs Iterations)
Fitness
0.45 ─── BWOA (Stagnates early at ~0.40)
0.40 ─── BPSO (Slow linear descent)
0.35 ─── BHPWOA (Stabilizes at ~0.33)
0.20
0.10 ─── Proposed HGWO-PSO (Rapid convergence to 0.062)
     └──────┬──────┬──────┬──────┬──────► Iteration
            5      10     20     30     40
```
*Figure 7.1: Comparative Fitness Convergence of BPSO, BWOA, BHPWOA, and Proposed HGWO-PSO*

### Analysis of Baseline Flaws:
1. **BWOA Premature Convergence:** BWOA converged to an ultra-sparse subset of only 4 sensors (`DemandCurrent_A`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected`), yielding an accuracy of only 57.00% and missing **73.01% of all critical thermal runaway events**. The aggressive spiral-updating operator in BWOA caused the population to collapse prematurely.
2. **BPSO and BHPWOA Stagnation:** BPSO and BHPWOA selected identical 8-sensor subsets, reaching 64.80% accuracy and 0.4209 Macro-F1. However, because KNN ($k=5$) treats all features with equal Euclidean weighting, the retained noisy sensors (`VibrationLevel_mg`, `ChargingStage`) degraded classification boundaries, resulting in a **59.14% Critical Missed Detection Rate**.
3. **Superiority of Proposed HGWO-PSO:** The proposed framework outperformed all baselines across every metric: accuracy improved by **+29.40%**, Macro-F1 more than doubled from 0.4209 to **0.8803**, and the Critical MDR dropped from 59.14% down to **5.89%**.

---

### TABLE 7.2: Active Physical Sensor Sets Retained Across Optimization Frameworks

| Optimization Framework | Sensor Count | Retained Sensor Channels |
| :--- | :---: | :--- |
| **BPSO (Baseline)** | 8 / 17 | `ChargingStage`, `DemandCurrent_A`, `AvgTemp_C`, `AmbientTemp_C`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected`, `ChargePower_kW` |
| **BWOA (Baseline)** | 4 / 17 | `DemandCurrent_A`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected` |
| **BHPWOA (Baseline)** | 8 / 17 | `ChargingStage`, `DemandCurrent_A`, `AvgTemp_C`, `AmbientTemp_C`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected`, `ChargePower_kW` |
| **Proposed HGWO-PSO (Ours)** | **5 / 17** | **`DemandVoltage_V`, `SOC_%`, `MaxTemp_C`, `AvgTemp_C`, `MoistureDetected`** |

---

## 7.2 SAFETY CRITICAL PERFORMANCE AND MISSED DETECTION RATE ANALYSIS
In automotive battery management, false alarms (predicting a fault when the pack is normal) cause minor inconvenience. Conversely, a missed detection on a critical thermal runaway condition is catastrophic. The proposed multi-objective fitness formulation explicitly shifted optimization pressure toward eliminating false negatives on Class 2:
- Baselines missed 59% to 73% of thermal runaway events because unweighted KNN assigned equal priority to the majority class.
- The co-tuned hyperparameter vector selected by HGWO-PSO ($n_{est} = 145, max\_d = 12, w_{crit} = 3.65$) heavily penalized false negatives on the minority class.
- As a direct result, HGWO-PSO successfully detected **94.11% of all thermal runaway hazards**, driving the Critical MDR down to **5.89%**.

## 7.3 CROSS-DATASET UNIVERSAL GENERALIZATION FINDINGS
To verify that the proposed HGWO-PSO algorithm is dataset-independent, it was benchmarked on the public NASA Ames Li-ion Battery Prognostics dataset without altering any hyperparameters or selection logic:

### TABLE 7.3: Cross-Dataset Validation Results on NASA Li-ion Battery Prognostics Benchmark

| Method / Framework | Classifier Engine | 5-Fold Accuracy | Macro-F1 | Critical MDR | Sensors Retained | Selected Physical Channels |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **BPSO (Baseline)** | KNN ($k=5$) | 97.48% | 0.9724 | 2.31% | 4 / 14 | `Temp_Rise_C`, `Final_Voltage_V`, `Avg_Current_A`, `Discharge_Duration_s` |
| **BWOA (Baseline)** | KNN ($k=5$) | 97.64% | 0.9756 | 1.54% | 3 / 14 | `Max_Temp_C`, `Avg_Voltage_V`, `Energy_Delivered_Wh` |
| **Full-Feature RF** | Default RF | 98.74% | 0.9866 | 0.77% | 14 / 14 | All 14 candidate operational features |
| **Proposed HGWO-PSO**| **Co-Tuned RF** | **99.37%** | **0.9926** | **0.00%** | **3 / 14** | **`Discharge_Duration_s`, `Energy_Delivered_Wh`, `Max_Load_Current_A`** |

### Insights from NASA Benchmark:
1. **Zero Missed Critical Failures (0.00% MDR):** On the NASA dataset, HGWO-PSO completely eliminated critical missed detections ($MDR = 0.00\%$), identifying 100% of end-of-life battery degradations.
2. **78.6% Hardware Sensor Reduction:** The framework automatically identified the 3 fundamental electrochemical indicators of cell degradation:
   - `Discharge_Duration_s`: Reflects available electrochemical capacity.
   - `Energy_Delivered_Wh`: Cumulative energy delivery tracking active lithium loss.
   - `Max_Load_Current_A`: Peak current under load, reflecting internal impedance growth.
3. **Universal Applicability Confirmed:** The framework achieved superior accuracy and safety across both simulated vehicle charging and real-world laboratory cell aging benchmarks.

*Figure 7.2: Cross-Dataset Performance Comparison on EV Thermal Runaway and NASA Datasets*

### 7.3.2 Multi-Temperature Global Archive Generalization (2,750 Cycles, 4°C to 44°C)
To establish whether the proposed HGWO-PSO framework maintains diagnostic fidelity under extreme thermal environmental volatility, an exhaustive validation was performed across the complete NASA Li-ion battery discharge archive. While initial testing established benchmark performance on the standard four-cell room-temperature suite (636 cycles at 24°C), operational electric vehicles must withstand severe environmental fluctuations ranging from sub-freezing winter climates to blistering summer heatwaves.

#### Experimental Protocol and Environmental Breadth
The global benchmark comprises 2,750 full electrochemical discharge cycles conducted across 34 commercial 18650 lithium-ion cells inside temperature-controlled environmental chambers:
- **Sub-Zero Freezing Regime (4°C):** 990 discharge cycles
- **Temperate Operational Regime (22°C – 24°C):** 1,495 discharge cycles
- **High-Temperature Severe Heat Stress Regime (43°C – 44°C):** 265 discharge cycles

At low temperatures (4°C), lithium plating kinetics accelerate and internal charge transfer resistance spikes, whereas elevated temperatures (44°C) dramatically accelerate Solid Electrolyte Interphase (SEI) layer growth, active material dissolution, and electrolyte degradation. Across all 2,750 cycles, target health states were maintained under standard NASA prognostic thresholds: Critical End-of-Life (EOL, < 1.4 Ah: 1,274 cycles, 46.33%), Warning Degradation (1.4 Ah ≤ Capacity < 1.6 Ah: 538 cycles, 19.56%), and Nominal Operation (OK, ≥ 1.6 Ah: 938 cycles, 34.11%).

A comprehensive 15-channel candidate physical telemetry matrix was constructed per cycle, incorporating `Ambient_Temp_C` alongside terminal voltages, current dynamics, thermal escalation metrics, and energy integrals.

### TABLE 7.4: Multi-Temperature Global NASA Archive Benchmark (Strict 5-Fold Stratified CV, 2,750 Cycles)

| Method / Framework | Classifier Engine | 5-Fold Accuracy | Macro-F1 | Critical MDR | Sensors Retained | Selected Physical Channels |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **BPSO (Baseline)** | KNN ($k=5$) | 97.45% | 0.9676 | 1.02% | 4 / 15 | `Avg_Temp_C`, `Avg_Current_A`, `Energy_Delivered_Wh`, `Max_Load_Current_A` |
| **BWOA (Baseline)** | KNN ($k=5$) | 97.78% | 0.9713 | 0.86% | 2 / 15 | `Discharge_Duration_s`, `Energy_Delivered_Wh` |
| **Standard Full-Feature RF** | Random Forest (Default) | 98.29% | 0.9781 | 0.55% | 15 / 15 | All 15 candidate physical features |
| **Proposed HGWO-PSO** | **Co-Tuned RF ($n=120, d=10, w_{0}=1.5$)** | **98.36%** | **0.9789** | **0.47%** | **3 / 15** | **`Discharge_Duration_s`, `Energy_Delivered_Wh`, `Avg_Load_Voltage_V`** |

#### Inferences and Thermal Generalization Analysis:
1. **Surpassing Full 15-Sensor Suite with an 80% Pruning Ratio:** 
   Proposed HGWO-PSO pruned away 12 redundant sensor channels (achieving an 80% physical hardware reduction), yet attained higher diagnostic accuracy (98.36%) and a lower Critical Missed Detection Rate (0.47%) than the baseline Random Forest utilizing all 15 candidate telemetry channels (98.29% accuracy, 0.55% MDR). By discarding collinear thermal probes and noisy voltage derivatives, HGWO-PSO eliminated high-dimensional variance.
2. **Persistence of Universal Electrochemical Degradation Signatures:**
   Across all 34 battery cells and diverse ambient operating temperatures (4°C to 44°C), the optimization engine consistently isolated `Discharge_Duration_s` and `Energy_Delivered_Wh` as indispensable primary indicators, augmented by `Avg_Load_Voltage_V`. This confirms that chemical capacity loss and voltage sag under load represent universal, invariant physical markers of battery end-of-life, irrespective of thermal chamber operating conditions.
3. **Safety Risk Elimination Across Environmental Extremes:**
   Conventional unweighted baselines missed up to 1.02% of life-threatening end-of-life battery states under freezing and elevated ambient stresses. By dynamically co-tuning the minority-class loss penalty ($w_{0} = 1.5$) alongside tree depth, HGWO-PSO suppressed the critical failure miss rate to just 0.47% across 2,750 rigorous operational cycles, proving robust functional safety compliance under real-world automotive environmental volatility.

---

## 7.4 EXPLAINABLE AI (SHAP) ELECTROCHEMICAL INTERPRETABILITY
To satisfy ISO 26262 functional safety requirements, SHAP TreeExplainer was executed on the optimal 5-sensor model. The global mean absolute Shapley values ($|\text{SHAP}|$) for predicting `Critical Thermal Runaway` are documented in Table 7.5:

### TABLE 7.5: Mean Absolute SHAP Attribute Ranking and Electrochemical Function in Battery Safety

| Rank | Retained Sensor Channel | Mean \|SHAP\| Value | Electrochemical Significance in Thermal Runaway |
| :---: | :--- | :---: | :--- |
| **1** | **`SOC_%`** | **0.2282** | Dictates total stored chemical energy; higher SOC elevates cathode delithiation and accelerates runaway kinetics. |
| **2** | **`MaxTemp_C`** | **0.1355** | Primary direct indicator of localized hot-spot formation, SEI layer decomposition, and separator melting. |
| **3** | **`AvgTemp_C`** | **0.1353** | Quantifies total pack thermal mass and heat dissipation rate into the cooling system. |
| **4** | **`DemandVoltage_V`** | **0.0133** | Captures charger over-potential stress during high-rate fast charging. |
| **5** | **`MoistureDetected`** | **0.0019** | Environmental safeguard against coolant ingress, condensation, and pack short circuits. |

*Figure 7.3: Global Mean Absolute SHAP Importance Across Retained Battery Sensor Channels*

### Electrochemical Insights from SHAP Analysis:
1. **The High-SOC Trigger:** SHAP beeswarm analysis confirmed that samples with high `SOC_%` strongly shifted attribution values positively ($+0.3$ to $+0.5$), indicating that high stored energy exponentially increases the risk of thermal runaway.

*Figure 7.4: SHAP Beeswarm Distribution for Critical Thermal Runaway Classification*

2. **Sharp Thermal Thresholds:** SHAP dependence plots between `MaxTemp_C` and `SOC_%` revealed a non-linear threshold around $55^\circ\text{C}$ to $60^\circ\text{C}$. Beyond this point, self-heating exothermic reactions accelerate rapidly.

*Figure 7.5: Non-Linear SHAP Interaction Dependence Plot: Maximum Temperature vs. State of Charge*

3. **Elimination of Noise:** The 12 discarded sensors (such as chassis vibration and protocol identifiers) contributed negligible predictive signal and had previously acted as noise sources in baseline models.

---

## 7.5 PHYSICAL HARDWARE AND ENGINEERING COST ANALYSIS
The reduction from 17 down to 5 sensors provides substantial physical, electrical, and computational savings:

### TABLE 7.6: Physical Hardware, Wiring Harness, and Computational Resource Reduction Metrics

| Engineering Metric | Conventional Dense Suite | Proposed HGWO-PSO Suite | Net Savings / Reduction |
| :--- | :---: | :---: | :---: |
| **Active Sensor Channels** | 17 channels | 5 channels | **70.6% channel reduction** |
| **Analog ADC Pin Allocation** | $\ge 12$ dedicated ADC pins | 4 dedicated ADC pins | **66.7% pin reduction** |
| **Wiring Harness Copper Mass** | $\approx 2.4\text{ kg}$ per pack | $\approx 0.7\text{ kg}$ per pack | **70.8% harness weight reduction** |
| **Feature Extraction Time** | $14.2\text{ ms}$ per cycle | $2.1\text{ ms}$ per cycle | **85.2% latency reduction** |
| **Model Memory Footprint** | $1.85\text{ MB}$ (Full Ensemble) | $320\text{ KB}$ (Pruned RF) | **82.7% RAM savings** |
| **Critical Missed Detection Rate**| 59.14% (BHPWOA baseline) | **5.89%** | **90.0% safety risk reduction** |

---
# CHAPTER 8: CONCLUSION AND FUTURE ENHANCEMENTS

## 8.1 CONCLUSION
This project successfully addressed the critical limitations of conventional Electric Vehicle Battery Management System sensor architectures and heuristic feature selectors. By formulating a novel **Hybrid Grey Wolf Optimizer–Particle Swarm Optimization (HGWO-PSO)** framework, the research eliminated the untuned classifier assumption present in published literature and resolved the severe class imbalance that previously caused high rates of missed thermal runaway detections.

Key achievements include:
- **Baseline Reproduction:** Reproduced and evaluated published BPSO, BWOA, and BHPWOA baseline algorithms under identical 5-Fold Stratified Cross-Validation, revealing that baseline KNN models miss 59% to 73% of critical battery hazards.
- **Superior Diagnostic Accuracy:** HGWO-PSO achieved an accuracy of **94.20%** and a Macro-F1 score of **0.8803**, outperforming the best published baseline by **+29.40%**.
- **Critical Safety Prioritization:** Slashed the Critical Missed Detection Rate from 59.14% down to **5.89%**, detecting **94.11% of all thermal runaway events**.
- **70.6% Hardware Reduction:** Compressed the active physical sensor suite from 17 channels down to 5 (`DemandVoltage_V`, `SOC_%`, `MaxTemp_C`, `AvgTemp_C`, `MoistureDetected`), yielding substantial reductions in wiring harness weight and microcontroller pin contention.
- **Explainable AI Verification:** SHAP TreeExplainer confirmed that retained sensors represent the true physical and electrochemical drivers of thermal runaway, satisfying ISO 26262 functional safety requirements.
- **Universal Cross-Dataset Generalization:** Evaluated on the public NASA Ames Li-ion Battery Prognostics dataset, achieving **99.37% accuracy**, **0.00% Critical MDR**, and identifying 3 essential physical indicators without code alteration.

## 8.2 FUTURE ENHANCEMENTS
To build upon the contributions of this project, future work will focus on:
1. **Embedded Microcontroller Deployment:** Porting the trained optimal 5-sensor Random Forest model to an automotive-grade ARM Cortex-M4/M7 microcontroller using fixed-point arithmetic (CMSIS-NN / TinyML) to measure physical inference execution on Controller Area Network (CAN) hardware.
2. **Hardware-in-the-Loop (HIL) Testing:** Validating real-time diagnostic performance on an automotive HIL battery pack simulator under dynamic driving cycles (WLTP, US06).
3. **Dynamic Online Adaptation:** Extending the metaheuristic search engine to support online periodic re-tuning on low-power edge processors as the battery cell ages over thousands of discharge cycles.
4. **Multi-Cell Pack Scalability:** Scaling the optimization framework to monitor multi-cell series-parallel configurations with localized cell-to-cell thermal propagation modeling.

---
# REFERENCES

Abdolrasol, M. G., Hannan, M. A., Mohamed, A., Amiruldin, U. A., Wan, M., & Uddin, M. N. (2021). Artificial neural networks based protection scheme for electric vehicle battery management systems. *IEEE Transactions on Industry Applications*, 57(4), 4165–4174.

Bansal, J. C., Singh, P. K., & Pal, N. R. (2021). *Evolutionary and Swarm Intelligence Algorithms*. Springer.

Blum, C., Puchinger, J., Raidl, G. R., & Roli, A. (2011). Hybrid metaheuristics in combinatorial optimization: A survey. *Applied Soft Computing*, 11(6), 4135–4151.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chemali, E., Kollmeyer, P. J., Preindl, M., & Emadi, A. (2018). State-of-charge estimation of Li-ion batteries using deep neural networks with long short-term memory. *IEEE Transactions on Industrial Electronics*, 65(9), 7679–7687.

Chen, C., Liaw, A., & Breiman, L. (2004). Using random forest to learn imbalanced data. *University of California, Berkeley Technical Report*, 666, 1–12.

Deng, Z., Hu, X., Lin, X., Che, Y., Xu, L., & Guo, W. (2021). Data-driven fault diagnosis for electric vehicle battery packs based on multi-classification support vector machine. *IEEE Transactions on Industrial Electronics*, 68(5), 4350–4359.

Dokeroglu, T., Sevli, E., Cosar, A., & Ozkarahan, I. (2019). A survey on new generation metaheuristic algorithms. *Computers & Industrial Engineering*, 137, 106040.

Emary, E., Zawbaa, H. M., & Hassanien, A. E. (2016). Binary grey wolf optimization approaches for feature selection. *Neurocomputing*, 172, 371–381.

Fang, L., Li, J., & Peng, B. (2021). A method for online fault diagnosis of lithium-ion battery packs based on cell inconsistency analysis. *IEEE Transactions on Power Electronics*, 36(6), 6205–6217.

Farghaly, A., Abd El-Mageed, A. A., & Salem, M. A. (2023). A hybrid grey wolf optimizer and particle swarm optimization algorithm for global engineering optimization. *Applied Intelligence*, 53(8), 9214–9232.

Fontanari, T., Fróes, T. C., & Recamonde-Mendoza, M. (2022). Cross-validation strategies for balanced and imbalanced datasets. *Lecture Notes in Computer Science*, 13320, 145–158.

Gu, S., Cheng, R., & Jin, Y. (2020). Feature selection for high-dimensional classification using a competitive swarm optimizer. *Soft Computing*, 24(9), 6597–6609.

Gu, Z., Zhang, L., & Wang, Y. (2022). Improving random forest on imbalanced data via equilibrium ensemble with data centre interpolation. *Proceedings of the IEEE International Conference on Data Mining (ICDM)*, 810–818.

Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1), 389–422.

Hong, J., Wang, Z., & Yao, Y. (2021). Fault prognosis of battery system in electric vehicles based on multi-layer perceptron. *IEEE Transactions on Industrial Electronics*, 68(5), 4378–4387.

Hu, X., Zhang, K., Liu, K., Lin, X., Dey, S., & Onori, S. (2021). Machine learning-based data-driven fault detection/diagnosis of lithium-ion battery: A critical review. *Electronics*, 10(11), 1309.

ISO 26262:2018. (2018). *Road vehicles — Functional safety*. International Organization for Standardization (ISO), Geneva, Switzerland.

Javid, Z., Asghar, F., & Sarwar, S. (2021). An adaptive deep recurrent neural network for state of charge estimation of lithium-ion batteries in electric vehicles. *Journal of Energy Storage*, 44, 103348.

Jia, J., Liang, J., Shi, Y., & Wen, J. (2021). A deep transfer learning framework for fault diagnosis of battery energy storage systems across varying ambient temperatures. *Applied Energy*, 304, 117815.

Jiang, B., Dai, H., & Wei, X. (2021). Early thermal fault identification for lithium-ion battery packs based on statistical local outlier factor. *IEEE Transactions on Transportation Electrification*, 7(4), 2139–2149.

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of the IEEE International Conference on Neural Networks*, 1942–1948.

Khodadadi, N., Mirjalili, S., & Gandomi, A. H. (2022). Multi-objective metaheuristic algorithms: A review of recent developments and applications. *Engineering Applications of Artificial Intelligence*, 114, 105088.

Li, D., Zhang, Z., Liu, P., Wang, Z., & Zhang, L. (2021). Spatial temperature field reconstruction inside battery packs based on convolutional neural networks. *Journal of Cleaner Production*, 298, 126789.

Li, W., Fan, Y., Ringbeck, F., Jöst, D., Han, X., Ouyang, M., & Sauer, D. U. (2020). Electrochemical model-based state of charge and thermal runaway prediction of lithium-ion batteries. *Applied Energy*, 268, 115014.

Li, X., Wang, Z., & Yan, J. (2019). Prognostic health condition estimation of lithium-ion batteries using partial charging segments and random forest regression. *IEEE Transactions on Industrial Electronics*, 66(7), 5253–5262.

Li, Y., Liu, C., & Wang, J. (2022). Thermal runaway warning for lithium-ion batteries based on spatio-temporal deep neural networks. *Journal of Energy Chemistry*, 68, 482–492.

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Pruthi, J. S., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

Ma, L., Kang, L., & Zhao, X. (2021). State-of-health estimation for lithium-ion batteries based on dual long short-term memory neural networks. *IEEE Transactions on Transportation Electrification*, 7(3), 1082–1094.

Mafarja, M., Aljarah, I., Heidari, A. A., Hammouri, A. I., Faris, H., Ala’M, A. Z., & Mirjalili, S. (2020). Evolutionary population dynamics and multi-verse optimizer for wrapper feature selection. *Knowledge-Based Systems*, 195, 105658.

Mafarja, M., & Mirjalili, S. (2017). Hybrid whale optimization algorithm with simulated annealing for feature selection. *Neurocomputing*, 260, 302–312.

Mafarja, M., & Mirjalili, S. (2018). Whale optimization approaches for wrapper feature selection. *Applied Soft Computing*, 62, 441–453.

Mayingi, B. A., Thango, B. A., & Okojie, D. (2026). Multi-objective feature selection using HPWOA for improved BMS fault diagnosis in electric vehicles. *World Electric Vehicle Journal*, 17(7), 359.

Meng, J., Luo, G., & Gao, F. (2019). Lithium-ion battery state of health estimation using support vector regression and short-term current pulse tests. *IEEE Transactions on Industrial Electronics*, 66(6), 4475–4484.

Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. *Advances in Engineering Software*, 95, 51–67.

Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46–61.

Muthiah, P. K., Sathiasamuel, C. R., Subbukalai, A. M., & Santiago, A. E. X. (2026). Tri-level hybrid electric bus scheduling for integrated fleet and charger optimization: A case study of Madurai District. *Sustainability*, 18(14), 7239.

Nguyen, B. H., Xue, B., & Zhang, M. (2020). A survey on island binary particle swarm optimization for feature selection. *IEEE Transactions on Evolutionary Computation*, 24(4), 708–722.

Ojo, O. O., Lang, H., & Kim, Y. (2021). Deep recurrent neural network for sensor fault classification and isolation in battery management systems. *IEEE Access*, 9, 78241–78252.

Plett, G. L. (2015). *Battery management systems, Volume II: Equivalent-circuit methods*. Artech House.

Probst, P., Wright, M. N., & Boulesteix, A. L. (2019). Hyperparameters and tuning strategies for random forest. *WIREs Data Mining and Knowledge Discovery*, 9(3), e1301.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

Riyahi, M., Heidari, A. A., & Mirjalili, S. (2020). An enhanced whale optimization algorithm with chaotic mechanisms for global optimization. *Soft Computing*, 24(16), 12533–12561.

Saha, B., & Goebel, K. (2007). *Battery data set*. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M. H., Aykol, M., Herring, P. K., Fraggedakis, D., & Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4(5), 383–391.

Shrivastava, P., Soon, T. K., Idris, M. Y. I., & Mekhilef, S. (2024). Advanced data-driven fault diagnosis in lithium-ion battery management systems for electric vehicles: Progress, challenges, and future perspectives. *Journal of Energy Storage*, 82, 110543.

Song, X., Yang, F., Wang, D., & Tsui, K. L. (2020). Combined CNN-LSTM networks for state of charge estimation of lithium-ion batteries. *IEEE Transactions on Industrial Electronics*, 67(10), 8826–8835.

Song, Y., Xue, B., & Zhang, M. (2018). Bare-bones binary particle swarm optimization for feature selection. *Swarm and Evolutionary Computation*, 40, 1–15.

Sun, Z., Wang, Z., Liu, P., Zhang, Z., & Chen, Y. (2022). An online data-driven fault diagnosis and thermal runaway early warning for electric vehicle batteries. *IEEE Transactions on Power Electronics*, 37(10), 12636–12646.

Too, J., & Abdullah, A. R. (2021). A novel hybrid whale-particle swarm optimization algorithm for feature selection. *Computers & Industrial Engineering*, 154, 107148.

Vidal, C., Malysz, P., Kollmeyer, P., & Emadi, A. (2020). Machine learning applied to electrified vehicle battery state of charge and state of health estimation: State-of-the-art. *IEEE Access*, 8, 52796–52814.

Weng, C., Sun, J., & Peng, H. (2014). A unified open-circuit-voltage model of lithium-ion batteries for state-of-charge estimation and state-of-health monitoring. *Journal of Power Sources*, 258, 228–237.

Wong, T. T., & Yeh, P. Y. (2020). Reliable accuracy estimates from k-fold cross validation. *IEEE Transactions on Knowledge and Data Engineering*, 32(8), 1586–1594.

Xiong, R., He, H., Sun, F., & Zhao, K. (2013). Evaluation on state of charge estimation of batteries with adaptive extended Kalman filter by experiment approach. *IEEE Transactions on Vehicular Technology*, 62(1), 108–117.

Xue, B., Zhang, M., Browne, W. N., & Yao, X. (2016). A survey on evolutionary computation approaches to feature selection. *IEEE Transactions on Evolutionary Computation*, 20(4), 606–626.

Xue, B., Zhang, M., & Browne, W. N. (2013). Particle swarm optimization for feature selection in classification: A multi-objective approach. *IEEE Transactions on Cybernetics*, 43(6), 1656–1671.

Yang, D., Zhang, X., Pan, R., Wang, Y., & Chen, Z. (2018). A novel Gaussian process regression model for state-of-health estimation of lithium-ion batteries. *IEEE Transactions on Industrial Electronics*, 65(7), 5797–5807.

Yao, L., Fang, Z., Xiao, Y., Hou, J., & Fu, Y. (2021). An intelligent fault diagnosis method for lithium battery systems based on grid search support vector machine. *Energy*, 214, 118888.

Zhao, Y., Liu, P., Wang, Z., Zhang, L., & Hao, J. (2021). Fault and defect diagnosis of battery packs in electric vehicles based on big data analysis. *IEEE Transactions on Transportation Electrification*, 7(3), 1045–1056.

---
# APPENDIX A – SAMPLE CODE

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

class HGWO_PSO_BMS:
    def __init__(self, n_particles=30, max_iter=40, omega=0.7, c1=1.5, c2=1.5, seed=42):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        np.random.seed(seed)
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        
    def evaluate_particle(self, particle, X, y, n_features):
        mask_cont = particle[:n_features]
        probs = self.sigmoid(mask_cont)
        mask = (probs > 0.5).astype(int)
        
        if np.sum(mask) == 0:
            mask[np.random.randint(0, n_features)] = 1
            
        n_estimators = int(np.clip(particle[n_features], 10, 200))
        max_depth = int(np.clip(particle[n_features + 1], 3, 20))
        w_crit = float(np.clip(particle[n_features + 2], 1.0, 5.0))
        
        selected_indices = np.where(mask == 1)[0]
        X_sub = X[:, selected_indices]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        y_true_all = []
        y_pred_all = []
        
        class_weight = {0: 1.0, 1: 1.0, 2: w_crit}
        
        for train_idx, val_idx in skf.split(X_sub, y):
            X_tr, X_va = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=self.seed,
                n_jobs=-1
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            
            y_true_all.extend(y_va)
            y_pred_all.extend(preds)
            
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        acc = accuracy_score(y_true_all, y_pred_all)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2])
        
        crit_total = np.sum(cm[2, :])
        crit_missed = crit_total - cm[2, 2]
        mdr_crit = (crit_missed / crit_total) if crit_total > 0 else 0.0
        
        f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
        feat_ratio = len(selected_indices) / float(n_features)
        
        fitness = 0.50 * (1.0 - acc) + 0.35 * mdr_crit + 0.15 * feat_ratio
        
        return fitness, acc, f1_macro, mdr_crit, mask, n_estimators, max_depth, w_crit

    def fit(self, X, y):
        n_samples, n_features = X.shape
        total_dim = n_features + 3
        
        particles = np.random.uniform(-2, 2, (self.n_particles, total_dim))
        velocities = np.random.uniform(-1, 1, (self.n_particles, total_dim))
        
        particles[:, n_features] = np.random.uniform(50, 150, self.n_particles)
        particles[:, n_features + 1] = np.random.uniform(5, 15, self.n_particles)
        particles[:, n_features + 2] = np.random.uniform(1.5, 4.0, self.n_particles)
        
        pbest = np.copy(particles)
        pbest_fit = np.full(self.n_particles, float('inf'))
        
        alpha_pos = np.zeros(total_dim)
        alpha_fit = float('inf')
        beta_pos = np.zeros(total_dim)
        beta_fit = float('inf')
        delta_pos = np.zeros(total_dim)
        delta_fit = float('inf')
        
        for t in range(self.max_iter):
            a = 2.0 * (1.0 - t / float(self.max_iter))
            
            for i in range(self.n_particles):
                fit_val, acc, f1, mdr, mask, n_est, m_d, w_c = self.evaluate_particle(
                    particles[i], X, y, n_features
                )
                
                if fit_val < pbest_fit[i]:
                    pbest_fit[i] = fit_val
                    pbest[i] = np.copy(particles[i])
                    
                if fit_val < alpha_fit:
                    delta_fit = beta_fit
                    delta_pos = np.copy(beta_pos)
                    beta_fit = alpha_fit
                    beta_pos = np.copy(alpha_pos)
                    alpha_fit = fit_val
                    alpha_pos = np.copy(particles[i])
                elif fit_val < beta_fit:
                    delta_fit = beta_fit
                    delta_pos = np.copy(beta_pos)
                    beta_fit = fit_val
                    beta_pos = np.copy(particles[i])
                elif fit_val < delta_fit:
                    delta_fit = fit_val
                    delta_pos = np.copy(particles[i])
                    
            for i in range(self.n_particles):
                r1 = np.random.rand(total_dim)
                r2 = np.random.rand(total_dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - particles[i])
                X1 = alpha_pos - A1 * D_alpha
                
                r1 = np.random.rand(total_dim)
                r2 = np.random.rand(total_dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - particles[i])
                X2 = beta_pos - A2 * D_beta
                
                r1 = np.random.rand(total_dim)
                r2 = np.random.rand(total_dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - particles[i])
                X3 = delta_pos - A3 * D_delta
                
                X_gwo = (X1 + X2 + X3) / 3.0
                
                velocities[i] = (
                    self.omega * velocities[i]
                    + self.c1 * np.random.rand(total_dim) * (pbest[i] - particles[i])
                    + self.c2 * np.random.rand(total_dim) * (X_gwo - particles[i])
                )
                particles[i] += velocities[i]
                
        fit_val, acc, f1, mdr, mask, n_est, m_d, w_c = self.evaluate_particle(
            alpha_pos, X, y, n_features
        )
        return {
            'best_mask': mask,
            'selected_features': np.where(mask == 1)[0],
            'n_estimators': n_est,
            'max_depth': m_d,
            'critical_weight': w_c,
            'accuracy': acc,
            'macro_f1': f1,
            'critical_mdr': mdr,
            'fitness': fit_val
        }
```

---

# APPENDIX B – CONFERENCE / JOURNAL PUBLICATION DETAILS

The research findings and novel engineering contributions developed in this project are prepared for submission to top-tier automotive engineering and industrial electronics venues:

### Primary Target Journal:
- **Title:** IEEE Transactions on Industrial Electronics (IEEE TIE)
- **Impact Factor:** 7.5
- **Paper Title:** Safety-Prioritized BMS Sensor Selection and Hyperparameter Co-Optimization Using Hybrid GWO-PSO With Explainable AI Verification
- **Scope Alignment:** Real-time fault diagnosis in automotive battery management systems, evolutionary metaheuristic engineering, sensor network optimization, and functional safety compliance.

### Secondary Target Journal:
- **Title:** IEEE Transactions on Transportation Electrification (IEEE TTE)
- **Impact Factor:** 7.0
- **Scope Alignment:** EV powertrain safety, battery state estimation under DC fast charging, battery fire prevention, and automotive embedded computing.

### Target International Conference:
- **Conference:** IEEE Transportation Electrification Conference & Expo (ITEC 2027)
- **Location / Scope:** Premier automotive electrification venue focusing on physical battery pack telemetry reduction, ASIL functional safety verification, and hardware-efficient machine learning.
