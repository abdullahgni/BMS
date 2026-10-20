"""
Generate high-resolution professional figures for Chapter 4:
- Figure 4.1: System Architecture
- Figure 4.2: Use Case Diagram
- Figure 4.3: Data Flow Diagram (DFD Level 1)
- Figure 4.4: State Transition Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure_4_1():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')
    
    # Outer box
    rect_outer = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02,rounding_size=0.03", 
                                        ec="#1E3A8A", fc="#F8FAFC", lw=2)
    ax.add_patch(rect_outer)
    ax.text(0.5, 0.94, "PROPOSED HGWO-PSO BMS OPTIMIZATION ARCHITECTURE", 
            ha='center', va='center', fontsize=12, fontweight='bold', color="#1E3A8A")
    
    # Tier 1: Ingestion
    rect_t1 = patches.FancyBboxPatch((0.06, 0.77), 0.88, 0.12, boxstyle="round,pad=0.02", ec="#3B82F6", fc="#EFF6FF", lw=1.5)
    ax.add_patch(rect_t1)
    ax.text(0.5, 0.84, "Tier 1: Multi-Channel Telemetry Ingestion & Preprocessing", ha='center', va='center', fontsize=10, fontweight='bold', color="#1E40AF")
    ax.text(0.5, 0.79, "Raw Sensor Stream (Voltage, Current, Temps, SOH, SOC, Moisture, Pressure, Vibration) ➔ Z-Score Scaling ➔ 3-Class Safety Mapping", 
            ha='center', va='center', fontsize=8.5, color="#1F2937")

    # Arrow 1
    ax.annotate('', xy=(0.5, 0.70), xytext=(0.5, 0.77), arrowprops=dict(arrowstyle="-|>", color="#1E3A8A", lw=2, mutation_scale=15))

    # Tier 2: HGWO-PSO Engine
    rect_t2 = patches.FancyBboxPatch((0.06, 0.38), 0.88, 0.31, boxstyle="round,pad=0.02", ec="#059669", fc="#ECFDF5", lw=1.5)
    ax.add_patch(rect_t2)
    ax.text(0.5, 0.65, "Tier 2: HGWO-PSO Safety Co-Optimization Engine", ha='center', va='center', fontsize=10.5, fontweight='bold', color="#065F46")
    ax.text(0.5, 0.60, "Chromosome: [ Binary Feature Selection Mask (17-bit / 15-bit) | n_estimators | max_depth | critical_weight ]", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#047857")
    
    # Inner boxes inside Tier 2
    b1 = patches.FancyBboxPatch((0.08, 0.41), 0.40, 0.15, boxstyle="round,pad=0.01", ec="#10B981", fc="#FFFFFF", lw=1)
    b2 = patches.FancyBboxPatch((0.52, 0.41), 0.40, 0.15, boxstyle="round,pad=0.01", ec="#10B981", fc="#FFFFFF", lw=1)
    ax.add_patch(b1)
    ax.add_patch(b2)
    
    ax.text(0.28, 0.52, "Swarm Exploration & Exploitation", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#065F46")
    ax.text(0.28, 0.45, "• Grey Wolf Hierarchy (α, β, δ guidance)\n• PSO Velocity Updates & Inertia\n• Sigmoid Transfer S(v) + Boundary Clamping", 
            ha='center', va='center', fontsize=7.5, color="#374151")
    
    ax.text(0.72, 0.52, "Inner Safety Fitness Evaluation", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#065F46")
    ax.text(0.72, 0.45, "• Strict Stratified 5-Fold Cross-Validation\n• Cost-Sensitive Random Forest (w_crit ≥ 1.0)\n• Fitness = 0.50·Err + 0.35·MDR + 0.15·(k/D)", 
            ha='center', va='center', fontsize=7.5, color="#374151")

    # Arrow 2
    ax.annotate('', xy=(0.5, 0.31), xytext=(0.5, 0.38), arrowprops=dict(arrowstyle="-|>", color="#1E3A8A", lw=2, mutation_scale=15))

    # Tier 3: Output Models & XAI
    rect_t3a = patches.FancyBboxPatch((0.06, 0.06), 0.42, 0.24, boxstyle="round,pad=0.02", ec="#7C3AED", fc="#F5F3FF", lw=1.5)
    rect_t3b = patches.FancyBboxPatch((0.52, 0.06), 0.42, 0.24, boxstyle="round,pad=0.02", ec="#D97706", fc="#FFFBEB", lw=1.5)
    ax.add_patch(rect_t3a)
    ax.add_patch(rect_t3b)

    ax.text(0.27, 0.26, "Optimal Parsimonious Pipeline", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#5B21B6")
    ax.text(0.27, 0.16, "• 5 Critical Sensors (70.6% Hardware Reduction)\n• EV Dataset: 94.20% Acc, 5.89% Critical MDR\n• NASA Dataset: 99.37% Acc, 0.00% Critical MDR\n• Embedded Latency: < 10 ms on ARM Cortex", 
            ha='center', va='center', fontsize=8, color="#374151")

    ax.text(0.73, 0.26, "Explainable AI (XAI) Verification", ha='center', va='center', fontsize=9.5, fontweight='bold', color="#B45309")
    ax.text(0.73, 0.16, "• SHAP TreeExplainer Attributions\n• Global Mean |SHAP| Sensor Ranking\n• Beeswarm Minority Class Attribution\n• Non-Linear Temp vs SOC Interactions", 
            ha='center', va='center', fontsize=8, color="#374151")

    plt.tight_layout()
    plt.savefig("fig_4_1_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_1_architecture.png")

def generate_figure_4_2():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.axis('off')
    
    # Boundary box
    rect_sys = patches.FancyBboxPatch((0.25, 0.05), 0.50, 0.90, boxstyle="round,pad=0.02", ec="#4B5563", fc="#F9FAFB", lw=1.5)
    ax.add_patch(rect_sys)
    ax.text(0.5, 0.91, "BMS Intelligent Telemetry Diagnostic System", ha='center', va='center', fontsize=11, fontweight='bold', color="#111827")
    
    # Use cases
    ucs = [
        ("UC-01: Ingest & Preprocess Telemetry Stream", 0.78),
        ("UC-02: Execute HGWO-PSO Co-Optimization", 0.58),
        ("UC-03: Evaluate Multi-Objective Safety Fitness", 0.38),
        ("UC-04: Generate SHAP Explainability Audit", 0.18)
    ]
    for text, y in ucs:
        el = patches.Ellipse((0.5, y), 0.44, 0.12, ec="#2563EB", fc="#EFF6FF", lw=1.2)
        ax.add_patch(el)
        ax.text(0.5, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1E40AF")
        
    # Actors
    # Actor 1: Telemetry Stream (Left)
    ax.text(0.10, 0.78, "Telemetry Sensor\nStream (Hardware)", ha='center', va='center', fontsize=9, fontweight='bold', color="#1F2937")
    circle1 = patches.Circle((0.10, 0.86), 0.035, ec="#374151", fc="#E5E7EB", lw=1.5)
    ax.add_patch(circle1)
    ax.annotate('', xy=(0.28, 0.78), xytext=(0.18, 0.78), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    
    # Actor 2: System Engineer (Left Bottom)
    ax.text(0.10, 0.54, "System Engineer /\nBMS Specialist", ha='center', va='center', fontsize=9, fontweight='bold', color="#1F2937")
    circle2 = patches.Circle((0.10, 0.62), 0.035, ec="#374151", fc="#E5E7EB", lw=1.5)
    ax.add_patch(circle2)
    ax.annotate('', xy=(0.28, 0.58), xytext=(0.18, 0.58), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    
    # Actor 3: Safety Auditor (Right)
    ax.text(0.90, 0.22, "Automotive Safety\nAuditor (ISO 26262)", ha='center', va='center', fontsize=9, fontweight='bold', color="#1F2937")
    circle3 = patches.Circle((0.90, 0.30), 0.035, ec="#374151", fc="#E5E7EB", lw=1.5)
    ax.add_patch(circle3)
    ax.annotate('', xy=(0.72, 0.18), xytext=(0.82, 0.18), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    
    # Internal relationship
    ax.annotate('', xy=(0.5, 0.44), xytext=(0.5, 0.52), arrowprops=dict(arrowstyle="->", linestyle="dashed", color="#6B7280", lw=1.2))
    ax.text(0.58, 0.48, "<<includes>>", ha='center', va='center', fontsize=7.5, color="#6B7280", style='italic')

    ax.annotate('', xy=(0.5, 0.24), xytext=(0.5, 0.32), arrowprops=dict(arrowstyle="->", linestyle="dashed", color="#6B7280", lw=1.2))
    ax.text(0.58, 0.28, "<<extends>>", ha='center', va='center', fontsize=7.5, color="#6B7280", style='italic')

    plt.tight_layout()
    plt.savefig("fig_4_2_usecase.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_2_usecase.png")

def generate_figure_4_3():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    
    # DFD Level 1 blocks
    blocks = [
        ("1.0 Telemetry Ingestion\n& Standardization", 0.15, 0.70, "#3B82F6", "#EFF6FF"),
        ("2.0 Feature Masking &\nChromosome Encoding", 0.50, 0.70, "#10B981", "#ECFDF5"),
        ("3.0 Stratified 5-Fold\nModel Training", 0.85, 0.70, "#8B5CF6", "#F5F3FF"),
        ("4.0 Multi-Objective Safety\nFitness Evaluation", 0.85, 0.25, "#F59E0B", "#FFFBEB"),
        ("5.0 Swarm Leader Update\n(α, β, δ Positions)", 0.50, 0.25, "#10B981", "#ECFDF5"),
        ("6.0 Global Model Extraction\n& SHAP Interpretation", 0.15, 0.25, "#EC4899", "#FDF2F8"),
    ]
    
    for text, x, y, ec, fc in blocks:
        r = patches.FancyBboxPatch((x-0.12, y-0.12), 0.24, 0.24, boxstyle="round,pad=0.01", ec=ec, fc=fc, lw=1.5)
        ax.add_patch(r)
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1F2937")
        
    # Data flow arrows
    # 1 -> 2
    ax.annotate('Cleaned Data', xy=(0.38, 0.70), xytext=(0.27, 0.70), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5), fontsize=7.5, ha='center', va='bottom')
    # 2 -> 3
    ax.annotate('Sub-Selected Features', xy=(0.73, 0.70), xytext=(0.62, 0.70), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5), fontsize=7.5, ha='center', va='bottom')
    # 3 -> 4
    ax.annotate('Validation Metrics', xy=(0.85, 0.37), xytext=(0.85, 0.58), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5), fontsize=7.5, ha='left', va='center')
    # 4 -> 5
    ax.annotate('Fitness Cost', xy=(0.62, 0.25), xytext=(0.73, 0.25), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5), fontsize=7.5, ha='center', va='bottom')
    # 5 -> 2 (Feedback loop)
    ax.annotate('Updated Velocities & Positions', xy=(0.50, 0.58), xytext=(0.50, 0.37), arrowprops=dict(arrowstyle="->", linestyle="dashed", color="#059669", lw=1.5), fontsize=7.5, ha='right', va='center')
    # 5 -> 6 (Upon convergence)
    ax.annotate('Optimal Solution (P*)', xy=(0.27, 0.25), xytext=(0.38, 0.25), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5), fontsize=7.5, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("fig_4_3_dfd.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_3_dfd.png")

def generate_figure_4_4():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.axis('off')
    
    states = [
        ("UNINITIALIZED\n(Awaiting Telemetry)", 0.15, 0.75, "#6B7280", "#F3F4F6"),
        ("NOMINAL_OPERATION (OK)\n(Safe Voltage & Temp)", 0.50, 0.75, "#10B981", "#ECFDF5"),
        ("ELEVATED_STRESS (Warning)\n(Temp > 45°C or High ΔV)", 0.50, 0.25, "#F59E0B", "#FFFBEB"),
        ("CRITICAL_RUNAWAY (Critical)\n(Thermal Catastrophe)", 0.85, 0.25, "#EF4444", "#FEF2F2"),
        ("SYSTEM_SHUTDOWN\n(Emergency Disconnect)", 0.85, 0.75, "#1F2937", "#E5E7EB")
    ]
    
    for text, x, y, ec, fc in states:
        r = patches.FancyBboxPatch((x-0.12, y-0.10), 0.24, 0.20, boxstyle="round,pad=0.01", ec=ec, fc=fc, lw=1.8)
        ax.add_patch(r)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color="#111827")
        
    # State transitions
    # Uninit -> OK
    ax.annotate('', xy=(0.38, 0.75), xytext=(0.27, 0.75), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5))
    ax.text(0.325, 0.78, "Sync Active", fontsize=7.5, ha='center')
    
    # OK <-> Warning
    ax.annotate('', xy=(0.48, 0.35), xytext=(0.48, 0.65), arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.5))
    ax.text(0.42, 0.50, "Thermal Rise", fontsize=7.5, ha='right', color="#D97706")
    
    ax.annotate('', xy=(0.52, 0.65), xytext=(0.52, 0.35), arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5))
    ax.text(0.58, 0.50, "Cooling Recovered", fontsize=7.5, ha='left', color="#059669")
    
    # Warning -> Critical
    ax.annotate('', xy=(0.73, 0.25), xytext=(0.62, 0.25), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8))
    ax.text(0.675, 0.28, "T > 60°C / Runaway", fontsize=7.5, ha='center', color="#DC2626", fontweight='bold')
    
    # Critical -> Shutdown
    ax.annotate('', xy=(0.85, 0.65), xytext=(0.85, 0.35), arrowprops=dict(arrowstyle="->", color="#111827", lw=2))
    ax.text(0.91, 0.50, "High-Voltage\nRelay Trip", fontsize=7.5, ha='left', fontweight='bold')

    plt.tight_layout()
    plt.savefig("fig_4_4_state_machine.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_4_state_machine.png")

if __name__ == '__main__':
    generate_figure_4_1()
    generate_figure_4_2()
    generate_figure_4_3()
    generate_figure_4_4()
