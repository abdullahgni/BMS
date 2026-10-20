"""
Generate professional figures for Figures 4.5, 4.6, 4.7:
- Figure 4.5: Sequence Diagram
- Figure 4.6: UML Class Diagram
- Figure 4.7: Activity Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure_4_5():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')
    
    lifelines = [
        ("MainEngine", 0.15),
        ("HGWO_PSO", 0.38),
        ("Evaluator (5-Fold CV)", 0.65),
        ("SHAP_Engine", 0.88)
    ]
    
    # Draw lifeline headers and vertical lines
    for name, x in lifelines:
        box = patches.FancyBboxPatch((x-0.09, 0.88), 0.18, 0.08, boxstyle="round,pad=0.01", ec="#1E3A8A", fc="#EFF6FF", lw=1.5)
        ax.add_patch(box)
        ax.text(x, 0.92, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1E3A8A")
        ax.plot([x, x], [0.10, 0.88], linestyle="dashed", color="#9CA3AF", lw=1.2)
        
    # Sequence messages
    msgs = [
        (0.15, 0.38, 0.80, "1. Ingest Telemetry & Initialize Swarm (N=30)", "->", "#1F2937"),
        (0.38, 0.65, 0.70, "2. Transmit Candidate Mask x_feat & Hyperparameters", "->", "#2563EB"),
        (0.65, 0.65, 0.62, "3. Execute Stratified 5-Fold Training", "self", "#7C3AED"),
        (0.65, 0.38, 0.52, "4. Return Accuracy, Critical MDR, & Sensor Count", "-->", "#059669"),
        (0.38, 0.38, 0.44, "5. Update α, β, δ Positions & PSO Velocities", "self", "#059669"),
        (0.38, 0.15, 0.34, "6. Convergence Reached (Optimal Mask & Params)", "-->", "#D97706"),
        (0.15, 0.88, 0.24, "7. Request Explainability on Selected 5 Sensors", "->", "#EC4899"),
        (0.88, 0.15, 0.14, "8. Return SHAP Beeswarm & Importance Plots", "-->", "#EC4899"),
    ]
    
    for x1, x2, y, text, style, color in msgs:
        if style == "self":
            # Loopback arrow
            ax.annotate('', xy=(x1, y-0.03), xytext=(x1, y+0.02),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4", color=color, lw=1.5))
            ax.text(x1+0.04, y, text, fontsize=7.5, va='center', color=color, fontweight='bold')
        else:
            ls = "dashed" if style == "-->" else "solid"
            ax.annotate('', xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", linestyle=ls, color=color, lw=1.5))
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, y + 0.02, text, fontsize=7.5, ha='center', va='bottom', color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig("fig_4_5_sequence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_5_sequence.png")

def generate_figure_4_6():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')
    
    classes = [
        ("TelemetryData", 0.18, 0.72, 
         "+ filepath: str\n+ X: DataFrame\n+ y: ndarray\n+ feature_names: list", 
         "+ load_data(): void\n+ z_score_scale(): void\n+ map_safety_target(): void"),
        ("Particle", 0.50, 0.72, 
         "+ position: ndarray\n+ velocity: ndarray\n+ pbest_pos: ndarray\n+ pbest_fit: float", 
         "+ update_position(): void\n+ sigmoid_discretize(): void"),
        ("HGWO_PSO", 0.82, 0.72, 
         "+ n_wolves: int\n+ max_iter: int\n+ alpha_pos: ndarray\n+ beta_pos: ndarray", 
         "+ optimize(): dict\n+ update_leaders(): void\n+ get_best_pipeline(): Estimator"),
        ("SafetyEvaluator", 0.34, 0.25, 
         "+ k_folds: int = 5\n+ w_acc: float = 0.50\n+ w_mdr: float = 0.35\n+ w_sens: float = 0.15", 
         "+ compute_fitness(): float\n+ calc_critical_mdr(): float\n+ stratified_cv(): tuple"),
        ("SHAPExplainer", 0.68, 0.25, 
         "+ model: Estimator\n+ X_selected: DataFrame\n+ shap_values: ndarray", 
         "+ generate_summary_bar(): void\n+ generate_beeswarm(): void\n+ plot_dependence(): void")
    ]
    
    for name, x, y, attrs, methods in classes:
        # Header
        r_head = patches.FancyBboxPatch((x-0.14, y+0.06), 0.28, 0.06, boxstyle="round,pad=0.01", ec="#1E3A8A", fc="#1E3A8A", lw=1.2)
        ax.add_patch(r_head)
        ax.text(x, y+0.09, name, ha='center', va='center', fontsize=9, fontweight='bold', color="#FFFFFF")
        
        # Body
        r_body = patches.FancyBboxPatch((x-0.14, y-0.12), 0.28, 0.18, boxstyle="square,pad=0.0", ec="#1E3A8A", fc="#F8FAFC", lw=1.2)
        ax.add_patch(r_body)
        ax.text(x-0.12, y+0.01, attrs, ha='left', va='center', fontsize=6.8, color="#1F2937")
        ax.plot([x-0.14, x+0.14], [y-0.03, y-0.03], color="#CBD5E1", lw=1)
        ax.text(x-0.12, y-0.07, methods, ha='left', va='center', fontsize=6.8, color="#065F46")

    # Connectors
    # TelemetryData -> HGWO_PSO
    ax.annotate('', xy=(0.70, 0.75), xytext=(0.32, 0.75), arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.2))
    # Particle aggregation into HGWO_PSO
    ax.annotate('', xy=(0.70, 0.72), xytext=(0.64, 0.72), arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.5))
    # HGWO_PSO -> SafetyEvaluator
    ax.annotate('', xy=(0.42, 0.35), xytext=(0.75, 0.60), arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.2))
    # HGWO_PSO -> SHAPExplainer
    ax.annotate('', xy=(0.72, 0.35), xytext=(0.80, 0.60), arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.2))

    plt.tight_layout()
    plt.savefig("fig_4_6_class.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_6_class.png")

def generate_figure_4_7():
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)
    ax.axis('off')
    
    steps = [
        ("Start: Telemetry Array Initialized", 0.5, 0.94, "start"),
        ("Initialize N Particles with Random Masks & Hyperparameters", 0.5, 0.84, "rect"),
        ("Convert Continuous Velocities to Binary Masks via Sigmoid S(v)", 0.5, 0.73, "rect"),
        ("Check: Is Active Sensor Mask Empty?", 0.5, 0.61, "diamond"),
        ("Activate Single Random Feature (Recovery Guard)", 0.18, 0.61, "rect_side"),
        ("Execute Stratified 5-Fold Cross-Validation on Cost-Sensitive RF", 0.5, 0.49, "rect"),
        ("Calculate Accuracy, Critical MDR, and Sensor Ratio (d_sel / D)", 0.5, 0.38, "rect"),
        ("Synthesize Multi-Objective Cost F = 0.50·Err + 0.35·MDR + 0.15·Ratio", 0.5, 0.27, "rect"),
        ("Update α, β, δ Wolf Positions & Personal Best Vectors", 0.5, 0.16, "rect"),
        ("Stop: Output Optimal 5-Sensor Subset & Trained Model", 0.5, 0.05, "end")
    ]
    
    for text, x, y, kind in steps:
        if kind == "start":
            circle = patches.Circle((x, y), 0.025, fc="#1E3A8A", ec="#1E3A8A", lw=1.5)
            ax.add_patch(circle)
            ax.text(x, y+0.04, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1E3A8A")
            ax.annotate('', xy=(x, y-0.05), xytext=(x, y-0.025), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5))
        elif kind == "end":
            circle1 = patches.Circle((x, y), 0.03, fc="#FFFFFF", ec="#1E3A8A", lw=1.5)
            circle2 = patches.Circle((x, y), 0.02, fc="#1E3A8A", ec="#1E3A8A", lw=1.5)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.text(x, y-0.04, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color="#1E3A8A")
        elif kind == "diamond":
            diamond = patches.Polygon([[x, y+0.05], [x+0.22, y], [x, y-0.05], [x-0.22, y]], ec="#D97706", fc="#FFFBEB", lw=1.5)
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', fontsize=7.5, fontweight='bold', color="#B45309")
            # Yes -> Side
            ax.annotate('Yes', xy=(0.30, 0.61), xytext=(0.28, 0.61), arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.2), fontsize=7.5, va='bottom')
            # No -> Down
            ax.annotate('No', xy=(x, 0.53), xytext=(x, y-0.05), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5), fontsize=7.5, ha='left')
        elif kind == "rect_side":
            r = patches.FancyBboxPatch((x-0.12, y-0.035), 0.24, 0.07, boxstyle="round,pad=0.01", ec="#EF4444", fc="#FEF2F2", lw=1.2)
            ax.add_patch(r)
            ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold', color="#DC2626")
            # Connect back to down
            ax.annotate('', xy=(0.40, 0.49), xytext=(x, y-0.035), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color="#DC2626", lw=1.2))
        else:
            r = patches.FancyBboxPatch((x-0.34, y-0.035), 0.68, 0.07, boxstyle="round,pad=0.01", ec="#2563EB", fc="#EFF6FF", lw=1.2)
            ax.add_patch(r)
            ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', color="#1E40AF")
            if y > 0.10:
                ax.annotate('', xy=(x, y-0.075), xytext=(x, y-0.035), arrowprops=dict(arrowstyle="->", color="#1F2937", lw=1.5))

    plt.tight_layout()
    plt.savefig("fig_4_7_activity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_4_7_activity.png")

if __name__ == '__main__':
    generate_figure_4_5()
    generate_figure_4_6()
    generate_figure_4_7()
