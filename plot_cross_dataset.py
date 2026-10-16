"""
Generates high-resolution cross-dataset benchmark comparison plots
showing HGWO-PSO generalization across EV Thermal Runaway and NASA Battery Aging.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

methods = ['BPSO (KNN)', 'BWOA (KNN)', 'HGWO-PSO (Ours)']
ev_acc = [64.8, 57.0, 94.2]
nasa_acc = [97.48, 97.64, 99.37]

ev_mdr = [59.14, 73.01, 5.89]
nasa_mdr = [2.31, 1.54, 0.00]

ev_sensors = [8, 4, 5]
nasa_sensors = [4, 3, 3]

x = np.arange(len(methods))
width = 0.35

# 1. Accuracy Comparison
rects1 = axes[0].bar(x - width/2, ev_acc, width, label='EV Thermal Runaway (17 Feat)', color='#4361EE', alpha=0.9, edgecolor='black')
rects2 = axes[0].bar(x + width/2, nasa_acc, width, label='NASA Battery Aging (14 Feat)', color='#4CC9F0', alpha=0.9, edgecolor='black')
axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Classification Accuracy Comparison (Higher is Better)', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, fontsize=10, fontweight='bold')
axes[0].set_ylim(40, 105)
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)

# Add values above bars
for rect in rects1:
    h = rect.get_height()
    axes[0].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    axes[0].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Critical MDR Comparison
rects3 = axes[1].bar(x - width/2, ev_mdr, width, label='EV Thermal Runaway', color='#E63946', alpha=0.9, edgecolor='black')
rects4 = axes[1].bar(x + width/2, nasa_mdr, width, label='NASA Battery Aging', color='#F77F00', alpha=0.9, edgecolor='black')
axes[1].set_ylabel('Critical Miss Detection Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Critical Safety MDR (Lower is Better - Safety Critical)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, fontsize=10, fontweight='bold')
axes[1].set_ylim(0, 85)
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)

for rect in rects3:
    h = rect.get_height()
    axes[1].annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects4:
    h = rect.get_height()
    axes[1].annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Selected Sensors Count
rects5 = axes[2].bar(x - width/2, ev_sensors, width, label='EV Sensors (Initial: 17)', color='#2A9D8F', alpha=0.9, edgecolor='black')
rects6 = axes[2].bar(x + width/2, nasa_sensors, width, label='NASA Sensors (Initial: 14)', color='#8338EC', alpha=0.9, edgecolor='black')
axes[2].set_ylabel('Selected Sensor Count', fontsize=12, fontweight='bold')
axes[2].set_title('Hardware Sensor Footprint (Compression)', fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(methods, fontsize=10, fontweight='bold')
axes[2].set_ylim(0, 12)
axes[2].legend(frameon=True, facecolor='white', framealpha=0.9)

for rect in rects5:
    h = rect.get_height()
    axes[2].annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects6:
    h = rect.get_height()
    axes[2].annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out_path = '/Users/abdullah/.gemini/antigravity-ide/brain/8911f183-f911-44ab-8ceb-076d231ad09f/cross_dataset_comparison.png'
plt.savefig(out_path, bbox_inches='tight')
plt.savefig('cross_dataset_comparison.png', bbox_inches='tight')
print(f"Plot saved to: {out_path}")
