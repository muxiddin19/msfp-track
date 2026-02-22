"""
Generate speed vs accuracy tradeoff plot for ECCV paper.
Each method gets a unique marker+color combination for distinguishability.
"""

import matplotlib.pyplot as plt
import numpy as np

# Publication quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data from Table 1 (MOT17 results)
# Each method: (HOTA, FPS, category, color, marker)
methods = {
    # Motion-only methods - different shades of green with different markers
    'SORT':          (43.1, 143.3, 'motion', '#4DAF4A', 's'),       # green square
    'ByteTrack':     (54.8, 29.7,  'motion', '#77CC77', 'D'),       # light green diamond
    'OC-SORT':       (55.1, 28.5,  'motion', '#2D8632', 'v'),       # dark green down-triangle
    'Deep OC-SORT':  (56.8, 24.2,  'motion', '#90EE90', 'p'),       # pale green pentagon
    # Separate ReID - different shades of purple with different markers
    'DeepSORT':      (45.6, 13.7,  'reid',   '#984EA3', '^'),       # purple up-triangle
    'StrongSORT':    (55.6, 5.1,   'reid',   '#CE6DBD', 'h'),       # pink hexagon
    'BoTSORT':       (56.3, 9.2,   'reid',   '#6A3D9A', '<'),       # dark purple left-triangle
    # Baseline
    'LITE':          (61.1, 28.3,  'lite',   '#377EB8', 'o'),       # blue circle
    # Ours - red family with different markers
    'MSFP (concat)':    (62.4, 27.8, 'ours', '#FF6666', 'd'),       # light red thin diamond
    'MSFP (attention)': (62.8, 26.5, 'ours', '#E41A1C', 'P'),       # red plus
    'MSFP-Track':       (63.2, 26.1, 'ours', '#B22222', '*'),       # dark red star
}

fig, ax = plt.subplots(figsize=(10, 7))

# Plot each method with its own legend entry
for method, (hota, fps, category, color, marker) in methods.items():
    size = 220 if category == 'ours' else (160 if category == 'lite' else 120)
    ax.scatter(fps, hota, c=color, marker=marker,
               s=size, label=method, edgecolors='black', linewidths=0.8, zorder=5)

# Add reference lines for MSFP-Track
ax.axhline(y=63.2, color='#B22222', linestyle='--', alpha=0.3, linewidth=1)
ax.axvline(x=26.1, color='#B22222', linestyle='--', alpha=0.3, linewidth=1)

# Formatting
ax.set_xlabel('FPS (frames per second)', fontsize=12)
ax.set_ylabel('HOTA (%)', fontsize=12)
ax.set_title('Speed vs Accuracy Trade-off on MOT17', fontsize=14, fontweight='bold')

ax.set_xlim(0, 160)
ax.set_ylim(40, 66)

# Legend with all methods, organized in two columns
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9, ncol=2,
          columnspacing=0.8, handletextpad=0.4, borderpad=0.6)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add annotation for best accuracy-speed trade-off
ax.annotate('Best trade-off', xy=(26.1, 63.2), xytext=(55, 64.5),
            fontsize=10, color='#B22222',
            arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.5))

plt.tight_layout()

# Save
output_dir = '/home/muhiddin/lite/eccv2026/figures'
plt.savefig(f'{output_dir}/speed_accuracy_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"Saved speed_accuracy_tradeoff.pdf and .png to {output_dir}")
