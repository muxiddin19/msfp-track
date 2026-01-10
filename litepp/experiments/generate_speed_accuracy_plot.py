"""
Generate speed vs accuracy tradeoff plot for ECCV paper.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 1 (MOT17 results)
methods = {
    # Method: (HOTA, FPS, category)
    'SORT': (43.1, 143.3, 'motion'),
    'ByteTrack': (54.8, 29.7, 'motion'),
    'OC-SORT': (55.1, 28.5, 'motion'),
    'Deep OC-SORT': (56.8, 24.2, 'motion'),
    'DeepSORT': (45.6, 13.7, 'reid'),
    'StrongSORT': (55.6, 5.1, 'reid'),
    'BoTSORT': (56.3, 9.2, 'reid'),
    'LITE': (61.1, 28.3, 'lite'),
    'MSFP (concat)': (62.4, 27.8, 'ours'),
    'MSFP (attention)': (62.8, 26.5, 'ours'),
    'MSFP-Track': (63.2, 26.1, 'ours'),
}

# Colors and markers for categories
category_styles = {
    'motion': {'color': '#4DAF4A', 'marker': 's', 'label': 'Motion-only'},
    'reid': {'color': '#984EA3', 'marker': '^', 'label': 'Separate ReID'},
    'lite': {'color': '#377EB8', 'marker': 'o', 'label': 'LITE (baseline)'},
    'ours': {'color': '#E41A1C', 'marker': '*', 'label': 'Ours (MSFP-Track)'},
}

fig, ax = plt.subplots(figsize=(10, 7))

# Plot each method
plotted_categories = set()
for method, (hota, fps, category) in methods.items():
    style = category_styles[category]
    label = style['label'] if category not in plotted_categories else None
    plotted_categories.add(category)

    size = 200 if category == 'ours' else 120
    ax.scatter(fps, hota, c=style['color'], marker=style['marker'],
               s=size, label=label, edgecolors='black', linewidths=0.5, zorder=5)

    # Add method labels
    offset_x = 1.5
    offset_y = 0.8
    if method == 'SORT':
        offset_x = -8
        offset_y = -1.5
    elif method == 'StrongSORT':
        offset_x = 1
        offset_y = -1.5
    elif method == 'DeepSORT':
        offset_x = 1
        offset_y = 1
    elif method == 'MSFP-Track':
        offset_x = -12
        offset_y = 1
    elif method == 'LITE':
        offset_x = 1.5
        offset_y = -1.5
    elif method == 'MSFP (attention)':
        offset_x = -15
        offset_y = -0.5
    elif method == 'MSFP (concat)':
        offset_x = 1.5
        offset_y = -1.2

    ax.annotate(method, (fps, hota), xytext=(fps + offset_x, hota + offset_y),
                fontsize=9, alpha=0.85)

# Draw Pareto frontier for our methods
ours_points = [(fps, hota) for method, (hota, fps, cat) in methods.items() if cat == 'ours']
ours_points.sort(key=lambda x: x[0])

# Add reference lines
ax.axhline(y=63.2, color='#E41A1C', linestyle='--', alpha=0.3, linewidth=1)
ax.axvline(x=26.1, color='#E41A1C', linestyle='--', alpha=0.3, linewidth=1)

# Formatting
ax.set_xlabel('FPS (frames per second)', fontsize=12)
ax.set_ylabel('HOTA (%)', fontsize=12)
ax.set_title('Speed vs Accuracy Trade-off on MOT17', fontsize=14, fontweight='bold')

ax.set_xlim(0, 160)
ax.set_ylim(40, 66)

ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add annotation for best accuracy-speed trade-off
ax.annotate('Best trade-off', xy=(26.1, 63.2), xytext=(50, 64),
            fontsize=10, color='#E41A1C',
            arrowprops=dict(arrowstyle='->', color='#E41A1C', lw=1.5))

plt.tight_layout()

# Save
output_dir = '/home/muhiddin/lite/eccv2026/figures'
plt.savefig(f'{output_dir}/speed_accuracy_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"Saved speed_accuracy_tradeoff.pdf and .png to {output_dir}")
