"""
Generate LITE++ Architecture Diagram for ECCV 2026 Paper

Creates a publication-quality figure showing:
1. YOLO backbone with multi-layer feature extraction
2. RoIAlign-based feature pooling
3. Multi-scale feature fusion module
4. Adaptive threshold learning module
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.patches import ConnectionPatch
import numpy as np

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Colors
COLORS = {
    'input': '#E8F4FD',
    'backbone': '#B8D4E8',
    'layer4': '#FFE4B5',
    'layer9': '#FFDAB9',
    'layer14': '#FFB6C1',
    'roialign': '#E6E6FA',
    'fusion': '#98FB98',
    'atl': '#DDA0DD',
    'output': '#F0FFF0',
    'arrow': '#404040',
    'text': '#202020',
    'border': '#606060',
}


def draw_rounded_box(ax, x, y, width, height, label, color, fontsize=8,
                     text_color='black', bold=False):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor=COLORS['border'], linewidth=1.2
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True)

    return box


def draw_arrow(ax, start, end, color=None, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=12,
        color=color,
        linewidth=1.2,
        connectionstyle=connectionstyle
    )
    ax.add_patch(arrow)
    return arrow


def draw_feature_map(ax, x, y, w, h, channels, label, color):
    """Draw a 3D-style feature map representation."""
    depth = 0.15

    # Front face
    rect = Rectangle((x, y), w, h, facecolor=color, edgecolor=COLORS['border'], linewidth=1)
    ax.add_patch(rect)

    # Top face (parallelogram)
    top_x = [x, x + depth, x + w + depth, x + w]
    top_y = [y + h, y + h + depth, y + h + depth, y + h]
    ax.fill(top_x, top_y, facecolor=color, edgecolor=COLORS['border'], linewidth=1, alpha=0.8)

    # Right face (parallelogram)
    right_x = [x + w, x + w + depth, x + w + depth, x + w]
    right_y = [y, y + depth, y + h + depth, y + h]
    ax.fill(right_x, right_y, facecolor=color, edgecolor=COLORS['border'], linewidth=1, alpha=0.6)

    # Label
    ax.text(x + w/2, y - 0.12, label, ha='center', va='top', fontsize=7)
    ax.text(x + w/2, y + h + 0.08, f'C={channels}', ha='center', va='bottom', fontsize=6, style='italic')


def create_architecture_diagram(output_path):
    """Create the main architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 6.2, 'LITE++ Architecture', ha='center', va='bottom',
            fontsize=14, weight='bold', color=COLORS['text'])

    # ===== Input Image =====
    draw_rounded_box(ax, 0.5, 3, 0.9, 1.2, 'Input\nImage', COLORS['input'], fontsize=8)

    # ===== YOLO Backbone =====
    # Main backbone box
    backbone_box = FancyBboxPatch(
        (1.3, 1.0), 2.0, 4.0,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['backbone'], edgecolor=COLORS['border'], linewidth=1.5
    )
    ax.add_patch(backbone_box)
    ax.text(2.3, 5.15, 'YOLOv8 Backbone', ha='center', va='bottom', fontsize=9, weight='bold')

    # Backbone layers
    layer_y = [4.2, 3.0, 1.8]
    layer_labels = ['Layer 4', 'Layer 9', 'Layer 14']
    layer_colors = [COLORS['layer4'], COLORS['layer9'], COLORS['layer14']]
    layer_info = ['64ch, H/4', '256ch, H/16', '192ch, H/32']

    for i, (y, label, color, info) in enumerate(zip(layer_y, layer_labels, layer_colors, layer_info)):
        box = FancyBboxPatch(
            (1.5, y - 0.35), 1.6, 0.7,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            facecolor=color, edgecolor=COLORS['border'], linewidth=1
        )
        ax.add_patch(box)
        ax.text(2.3, y, label, ha='center', va='center', fontsize=8, weight='bold')
        ax.text(2.3, y - 0.22, info, ha='center', va='center', fontsize=6, style='italic')

    # Arrow from input to backbone
    draw_arrow(ax, (0.95, 3), (1.3, 3))

    # ===== Detection Head (simplified) =====
    draw_rounded_box(ax, 2.3, 0.4, 1.4, 0.5, 'Detection Head', '#D3D3D3', fontsize=7)
    draw_arrow(ax, (2.3, 1.45), (2.3, 0.65))

    # ===== Bounding Boxes =====
    draw_rounded_box(ax, 4.2, 0.4, 1.0, 0.5, 'Boxes', '#F5F5F5', fontsize=7)
    draw_arrow(ax, (3.0, 0.4), (3.7, 0.4))

    # ===== RoIAlign Module =====
    roialign_box = FancyBboxPatch(
        (3.8, 1.5), 1.4, 3.2,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=COLORS['roialign'], edgecolor=COLORS['border'], linewidth=1.5
    )
    ax.add_patch(roialign_box)
    ax.text(4.5, 4.85, 'RoIAlign', ha='center', va='bottom', fontsize=9, weight='bold')

    # RoIAlign sub-boxes
    roi_y = [4.0, 3.0, 2.0]
    for i, y in enumerate(roi_y):
        box = FancyBboxPatch(
            (4.0, y - 0.3), 1.0, 0.6,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor='white', edgecolor=COLORS['border'], linewidth=0.8
        )
        ax.add_patch(box)
        ax.text(4.5, y, f'f{["4", "9", "14"][i]}', ha='center', va='center', fontsize=7)

    # Arrows from backbone layers to RoIAlign
    for i, y in enumerate(layer_y):
        draw_arrow(ax, (3.1, y), (4.0, roi_y[i]), connectionstyle='arc3,rad=0')

    # Arrow from boxes to RoIAlign
    draw_arrow(ax, (4.5, 0.65), (4.5, 1.5), connectionstyle='arc3,rad=0')

    # ===== Feature Fusion Module =====
    fusion_box = FancyBboxPatch(
        (5.6, 2.0), 1.8, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=COLORS['fusion'], edgecolor=COLORS['border'], linewidth=1.5
    )
    ax.add_patch(fusion_box)
    ax.text(6.5, 4.65, 'Feature Fusion', ha='center', va='bottom', fontsize=9, weight='bold')

    # Fusion internals
    ax.text(6.5, 4.0, 'Instance-Adaptive', ha='center', va='center', fontsize=7)
    ax.text(6.5, 3.6, 'Attention', ha='center', va='center', fontsize=7, weight='bold')

    # Attention weights visualization
    ax.text(6.5, 3.0, r'$\alpha_i = \mathrm{softmax}(W_\alpha \cdot f)$',
            ha='center', va='center', fontsize=7, style='italic')

    # Concat visualization
    concat_box = FancyBboxPatch(
        (5.9, 2.2), 1.2, 0.5,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor='white', edgecolor=COLORS['border'], linewidth=0.8
    )
    ax.add_patch(concat_box)
    ax.text(6.5, 2.45, 'Concat + MLP', ha='center', va='center', fontsize=6)

    # Arrows from RoIAlign to Fusion
    draw_arrow(ax, (5.0, 3.0), (5.6, 3.0))

    # ===== Adaptive Threshold Module =====
    atl_box = FancyBboxPatch(
        (5.6, 0.2), 1.8, 1.4,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=COLORS['atl'], edgecolor=COLORS['border'], linewidth=1.5
    )
    ax.add_patch(atl_box)
    ax.text(6.5, 1.75, 'Adaptive Threshold', ha='center', va='bottom', fontsize=8, weight='bold')

    ax.text(6.5, 1.1, 'Scene Encoder', ha='center', va='center', fontsize=7)
    ax.text(6.5, 0.7, 'GAP + MLP', ha='center', va='center', fontsize=6)
    ax.text(6.5, 0.4, r'$\tau \in [0.01, 0.50]$', ha='center', va='center', fontsize=6, style='italic')

    # Arrow from Layer 14 to ATL (global features)
    draw_arrow(ax, (3.1, 1.8), (3.5, 0.9), connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (3.5, 0.9), (5.6, 0.9), connectionstyle='arc3,rad=0')
    ax.text(4.3, 1.15, 'Global\nFeatures', ha='center', va='center', fontsize=6, style='italic')

    # ===== Output Features =====
    output_box = FancyBboxPatch(
        (7.8, 2.5), 1.2, 1.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['output'], edgecolor=COLORS['border'], linewidth=1.5
    )
    ax.add_patch(output_box)
    ax.text(8.4, 4.15, 'Output', ha='center', va='bottom', fontsize=9, weight='bold')
    ax.text(8.4, 3.5, 'ReID\nFeatures', ha='center', va='center', fontsize=8)
    ax.text(8.4, 2.9, r'$f_i \in \mathbb{R}^{128}$', ha='center', va='center', fontsize=7, style='italic')

    # Arrow from Fusion to Output
    draw_arrow(ax, (7.4, 3.25), (7.8, 3.25))

    # ===== Threshold Output =====
    tau_box = FancyBboxPatch(
        (7.8, 0.5), 1.2, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor='#FFF0F5', edgecolor=COLORS['border'], linewidth=1.2
    )
    ax.add_patch(tau_box)
    ax.text(8.4, 0.9, r'Threshold $\tau$', ha='center', va='center', fontsize=8)
    ax.text(8.4, 0.6, '(EMA smoothed)', ha='center', va='center', fontsize=6, style='italic')

    # Arrow from ATL to Threshold
    draw_arrow(ax, (7.4, 0.9), (7.8, 0.9))

    # ===== Association Module =====
    assoc_box = FancyBboxPatch(
        (9.2, 1.5), 1.0, 2.5,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='#F0F8FF', edgecolor=COLORS['border'], linewidth=1.2
    )
    ax.add_patch(assoc_box)
    ax.text(9.7, 3.5, 'Tracker', ha='center', va='center', fontsize=8, weight='bold')
    ax.text(9.7, 3.0, 'Cascade', ha='center', va='center', fontsize=7)
    ax.text(9.7, 2.6, 'Matching', ha='center', va='center', fontsize=7)
    ax.text(9.7, 2.0, '+', ha='center', va='center', fontsize=10)
    ax.text(9.7, 1.6, 'Kalman', ha='center', va='center', fontsize=7)

    # Arrows to Association
    draw_arrow(ax, (9.0, 3.25), (9.2, 2.8), connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (9.0, 0.9), (9.2, 1.8), connectionstyle='arc3,rad=0.2')

    # ===== Legend =====
    legend_y = 5.8
    legend_items = [
        ('Feature Extraction', COLORS['backbone']),
        ('RoIAlign Pooling', COLORS['roialign']),
        ('Multi-Scale Fusion', COLORS['fusion']),
        ('Threshold Learning', COLORS['atl']),
    ]

    for i, (label, color) in enumerate(legend_items):
        x = 1.5 + i * 2.3
        rect = Rectangle((x - 0.15, legend_y - 0.08), 0.3, 0.16,
                         facecolor=color, edgecolor=COLORS['border'], linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.25, legend_y, label, ha='left', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Architecture diagram saved to {output_path}")


def create_fusion_detail_diagram(output_path):
    """Create detailed fusion module diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(3, 4.2, 'Instance-Adaptive Attention Fusion', ha='center', va='bottom',
            fontsize=12, weight='bold')

    # Input features
    y_positions = [3.2, 2.2, 1.2]
    labels = [r'$f_i^{(4)}$ (64ch)', r'$f_i^{(9)}$ (256ch)', r'$f_i^{(14)}$ (192ch)']
    colors = [COLORS['layer4'], COLORS['layer9'], COLORS['layer14']]

    for y, label, color in zip(y_positions, labels, colors):
        draw_rounded_box(ax, 0.8, y, 1.2, 0.6, label, color, fontsize=8)

    # Projection layers
    for i, y in enumerate(y_positions):
        draw_rounded_box(ax, 2.3, y, 0.8, 0.5, f'Proj{[4,9,14][i]}', '#E0E0E0', fontsize=7)
        draw_arrow(ax, (1.4, y), (1.9, y))

    # Concatenation for attention
    draw_rounded_box(ax, 3.5, 2.2, 0.9, 1.8, 'Concat', '#F5F5F5', fontsize=8)
    for y in y_positions:
        draw_arrow(ax, (2.7, y), (3.05, 2.2), connectionstyle='arc3,rad=0')

    # Attention computation
    draw_rounded_box(ax, 4.7, 2.2, 1.0, 0.8, r'$W_\alpha$' + '\nSoftmax', '#DDA0DD', fontsize=7)
    draw_arrow(ax, (3.95, 2.2), (4.2, 2.2))

    # Attention weights
    ax.text(4.7, 3.3, r'$\alpha_i^{(l)}$', ha='center', va='center', fontsize=9, style='italic')
    draw_arrow(ax, (4.7, 2.6), (4.7, 3.1))

    # Weighted sum
    draw_rounded_box(ax, 5.8, 2.2, 0.8, 0.6, r'$\Sigma$', COLORS['fusion'], fontsize=12)
    draw_arrow(ax, (5.2, 2.2), (5.4, 2.2))

    # Output
    ax.text(5.8, 1.3, r'$f_i \in \mathbb{R}^{128}$', ha='center', va='center', fontsize=9, style='italic')
    draw_arrow(ax, (5.8, 1.9), (5.8, 1.5))

    # Dashed lines showing attention weighting
    for y in y_positions:
        draw_arrow(ax, (2.7, y), (5.4, 2.2 + (y - 2.2) * 0.1),
                  style='->', color='#A0A0A0')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Fusion detail diagram saved to {output_path}")


if __name__ == '__main__':
    import os

    output_dir = '../eccv2026/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate main architecture diagram
    create_architecture_diagram(os.path.join(output_dir, 'architecture.pdf'))

    # Generate fusion detail diagram
    create_fusion_detail_diagram(os.path.join(output_dir, 'fusion_detail.pdf'))

    print("\nAll diagrams generated successfully!")
