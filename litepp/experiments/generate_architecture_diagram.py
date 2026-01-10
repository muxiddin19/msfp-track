"""
Generate Publication-Quality Architecture Diagram for LITE++

Creates a detailed architecture figure suitable for ECCV 2026 submission.
The diagram shows:
1. YOLOv8 backbone with multi-scale feature extraction
2. RoIAlign-based feature pooling
3. Instance-adaptive attention fusion
4. Adaptive Threshold Learning with EMA
5. Association pipeline

References architecture styles from:
- ByteTrack (ECCV 2022)
- FairMOT (IJCV 2021)
- StrongSORT (TMM 2023)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as path_effects
import numpy as np

# Set up publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette (colorblind-friendly)
COLORS = {
    'backbone': '#4ECDC4',      # Teal
    'layer4': '#95E1D3',        # Light teal
    'layer9': '#F38181',        # Coral
    'layer14': '#FCE38A',       # Yellow
    'roialign': '#AA96DA',      # Purple
    'fusion': '#FF6B6B',        # Red
    'atl': '#4ECDC4',           # Teal
    'output': '#95E1D3',        # Light green
    'arrow': '#2C3E50',         # Dark blue-gray
    'text': '#2C3E50',
    'box_edge': '#34495E',
    'detection': '#E74C3C',     # Red for detection boxes
}


def draw_rounded_box(ax, x, y, width, height, text, color, text_color='white',
                     fontsize=9, fontweight='bold', alpha=0.9, edge_color=None):
    """Draw a rounded rectangle with centered text."""
    if edge_color is None:
        edge_color = color

    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor=edge_color,
        linewidth=1.5, alpha=alpha, zorder=2
    )
    ax.add_patch(box)

    # Add text with shadow for better readability
    txt = ax.text(x, y, text, ha='center', va='center',
                  fontsize=fontsize, fontweight=fontweight,
                  color=text_color, zorder=3)
    txt.set_path_effects([
        path_effects.withStroke(linewidth=2, foreground='white', alpha=0.3)
    ])

    return box


def draw_arrow(ax, start, end, color=COLORS['arrow'], style='->',
               connectionstyle='arc3,rad=0', linewidth=1.5):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color,
        linewidth=linewidth,
        mutation_scale=12,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow


def draw_feature_map(ax, x, y, width, height, channels, label, color):
    """Draw a 3D-style feature map representation."""
    depth = 0.15
    # Back face
    back = plt.Polygon([
        (x - width/2 + depth, y - height/2 + depth),
        (x + width/2 + depth, y - height/2 + depth),
        (x + width/2 + depth, y + height/2 + depth),
        (x - width/2 + depth, y + height/2 + depth)
    ], facecolor=color, edgecolor=COLORS['box_edge'],
       linewidth=1, alpha=0.5, zorder=1)
    ax.add_patch(back)

    # Front face
    front = plt.Polygon([
        (x - width/2, y - height/2),
        (x + width/2, y - height/2),
        (x + width/2, y + height/2),
        (x - width/2, y + height/2)
    ], facecolor=color, edgecolor=COLORS['box_edge'],
       linewidth=1.5, alpha=0.9, zorder=2)
    ax.add_patch(front)

    # Top face
    top = plt.Polygon([
        (x - width/2, y + height/2),
        (x - width/2 + depth, y + height/2 + depth),
        (x + width/2 + depth, y + height/2 + depth),
        (x + width/2, y + height/2)
    ], facecolor=color, edgecolor=COLORS['box_edge'],
       linewidth=1, alpha=0.7, zorder=2)
    ax.add_patch(top)

    # Right face
    right = plt.Polygon([
        (x + width/2, y - height/2),
        (x + width/2 + depth, y - height/2 + depth),
        (x + width/2 + depth, y + height/2 + depth),
        (x + width/2, y + height/2)
    ], facecolor=color, edgecolor=COLORS['box_edge'],
       linewidth=1, alpha=0.7, zorder=2)
    ax.add_patch(right)

    # Label
    ax.text(x, y - height/2 - 0.15, label, ha='center', va='top',
            fontsize=8, color=COLORS['text'])
    ax.text(x, y, f'C={channels}', ha='center', va='center',
            fontsize=7, color='white', fontweight='bold')


def create_architecture_diagram():
    """Create the main architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 5.7, 'LITE++ Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['text'])

    # =====================
    # 1. Input Image
    # =====================
    img_x, img_y = 0.8, 3
    img_rect = FancyBboxPatch(
        (img_x - 0.5, img_y - 0.7), 1.0, 1.4,
        boxstyle="round,pad=0.02",
        facecolor='#ECF0F1', edgecolor=COLORS['box_edge'],
        linewidth=1.5, zorder=2
    )
    ax.add_patch(img_rect)
    ax.text(img_x, img_y, 'Input\nImage', ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLORS['text'])

    # Detection boxes on image
    for dy in [-0.3, 0.1, 0.4]:
        det_box = Rectangle((img_x - 0.25, img_y + dy - 0.1), 0.3, 0.25,
                            facecolor='none', edgecolor=COLORS['detection'],
                            linewidth=1.5, linestyle='-', zorder=3)
        ax.add_patch(det_box)

    # =====================
    # 2. YOLOv8 Backbone
    # =====================
    backbone_x = 2.5
    draw_rounded_box(ax, backbone_x, 3, 1.2, 2.5, 'YOLOv8\nBackbone',
                     '#3498DB', fontsize=10)

    # Arrow from input to backbone
    draw_arrow(ax, (img_x + 0.5, 3), (backbone_x - 0.65, 3))

    # =====================
    # 3. Multi-Scale Feature Maps
    # =====================
    # Layer 4
    layer4_x, layer4_y = 4.2, 4.3
    draw_feature_map(ax, layer4_x, layer4_y, 0.7, 0.7, 64, 'Layer 4\n(H/4)', COLORS['layer4'])

    # Layer 9
    layer9_x, layer9_y = 4.2, 3.0
    draw_feature_map(ax, layer9_x, layer9_y, 0.55, 0.55, 256, 'Layer 9\n(H/16)', COLORS['layer9'])

    # Layer 14
    layer14_x, layer14_y = 4.2, 1.7
    draw_feature_map(ax, layer14_x, layer14_y, 0.45, 0.45, 192, 'Layer 14\n(H/32)', COLORS['layer14'])

    # Arrows from backbone to layers
    draw_arrow(ax, (backbone_x + 0.65, 3.8), (layer4_x - 0.5, layer4_y),
               connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (backbone_x + 0.65, 3), (layer9_x - 0.5, layer9_y))
    draw_arrow(ax, (backbone_x + 0.65, 2.2), (layer14_x - 0.5, layer14_y),
               connectionstyle='arc3,rad=0.2')

    # =====================
    # 4. RoIAlign Modules
    # =====================
    roi_x = 5.8

    draw_rounded_box(ax, roi_x, layer4_y, 0.9, 0.4, 'RoIAlign',
                     COLORS['roialign'], fontsize=8)
    draw_rounded_box(ax, roi_x, layer9_y, 0.9, 0.4, 'RoIAlign',
                     COLORS['roialign'], fontsize=8)
    draw_rounded_box(ax, roi_x, layer14_y, 0.9, 0.4, 'RoIAlign',
                     COLORS['roialign'], fontsize=8)

    # Arrows to RoIAlign
    draw_arrow(ax, (layer4_x + 0.5, layer4_y), (roi_x - 0.5, layer4_y))
    draw_arrow(ax, (layer9_x + 0.5, layer9_y), (roi_x - 0.5, layer9_y))
    draw_arrow(ax, (layer14_x + 0.5, layer14_y), (roi_x - 0.5, layer14_y))

    # Detection boxes input to RoIAlign
    ax.text(roi_x, 5.0, 'Detection\nBoxes', ha='center', va='center',
            fontsize=7, color=COLORS['text'], style='italic')
    draw_arrow(ax, (roi_x, 4.7), (roi_x, layer4_y + 0.25), style='->')

    # =====================
    # 5. Feature Vectors
    # =====================
    feat_x = 7.2

    # Feature vector representations
    for y, label in [(layer4_y, 'f⁴'), (layer9_y, 'f⁹'), (layer14_y, 'f¹⁴')]:
        vec = FancyBboxPatch(
            (feat_x - 0.15, y - 0.25), 0.3, 0.5,
            boxstyle="round,pad=0.02",
            facecolor='#BDC3C7', edgecolor=COLORS['box_edge'],
            linewidth=1, zorder=2
        )
        ax.add_patch(vec)
        ax.text(feat_x, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold')

    # Arrows from RoIAlign to features
    draw_arrow(ax, (roi_x + 0.5, layer4_y), (feat_x - 0.2, layer4_y))
    draw_arrow(ax, (roi_x + 0.5, layer9_y), (feat_x - 0.2, layer9_y))
    draw_arrow(ax, (roi_x + 0.5, layer14_y), (feat_x - 0.2, layer14_y))

    # =====================
    # 6. Instance-Adaptive Attention Fusion
    # =====================
    fusion_x, fusion_y = 9.0, 3.0

    # Fusion module box
    fusion_box = FancyBboxPatch(
        (fusion_x - 0.7, fusion_y - 1.0), 1.4, 2.0,
        boxstyle="round,pad=0.05",
        facecolor=COLORS['fusion'], edgecolor=COLORS['box_edge'],
        linewidth=2, alpha=0.9, zorder=2
    )
    ax.add_patch(fusion_box)

    ax.text(fusion_x, fusion_y + 0.5, 'Instance-\nAdaptive', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    ax.text(fusion_x, fusion_y - 0.1, 'Attention', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    ax.text(fusion_x, fusion_y - 0.6, 'Fusion', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')

    # Arrows from features to fusion
    draw_arrow(ax, (feat_x + 0.2, layer4_y), (fusion_x - 0.75, fusion_y + 0.7),
               connectionstyle='arc3,rad=-0.15')
    draw_arrow(ax, (feat_x + 0.2, layer9_y), (fusion_x - 0.75, fusion_y))
    draw_arrow(ax, (feat_x + 0.2, layer14_y), (fusion_x - 0.75, fusion_y - 0.7),
               connectionstyle='arc3,rad=0.15')

    # Attention weights annotation
    ax.text(8.0, 4.2, 'α = softmax(Wα · [f⁴; f⁹; f¹⁴])', ha='center', va='center',
            fontsize=7, color=COLORS['text'], style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # =====================
    # 7. Output Features
    # =====================
    out_feat_x = 10.5

    # Fused feature vector
    fused_vec = FancyBboxPatch(
        (out_feat_x - 0.2, fusion_y - 0.4), 0.4, 0.8,
        boxstyle="round,pad=0.02",
        facecolor=COLORS['output'], edgecolor=COLORS['box_edge'],
        linewidth=1.5, zorder=2
    )
    ax.add_patch(fused_vec)
    ax.text(out_feat_x, fusion_y, 'fᵢ', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['text'])
    ax.text(out_feat_x, fusion_y - 0.7, 'D=128', ha='center', va='top',
            fontsize=7, color=COLORS['text'])

    draw_arrow(ax, (fusion_x + 0.75, fusion_y), (out_feat_x - 0.25, fusion_y))

    # =====================
    # 8. Adaptive Threshold Learning
    # =====================
    atl_x, atl_y = 9.0, 0.3

    # ATL module
    atl_box = FancyBboxPatch(
        (atl_x - 1.0, atl_y - 0.5), 2.0, 1.0,
        boxstyle="round,pad=0.05",
        facecolor=COLORS['atl'], edgecolor=COLORS['box_edge'],
        linewidth=2, alpha=0.9, zorder=2
    )
    ax.add_patch(atl_box)

    ax.text(atl_x, atl_y + 0.15, 'Adaptive Threshold', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    ax.text(atl_x, atl_y - 0.2, 'Learning (ATL)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

    # Arrow from Layer 14 to ATL
    draw_arrow(ax, (layer14_x + 0.5, layer14_y - 0.3), (atl_x - 1.0, atl_y + 0.3),
               connectionstyle='arc3,rad=0.3')
    ax.text(6.5, 0.8, 'GAP(F¹⁴)', ha='center', va='center',
            fontsize=7, color=COLORS['text'], style='italic')

    # EMA annotation
    ax.text(atl_x + 1.5, atl_y, 'τₜ = β·τₜ₋₁ + (1-β)·τᵣₐᵥ', ha='left', va='center',
            fontsize=7, color=COLORS['text'], style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Threshold output
    thresh_x = 11.5
    thresh_box = FancyBboxPatch(
        (thresh_x - 0.3, atl_y - 0.3), 0.6, 0.6,
        boxstyle="round,pad=0.02",
        facecolor='#F39C12', edgecolor=COLORS['box_edge'],
        linewidth=1.5, zorder=2
    )
    ax.add_patch(thresh_box)
    ax.text(thresh_x, atl_y, 'τ', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(thresh_x, atl_y - 0.6, 'Threshold', ha='center', va='top',
            fontsize=7, color=COLORS['text'])

    draw_arrow(ax, (atl_x + 1.05, atl_y), (thresh_x - 0.35, atl_y))

    # =====================
    # 9. Association Module
    # =====================
    assoc_x, assoc_y = 12.5, 2.0

    draw_rounded_box(ax, assoc_x, assoc_y, 1.4, 1.8, 'Data\nAssociation',
                     '#9B59B6', fontsize=9)

    # Arrows to association
    draw_arrow(ax, (out_feat_x + 0.25, fusion_y), (assoc_x - 0.75, assoc_y + 0.5),
               connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (thresh_x + 0.35, atl_y), (assoc_x - 0.75, assoc_y - 0.5),
               connectionstyle='arc3,rad=0.2')

    # Labels for inputs
    ax.text(11.5, 2.8, 'Appearance', ha='center', va='center',
            fontsize=7, color=COLORS['text'], style='italic')
    ax.text(12.0, 0.8, 'Confidence', ha='center', va='center',
            fontsize=7, color=COLORS['text'], style='italic')

    # =====================
    # 10. Output Tracks
    # =====================
    track_x = 13.8

    draw_rounded_box(ax, track_x, assoc_y, 0.8, 1.0, 'Tracks',
                     '#27AE60', fontsize=9)

    draw_arrow(ax, (assoc_x + 0.75, assoc_y), (track_x - 0.45, assoc_y))

    # =====================
    # Legend
    # =====================
    legend_y = -0.3
    legend_items = [
        ('Multi-Scale Features', [COLORS['layer4'], COLORS['layer9'], COLORS['layer14']]),
        ('RoIAlign Pooling', COLORS['roialign']),
        ('Attention Fusion', COLORS['fusion']),
        ('Threshold Learning', COLORS['atl']),
    ]

    legend_x = 2.0
    for i, (label, color) in enumerate(legend_items):
        x = legend_x + i * 3.2
        if isinstance(color, list):
            for j, c in enumerate(color):
                box = Rectangle((x - 0.3 + j*0.25, legend_y - 0.1), 0.2, 0.2,
                               facecolor=c, edgecolor=COLORS['box_edge'], linewidth=1)
                ax.add_patch(box)
            ax.text(x + 0.6, legend_y, label, ha='left', va='center',
                   fontsize=7, color=COLORS['text'])
        else:
            box = Rectangle((x - 0.15, legend_y - 0.1), 0.3, 0.2,
                           facecolor=color, edgecolor=COLORS['box_edge'], linewidth=1)
            ax.add_patch(box)
            ax.text(x + 0.25, legend_y, label, ha='left', va='center',
                   fontsize=7, color=COLORS['text'])

    return fig, ax


def create_detailed_fusion_diagram():
    """Create a detailed diagram of the fusion mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(4, 4.7, 'Instance-Adaptive Attention Fusion', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['text'])

    # Input features
    for i, (y, label, ch) in enumerate([(3.5, 'f⁴', 64), (2.5, 'f⁹', 256), (1.5, 'f¹⁴', 192)]):
        box = FancyBboxPatch((0.3, y - 0.3), 0.8, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=['#95E1D3', '#F38181', '#FCE38A'][i],
                             edgecolor=COLORS['box_edge'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(0.7, y, f'{label}\n({ch})', ha='center', va='center',
                fontsize=8, fontweight='bold')

    # Projection layers
    for i, y in enumerate([3.5, 2.5, 1.5]):
        box = FancyBboxPatch((1.8, y - 0.25), 1.0, 0.5,
                             boxstyle="round,pad=0.02",
                             facecolor='#BDC3C7', edgecolor=COLORS['box_edge'])
        ax.add_patch(box)
        ax.text(2.3, y, f'Proj{["⁴","⁹","¹⁴"][i]}\n→128', ha='center', va='center',
                fontsize=7)
        draw_arrow(ax, (1.15, y), (1.75, y))

    # Concatenation for attention
    concat_box = FancyBboxPatch((3.5, 2.0), 1.2, 1.2,
                                boxstyle="round,pad=0.02",
                                facecolor='#AA96DA', edgecolor=COLORS['box_edge'],
                                linewidth=1.5)
    ax.add_patch(concat_box)
    ax.text(4.1, 2.6, 'Concat\n+\nAttention', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')

    # Arrows to concat
    for y in [3.5, 2.5, 1.5]:
        draw_arrow(ax, (2.85, y), (3.45, 2.6))

    # Attention weights
    ax.text(4.1, 3.8, 'α = softmax(Wα·x)', ha='center', va='center',
            fontsize=7, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Weighted sum
    weighted_box = FancyBboxPatch((5.3, 2.2), 1.0, 0.8,
                                  boxstyle="round,pad=0.02",
                                  facecolor=COLORS['fusion'], edgecolor=COLORS['box_edge'],
                                  linewidth=1.5)
    ax.add_patch(weighted_box)
    ax.text(5.8, 2.6, 'Σαᵢ·fᵢ', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    draw_arrow(ax, (4.75, 2.6), (5.25, 2.6))

    # Output
    out_box = FancyBboxPatch((6.8, 2.3), 0.6, 0.6,
                             boxstyle="round,pad=0.02",
                             facecolor=COLORS['output'], edgecolor=COLORS['box_edge'],
                             linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(7.1, 2.6, 'fᵢ', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(7.1, 1.8, 'D=128', ha='center', va='top', fontsize=7)

    draw_arrow(ax, (6.35, 2.6), (6.75, 2.6))

    return fig, ax


if __name__ == '__main__':
    import os

    # Create output directory
    output_dir = '/home/muhiddin/lite/eccv2026/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate main architecture diagram
    print("Generating main architecture diagram...")
    fig1, ax1 = create_architecture_diagram()
    fig1.savefig(f'{output_dir}/architecture.pdf', format='pdf', bbox_inches='tight')
    fig1.savefig(f'{output_dir}/architecture.png', format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to {output_dir}/architecture.pdf")

    # Generate fusion detail diagram
    print("Generating fusion detail diagram...")
    fig2, ax2 = create_detailed_fusion_diagram()
    fig2.savefig(f'{output_dir}/fusion_detail.pdf', format='pdf', bbox_inches='tight')
    fig2.savefig(f'{output_dir}/fusion_detail.png', format='png', bbox_inches='tight', dpi=300)
    print(f"Saved to {output_dir}/fusion_detail.pdf")

    print("\nArchitecture diagrams generated successfully!")
    plt.close('all')
