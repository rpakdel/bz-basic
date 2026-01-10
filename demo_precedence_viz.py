"""
Demonstration of precedence graph visualizations.

Run this to see various ways to visualize the precedence relationships
in a block model with 1:3 slope constraints.
"""

import matplotlib.pyplot as plt

from deposit_utils import read_block_model_csv
from utils.plotting import (
    visualize_block_predecessors_3d,
    visualize_precedence_cross_section,
    visualize_precedence_heatmap,
    visualize_precedence_region_3d,
)


def demo_single_block_visualization():
    """Demonstrate visualization of a single block and its predecessors."""
    print("Loading block model...")
    blocks, x_size, y_size, z_size = read_block_model_csv(
        "data/block_model_25_25_25_2026_01_10_03_26.csv"
    )
    
    print(f"Loaded {len(blocks)} blocks from {x_size}x{y_size}x{z_size} model")
    
    # Visualize an interior block (should have 3 predecessors)
    target_block_id = 1000  # Example block
    target_block = [b for b in blocks if b.id == target_block_id][0]
    
    print(f"\nVisualizing block {target_block_id} at ({target_block.x}, {target_block.y}, {target_block.z})")
    print(f"This block has {len(target_block.predecessors)} predecessors")
    
    fig = visualize_block_predecessors_3d(blocks, target_block_id, x_size, y_size, z_size)
    plt.savefig("precedence_single_block.png", dpi=150, bbox_inches='tight')
    print("Saved: precedence_single_block.png")
    plt.close()


def demo_cross_section_visualization():
    """Demonstrate 2D cross-section showing slope constraints."""
    print("\n" + "="*60)
    print("Loading block model for cross-section...")
    blocks, x_size, y_size, z_size = read_block_model_csv(
        "data/block_model_25_25_25_2026_01_10_03_26.csv"
    )
    
    # Show cross-section at middle Y
    y_slice = y_size // 2
    print(f"Creating cross-section at Y = {y_slice}")
    
    fig = visualize_precedence_cross_section(blocks, y_slice, x_size, y_size, z_size)
    plt.savefig("precedence_cross_section.png", dpi=150, bbox_inches='tight')
    print("Saved: precedence_cross_section.png")
    plt.close()


def demo_region_visualization():
    """Demonstrate 3D visualization of a small region."""
    print("\n" + "="*60)
    print("Loading block model for region visualization...")
    blocks, x_size, y_size, z_size = read_block_model_csv(
        "data/block_model_25_25_25_2026_01_10_03_26.csv"
    )
    
    # Show a small 5x5x5 region in the center
    x_center, y_center, z_center = x_size // 2, y_size // 2, z_size // 2
    
    x_range = (x_center - 2, x_center + 2)
    y_range = (y_center - 2, y_center + 2)
    z_range = (z_center - 2, z_center + 2)
    
    print(f"Visualizing region: X{x_range}, Y{y_range}, Z{z_range}")
    
    fig = visualize_precedence_region_3d(blocks, x_range, y_range, z_range)
    plt.savefig("precedence_region_3d.png", dpi=150, bbox_inches='tight')
    print("Saved: precedence_region_3d.png")
    plt.close()


def demo_heatmap_visualization():
    """Demonstrate heatmap of predecessor counts by Z level."""
    print("\n" + "="*60)
    print("Loading block model for heatmap...")
    blocks, x_size, y_size, z_size = read_block_model_csv(
        "data/block_model_25_25_25_2026_01_10_03_26.csv"
    )
    
    print("Creating heatmap of predecessor counts...")
    
    fig = visualize_precedence_heatmap(blocks, x_size, y_size, z_size)
    plt.savefig("precedence_heatmap.png", dpi=150, bbox_inches='tight')
    print("Saved: precedence_heatmap.png")
    plt.close()


def demo_all_visualizations():
    """Run all visualization demos."""
    print("="*60)
    print("PRECEDENCE GRAPH VISUALIZATION DEMO")
    print("="*60)
    
    demo_single_block_visualization()
    demo_cross_section_visualization()
    demo_region_visualization()
    demo_heatmap_visualization()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("Generated files:")
    print("  - precedence_single_block.png")
    print("  - precedence_cross_section.png")
    print("  - precedence_region_3d.png")
    print("  - precedence_heatmap.png")
    print("="*60)


if __name__ == "__main__":
    demo_all_visualizations()
