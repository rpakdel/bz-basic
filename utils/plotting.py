"""Plotting utilities for BZ scheduler."""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_results(blocks, schedule, width, depth, history):
    """
    Create visualization of optimization results.
    
    Args:
        blocks: List of Block objects
        schedule: Dict mapping block IDs to extraction periods
        width: Grid width
        depth: Grid depth
        history: List of (iteration, npv, violation) tuples
    
    Returns:
        matplotlib figure with extraction periods and convergence history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    matrix = np.full((depth, width), -1.0)
    for b in blocks:
        period = schedule[b.id]
        if period != -1:
            matrix[b.y, b.x] = period

    cmap = plt.cm.viridis
    cmap.set_under("lightgrey")
    im = ax1.imshow(matrix, cmap=cmap, vmin=0)
    ax1.set_title("Optimization Result: Extraction Period")
    ax1.set_xlabel("X (Block Coordinates)")
    ax1.set_ylabel("Y (Depth)")
    fig.colorbar(im, ax=ax1, label="Period Mined", ticks=range(10))

    for b in blocks:
        if b.economic_value > 0 and schedule[b.id] != -1:
            ax1.text(b.x, b.y, ".", ha="center", va="center", color="white", fontsize=6)

    iterations = [h[0] for h in history]
    profits = [h[1] for h in history]
    violations = [h[2] for h in history]

    ax2.set_title("BZ Convergence History")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("NPV ($)", color="tab:blue")
    ax2.plot(iterations, profits, color="tab:blue", marker="o", label="NPV")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax3 = ax2.twinx()
    ax3.set_ylabel("Constraint Violation (Tons)", color="tab:red")
    ax3.plot(iterations, violations, color="tab:red", linestyle="--", label="Violation")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    plt.tight_layout()
    return fig


def visualize_block_predecessors_3d(blocks, target_block_id, x_size, y_size, z_size):
    """
    3D visualization of a specific block and its predecessors.
    
    Shows the target block and all blocks it depends on (predecessors),
    with arrows indicating the precedence relationships.
    
    Args:
        blocks: List of Block objects with predecessors populated
        target_block_id: ID of the block to visualize
        x_size, y_size, z_size: Model dimensions
    
    Returns:
        matplotlib figure
    """
    # Find the target block
    id_to_block = {b.id: b for b in blocks}
    target_block = id_to_block.get(target_block_id)
    
    if target_block is None:
        raise ValueError(f"Block ID {target_block_id} not found")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the target block in red
    ax.scatter([target_block.x], [target_block.y], [target_block.z], 
               c='red', s=200, marker='o', label='Target Block', 
               edgecolors='black', linewidths=2)
    
    # Plot predecessors in blue
    if target_block.predecessors:
        pred_x = [p.x for p in target_block.predecessors]
        pred_y = [p.y for p in target_block.predecessors]
        pred_z = [p.z for p in target_block.predecessors]
        
        ax.scatter(pred_x, pred_y, pred_z, 
                   c='blue', s=150, marker='^', label='Predecessors',
                   edgecolors='black', linewidths=1.5)
        
        # Draw arrows from predecessors to target
        for pred in target_block.predecessors:
            ax.plot([pred.x, target_block.x], 
                   [pred.y, target_block.y], 
                   [pred.z, target_block.z],
                   'gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('X (Easting)', fontsize=11)
    ax.set_ylabel('Y (Northing)', fontsize=11)
    ax.set_zlabel('Z (Elevation)', fontsize=11)
    ax.set_title(f'Block {target_block_id} at ({target_block.x}, {target_block.y}, {target_block.z})\n'
                 f'with {len(target_block.predecessors)} Predecessor(s)',
                 fontsize=13, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, x_size - 0.5)
    ax.set_ylim(-0.5, y_size - 0.5)
    ax.set_zlim(-0.5, z_size - 0.5)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    return fig


def visualize_precedence_cross_section(blocks, y_slice, x_size, y_size, z_size):
    """
    2D cross-section visualization showing precedence relationships.
    
    Shows a slice through the Y dimension, displaying blocks in the X-Z plane
    with arrows showing the 1:3 slope constraint.
    
    Args:
        blocks: List of Block objects with predecessors populated
        y_slice: Y coordinate to slice at
        x_size, y_size, z_size: Model dimensions
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Filter blocks at this Y slice
    slice_blocks = [b for b in blocks if b.y == y_slice]
    
    if not slice_blocks:
        raise ValueError(f"No blocks found at y={y_slice}")
    
    # Create a color map based on economic value
    values = [b.economic_value for b in slice_blocks]
    vmin, vmax = min(values), max(values)
    
    # Plot blocks as squares
    for block in slice_blocks:
        # Normalize color
        if vmax > vmin:
            norm_value = (block.economic_value - vmin) / (vmax - vmin)
        else:
            norm_value = 0.5
        
        color = plt.cm.RdYlGn(norm_value)
        
        # Draw block as square
        ax.add_patch(plt.Rectangle((block.x - 0.4, block.z - 0.4), 0.8, 0.8,
                                   facecolor=color, edgecolor='black', linewidth=1))
    
    # Draw precedence arrows for blocks at this slice
    for block in slice_blocks:
        for pred in block.predecessors:
            if pred.y == y_slice:  # Only show predecessors in same slice
                # Arrow from predecessor to block
                ax.annotate('', xy=(block.x, block.z), 
                           xytext=(pred.x, pred.z),
                           arrowprops=dict(arrowstyle='->', lw=1.5, 
                                         color='blue', alpha=0.4))
    
    ax.set_xlabel('X (Easting)', fontsize=12)
    ax.set_ylabel('Z (Elevation)', fontsize=12)
    ax.set_title(f'Precedence Cross-Section at Y={y_slice}\n'
                 f'1:3 Slope Constraint (Arrows: Predecessor → Block)',
                 fontsize=13, fontweight='bold')
    
    ax.set_xlim(-0.5, x_size - 0.5)
    ax.set_ylim(-0.5, z_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Higher Z at top
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Economic Value ($)', fontsize=11)
    
    plt.tight_layout()
    return fig


def visualize_precedence_region_3d(blocks, x_range, y_range, z_range):
    """
    3D visualization of a region showing all precedence relationships.
    
    Args:
        blocks: List of Block objects with predecessors populated
        x_range: Tuple of (x_min, x_max)
        y_range: Tuple of (y_min, y_max)
        z_range: Tuple of (z_min, z_max)
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filter blocks in region
    region_blocks = [
        b for b in blocks
        if (x_range[0] <= b.x <= x_range[1] and
            y_range[0] <= b.y <= y_range[1] and
            z_range[0] <= b.z <= z_range[1])
    ]
    
    if not region_blocks:
        raise ValueError("No blocks found in specified region")
    
    # Create set of region block IDs for quick lookup
    region_ids = {b.id for b in region_blocks}
    
    # Color blocks by Z level
    z_values = [b.z for b in region_blocks]
    colors = plt.cm.viridis((np.array(z_values) - min(z_values)) / 
                            (max(z_values) - min(z_values) + 1))
    
    # Plot blocks
    ax.scatter([b.x for b in region_blocks],
               [b.y for b in region_blocks],
               [b.z for b in region_blocks],
               c=colors, s=100, marker='o', alpha=0.7,
               edgecolors='black', linewidths=1)
    
    # Draw precedence arrows (only within region)
    arrow_count = 0
    for block in region_blocks:
        for pred in block.predecessors:
            if pred.id in region_ids:  # Only draw if both blocks in region
                ax.plot([pred.x, block.x],
                       [pred.y, block.y],
                       [pred.z, block.z],
                       'gray', linestyle='--', linewidth=1, alpha=0.5)
                arrow_count += 1
    
    ax.set_xlabel('X (Easting)', fontsize=11)
    ax.set_ylabel('Y (Northing)', fontsize=11)
    ax.set_zlabel('Z (Elevation)', fontsize=11)
    ax.set_title(f'Precedence Graph Region\n'
                 f'X:[{x_range[0]}-{x_range[1]}], Y:[{y_range[0]}-{y_range[1]}], Z:[{z_range[0]}-{z_range[1]}]\n'
                 f'{len(region_blocks)} blocks, {arrow_count} precedence links',
                 fontsize=12, fontweight='bold')
    
    # Add colorbar for Z levels
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=min(z_values), vmax=max(z_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Z Level (Elevation)', fontsize=10)
    
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    return fig


def visualize_precedence_heatmap(blocks, x_size, y_size, z_size):
    """
    Heatmap showing number of predecessors per block across different Z levels.
    
    Args:
        blocks: List of Block objects with predecessors populated
        x_size, y_size, z_size: Model dimensions
    
    Returns:
        matplotlib figure with subplots for each Z level
    """
    # Calculate grid layout for subplots
    n_levels = min(z_size, 6)  # Show max 6 levels
    z_levels = np.linspace(0, z_size - 1, n_levels, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    id_to_block = {b.id: b for b in blocks}
    
    for idx, z in enumerate(z_levels):
        ax = axes[idx]
        
        # Create matrix of predecessor counts
        matrix = np.full((y_size, x_size), np.nan)
        
        for block in blocks:
            if block.z == z:
                matrix[block.y, block.x] = len(block.predecessors)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=3)
        ax.set_title(f'Z = {z}', fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='# Predecessors')
    
    fig.suptitle('Predecessor Count by Z Level\n'
                 '(0 = surface, 2-3 = typical interior blocks)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig
