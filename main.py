import matplotlib

# Use a non-interactive backend so the script runs headless without GUI support
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from block import Block
from bz_algorithm_logic import BZScheduler
from deposit_utils import generate_2d_deposit




def visualize_results(blocks, schedule, width, depth, history):
    """
    Plots the pit phases and the convergence history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Plot the pit shells / phases
    matrix = np.full((depth, width), -1.0)  # -1 for unmined

    for b in blocks:
        period = schedule[b.id]
        if period != -1:
            matrix[b.y, b.x] = period  # Mine value is the period index

    # Create custom colormap: Grey for unmined, Viridis for periods
    cmap = plt.cm.viridis
    cmap.set_under("lightgrey")

    im = ax1.imshow(matrix, cmap=cmap, vmin=0)
    ax1.set_title("Optimization Result: Extraction Period")
    ax1.set_xlabel("X (Block Coordinates)")
    ax1.set_ylabel("Y (Depth)")
    fig.colorbar(im, ax=ax1, label="Period Mined", ticks=range(10))

    # Annotate blocks with '.' for ore mined
    for b in blocks:
        if b.economic_value > 0 and schedule[b.id] != -1:
            ax1.text(b.x, b.y, ".", ha="center", va="center", color="white", fontsize=6)

    # 2. Plot convergence history
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

    # Save instead of showing (headless-safe)
    output_path = "bz_result.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots to {output_path}")


if __name__ == "__main__":
    # Parameters
    WIDTH = 25
    DEPTH = 15
    PERIODS = 4
    DISCOUNT_RATE = 0.10

    # Generate data
    print("Generating Block Model...")
    blocks, w, d = generate_2d_deposit(WIDTH, DEPTH)
    print(f"Model created: {len(blocks)} blocks.")

    # Define limits (tuning these makes the problem harder/easier)
    # Estimate total tonnage to set realistic constraints
    total_ore_tonnage = sum(b.tonnage for b in blocks if b.economic_value > 0)

    # Constraint: Force spreading extraction over periods
    # e.g., limit to 30% of total possible ore per period
    MINING_CAP = 30000
    PROC_CAP = total_ore_tonnage / 2.5

    print(f"Constraints :: Mining: {MINING_CAP}t/yr | Processing: {PROC_CAP:.0f}t/yr")

    # Initialize solver
    scheduler = BZScheduler(
        blocks,
        PERIODS,
        DISCOUNT_RATE,
        MINING_CAP,
        PROC_CAP,
    )

    # Run BZ algorithm
    # Note: Step size factor is sensitive. In real BZ, this is dynamic.
    final_schedule, history = scheduler.solve(max_iterations=15, step_size_factor=0.00008)

    # Visualize
    visualize_results(blocks, final_schedule, w, d, history)
