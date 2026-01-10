import matplotlib
matplotlib.use("Agg")  # headless-friendly backend
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from bz_algorithm_logic import Block, BZScheduler


def generate_2d_deposit(width, depth, rng):
    """
    Build a synthetic 2D block model with an ore blob in the center/bottom.
    """
    blocks = []
    grid = {}
    block_id = 0

    for y in range(depth):
        for x in range(width):
            center_x = width // 2
            center_y = depth // 2 + 2
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            tonnage = 1000

            if dist < width / 4:
                grade = 1.5 + rng.normal(0, 0.2)
                val_per_ton = (grade * 50) - 20
                value = val_per_ton * tonnage
            elif dist < width / 2.5:
                grade = 0.6 + rng.normal(0, 0.1)
                val_per_ton = (grade * 50) - 20
                value = val_per_ton * tonnage
            else:
                grade = 0.0
                value = -5 * tonnage

            b = Block(block_id, x, y, 0, tonnage, grade, value)
            blocks.append(b)
            grid[(x, y)] = b
            block_id += 1

    for b in blocks:
        if b.y > 0:
            preds = [(b.x - 1, b.y - 1), (b.x, b.y - 1), (b.x + 1, b.y - 1)]
            for px, py in preds:
                if (px, py) in grid:
                    b.add_predecessor(grid[(px, py)])

    return blocks, width, depth


def plot_results(blocks, schedule, width, depth, history):
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


def run_solver(width, depth, periods, discount_rate, mining_cap, proc_cap, max_iters, step_size, seed):
    rng = np.random.default_rng(seed)
    blocks, w, d = generate_2d_deposit(width, depth, rng)
    scheduler = BZScheduler(blocks, periods, discount_rate, mining_cap, proc_cap)
    final_schedule, history = scheduler.solve(max_iterations=max_iters, step_size_factor=step_size)
    fig = plot_results(blocks, final_schedule, w, d, history)
    return blocks, final_schedule, history, fig


def main():
    st.title("BZ Scheduler Demo (Streamlit)")
    st.markdown("Run the Bienstock-Zuckerberg toy optimizer and visualize the schedule and convergence.")

    with st.sidebar:
        st.header("Model Parameters")
        width = st.number_input("Width (blocks)", min_value=5, max_value=60, value=25, step=1)
        depth = st.number_input("Depth (blocks)", min_value=5, max_value=40, value=15, step=1)
        periods = st.number_input("Periods", min_value=2, max_value=10, value=4, step=1)
        discount_rate = st.slider("Discount Rate", min_value=0.0, max_value=0.25, value=0.10, step=0.01)
        max_iters = st.number_input("Max Iterations", min_value=5, max_value=100, value=15, step=1)
        step_size = st.number_input("Step Size Factor", min_value=0.00001, max_value=0.001, value=0.00008, format="%f")
        seed = st.number_input("Random Seed", min_value=0, max_value=10_000, value=42, step=1)

        st.header("Capacity Settings")
        mining_cap = st.number_input("Mining Capacity (t/period)", min_value=1000, max_value=200_000, value=30_000, step=1000)
        proc_cap_factor = st.slider("Processing Cap as % of total ore", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

    if st.button("Run Optimization", type="primary"):
        with st.spinner("Running BZ optimization..."):
            blocks, schedule, history, fig = run_solver(
                width,
                depth,
                periods,
                discount_rate,
                mining_cap,
                0,  # placeholder, set below
                max_iters,
                step_size,
                seed,
            )

            total_ore_tonnage = sum(b.tonnage for b in blocks if b.economic_value > 0)
            proc_cap = total_ore_tonnage * proc_cap_factor

            # Re-run with calculated processing cap so both caps reflect slider choices
            blocks, schedule, history, fig = run_solver(
                width,
                depth,
                periods,
                discount_rate,
                mining_cap,
                proc_cap,
                max_iters,
                step_size,
                seed,
            )

        st.success("Optimization complete.")

        latest_profit = history[-1][1] if history else 0
        latest_violation = history[-1][2] if history else 0

        col1, col2 = st.columns(2)
        col1.metric("Final NPV ($)", f"{latest_profit:,.0f}")
        col2.metric("Constraint Violation (t)", f"{latest_violation:,.0f}")
        if latest_violation > 0:
            st.warning("Constraints not fully satisfied; adjust capacities or iterations.")

        st.pyplot(fig)

        st.subheader("Convergence Data")
        data = {"iteration": [h[0] for h in history], "npv": [h[1] for h in history], "violation_tons": [h[2] for h in history]}
        st.dataframe(data)
    else:
        st.info("Set parameters on the left, then click Run Optimization.")


if __name__ == "__main__":
    main()
