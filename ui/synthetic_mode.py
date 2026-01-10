"""Synthetic deposit demo interface."""
import matplotlib
matplotlib.use("Agg")  # headless-friendly backend

import numpy as np
import streamlit as st

from block import Block
from bz_algorithm_logic import BZScheduler
from utils.plotting import plot_results


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


def run_solver(width, depth, periods, discount_rate, mining_cap, proc_cap, max_iters, step_size, seed):
    """Run the BZ solver on synthetic deposit."""
    rng = np.random.default_rng(seed)
    blocks, w, d = generate_2d_deposit(width, depth, rng)
    scheduler = BZScheduler(blocks, periods, discount_rate, mining_cap, proc_cap)
    final_schedule, history = scheduler.solve(max_iterations=max_iters, step_size_factor=step_size)
    fig = plot_results(blocks, final_schedule, w, d, history)
    return blocks, final_schedule, history, fig


def show_synthetic_sidebar():
    """Show sidebar controls for synthetic demo."""
    st.header("Synthetic Model")
    width = st.number_input("Width (blocks)", min_value=5, max_value=60, value=25, step=1, key="synth_width")
    depth = st.number_input("Depth (blocks)", min_value=5, max_value=40, value=15, step=1, key="synth_depth")
    max_iters = st.number_input("Max Iterations", min_value=5, max_value=100, value=15, step=1, key="synth_iters")
    step_size = st.number_input("Step Size Factor", min_value=0.00001, max_value=0.001, value=0.00008, format="%f", key="synth_step")
    seed = st.number_input("Random Seed", min_value=0, max_value=10_000, value=42, step=1, key="synth_seed")
    proc_cap_factor = st.slider("Processing Cap as % of total ore", min_value=0.1, max_value=1.0, value=0.4, step=0.05, key="synth_proc")

    return {
        "width": width,
        "depth": depth,
        "max_iters": max_iters,
        "step_size": step_size,
        "seed": seed,
        "proc_cap_factor": proc_cap_factor,
    }


def show_synthetic_results(params, periods, discount_rate, mining_cap):
    """Run and display synthetic demo results."""
    with st.spinner("Running simple BZ optimization..."):
        blocks, schedule, history, fig = run_solver(
            params["width"],
            params["depth"],
            periods,
            discount_rate,
            mining_cap,
            0,  # placeholder, set below
            params["max_iters"],
            params["step_size"],
            params["seed"],
        )

        total_ore_tonnage = sum(b.tonnage for b in blocks if b.economic_value > 0)
        proc_cap = total_ore_tonnage * params["proc_cap_factor"]

        blocks, schedule, history, fig = run_solver(
            params["width"],
            params["depth"],
            periods,
            discount_rate,
            mining_cap,
            proc_cap,
            params["max_iters"],
            params["step_size"],
            params["seed"],
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
