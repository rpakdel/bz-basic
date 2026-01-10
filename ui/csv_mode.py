"""CSV-based optimization interface with partition refinement."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from block import Block
from bz_partition_refinement import BZSolver
from deposit_utils import read_block_model_dataframe


def show_csv_sidebar(periods, mining_cap):
    """Show sidebar controls for CSV mode."""
    st.header("Upload Block Model (CSV)")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_files = sorted(p for p in data_dir.glob("*.csv"))
    csv_names = [p.name for p in csv_files]

    if csv_names:
        selected_name = st.selectbox("Pick CSV", options=csv_names, key="csv_mode_csv")
    else:
        selected_name = None
        st.warning("No CSV files found in data/. Add CSVs to proceed.")

    max_iters_gen = st.number_input("Max BZ Iterations", min_value=5, max_value=100, value=20, step=1, key="csv_iters")

    df_preview = None
    if selected_name:
        df_preview = read_block_model_dataframe(str(data_dir / selected_name))

    def _bounds(col: str, fallback_min: int = 0, fallback_max: int = 100):
        if df_preview is None or col not in df_preview.columns:
            return fallback_min, fallback_max
        return int(df_preview[col].min()), int(df_preview[col].max())

    x_min, x_max = _bounds("x")
    y_min, y_max = _bounds("y")
    z_min, z_max = _bounds("z")

    # Ensure sliders have a valid range even if bounds collapse
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1
    if z_min == z_max:
        z_max = z_min + 1

    x_range = st.slider("X range", min_value=x_min, max_value=x_max, value=(x_min, x_max), key="csv_x")
    y_range = st.slider("Y range", min_value=y_min, max_value=y_max, value=(y_min, y_max), key="csv_y")
    z_range = st.slider("Z range", min_value=z_min, max_value=z_max, value=(z_min, z_max), key="csv_z")

    color_by = st.selectbox(
        "Color by",
        options=["tonnage", "grade", "economic_value"],
        index=1,
        key="csv_color",
    )

    return {
        "selected_name": selected_name,
        "max_iters": max_iters_gen,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "color_by": color_by,
    }


def show_csv_results(params, periods, discount_rate, mining_cap):
    """Run and display CSV-based optimization results."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_files = sorted(p for p in data_dir.glob("*.csv"))
    if not csv_files:
        st.error("No CSV files found in data/. Add CSVs and try again.")
        return

    if params["selected_name"] is None:
        st.error("Pick a CSV from the dropdown to continue.")
        return

    selected_path = data_dir / params["selected_name"]
    
    # Use read_block_model_dataframe to automatically calculate BlockID and Predecessors
    df_blocks = read_block_model_dataframe(str(selected_path))

    # Apply spatial filters
    x_low, x_high = params["x_range"]
    y_low, y_high = params["y_range"]
    z_low, z_high = params["z_range"]
    df_blocks = df_blocks[
        (df_blocks["x"] >= x_low)
        & (df_blocks["x"] <= x_high)
        & (df_blocks["y"] >= y_low)
        & (df_blocks["y"] <= y_high)
        & (df_blocks["z"] >= z_low)
        & (df_blocks["z"] <= z_high)
    ]

    required_cols = {"BlockID", "Value", "Tonnage", "Predecessors"}
    can_solve = required_cols.issubset(df_blocks.columns)
    if can_solve:
        # Convert DataFrame to Block objects for the solver
        blocks_csv = []
        id_to_block = {}
        
        # First pass: create all blocks
        for _, row in df_blocks.iterrows():
            block_id = int(row["BlockID"])
            value = float(row["Value"])
            tonnage = float(row["Tonnage"])
            x = int(row["x"])
            y = int(row["y"])
            z = int(row["z"])
            grade = float(row.get("grade", 0.0))
            
            block = Block(block_id, x, y, z, tonnage, grade, value)
            blocks_csv.append(block)
            id_to_block[block_id] = block
        
        # Second pass: add predecessor relationships
        for _, row in df_blocks.iterrows():
            block_id = int(row["BlockID"])
            block = id_to_block[block_id]
            pred_str = str(row.get("Predecessors", "")).strip()
            if pred_str:
                pred_ids = [int(x) for x in pred_str.split(";") if x.strip()]
                for pred_id in pred_ids:
                    pred_block = id_to_block.get(pred_id)
                    if pred_block is not None:
                        block.add_predecessor(pred_block)

    with st.spinner("Running generalized BZ (partition refinement)..."):
        if can_solve:
            solver = BZSolver(
                blocks=blocks_csv,
                periods=periods,
                discount_rate=discount_rate,
                mining_capacity=mining_cap,
            )
            final_schedule, logs = solver.solve(max_iterations=params["max_iters"])
        else:
            final_schedule, logs = {}, []

    st.success("Optimization complete.")

    # Load and display metadata
    metadata_path = selected_path.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    if metadata and "attributes" in metadata:
        st.subheader("Block Model Metadata")
        # Filter out categorical attributes (like reserve) that don't have min/max
        numeric_attrs = {k: v for k, v in metadata["attributes"].items() if "min" in v and "max" in v}
        if numeric_attrs:
            metadata_cols = st.columns(len(numeric_attrs))
            for idx, (attr_name, attr_stats) in enumerate(numeric_attrs.items()):
                with metadata_cols[idx]:
                    st.metric(
                        f"{attr_name.capitalize()} Range",
                        f"{attr_stats['min']:.2f} - {attr_stats['max']:.2f}"
                    )

    st.subheader("3D Block Model")
    
    fig3d = px.scatter_3d(
        df_blocks,
        x="x",
        y="y",
        z="z",
        color=params["color_by"],
        color_continuous_scale="Viridis",
        title="Sampled Block Model",
    )
    
    # Update marker properties
    fig3d.update_traces(marker=dict(size=3, opacity=0.6))
    
    # Apply consistent color scale from metadata
    if metadata and "attributes" in metadata and params["color_by"] in metadata["attributes"]:
        color_range = metadata["attributes"][params["color_by"]]
        fig3d.update_layout(
            coloraxis=dict(
                colorscale="Viridis",
                cmin=color_range["min"],
                cmax=color_range["max"],
            )
        )

    if can_solve:
        # Charts: LP Bound vs Iteration, Partitions count
        st.subheader("Iteration Logs")
        st.line_chart({"LP Bound": [l[1] for l in logs]})
        st.bar_chart({"Partitions": [l[2] for l in logs]})

        # Final schedule dataframe
        id_to_block = {b.id: b for b in blocks_csv}
        rows = []
        for bid, period in final_schedule.items():
            dest = "process" if id_to_block[bid].economic_value > 0 else "waste"
            rows.append({"BlockID": bid, "Period": period + 1 if period >= 0 else -1, "Destination": dest})
        st.subheader("Final Schedule")
        st.dataframe(rows)

        # Tonnage per period
        tons_per_t = np.zeros(periods)
        npv_cum = []
        npv_total = 0.0
        for bid, period in final_schedule.items():
            if period >= 0:
                tons_per_t[period] += id_to_block[bid].tonnage
                df = 1.0 / ((1.0 + discount_rate) ** (period + 1))
                npv_total += id_to_block[bid].economic_value * df
            npv_cum.append(npv_total)

        st.subheader("Tonnage Mined per Period")
        st.bar_chart(tons_per_t.tolist())

        st.subheader("Cumulative NPV (Final Schedule)")
        st.line_chart(npv_cum)
    else:
        st.warning("Selected CSV lacks required columns for optimization (needs BlockID, Value, Tonnage, Predecessors). Showing visualization only.")
