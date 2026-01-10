"""BZ optimization demo interface."""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from block import Block
from bz_partition_refinement import BZSolver
from baseline_lp_solver import LPSolver, decode_lp_to_schedule
from deposit_utils import read_block_model_dataframe


def show_bz_demo_section(view_params):
    """Show the BZ optimization demo section."""
    st.header("Run BZ Demo")
    
    # Use expander for the demo controls
    with st.expander("BZ Optimization Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            periods = st.number_input(
                "Periods", 
                min_value=2, 
                max_value=100, 
                value=4, 
                step=1,
                key="bz_periods",
                help="Number of mining periods"
            )
            discount_rate = st.slider(
                "Discount Rate", 
                min_value=0.0, 
                max_value=0.25, 
                value=0.10, 
                step=0.01,
                key="bz_discount",
                help="NPV discount rate"
            )
        
        with col2:
            mining_cap = st.number_input(
                "Mining Capacity (t/period)", 
                min_value=1000, 
                max_value=200_000, 
                value=30_000, 
                step=1000,
                key="bz_mining_cap",
                help="Hard cap on mining tonnage per period"
            )
            max_iters = st.number_input(
                "Max BZ Iterations", 
                min_value=5, 
                max_value=100, 
                value=20, 
                step=1,
                key="bz_max_iters",
                help="Maximum solver iterations"
            )
        
        proc_cap_factor = st.slider(
            "Processing Cap (% of ore tonnage)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.05,
            key="bz_proc_cap",
            help="Processing capacity as fraction of total ore tonnage (0 = no constraint)"
        )

    # Run Optimization button
    if st.button("▶ Run Optimization", type="primary", key="bz_run_button"):
        run_bz_optimization(
            view_params=view_params,
            periods=periods,
            discount_rate=discount_rate,
            mining_cap=mining_cap,
            max_iters=max_iters,
            proc_cap_factor=proc_cap_factor,
        )
    
    # Display results if they exist in session state
    if "bz_results" in st.session_state:
        display_bz_results(st.session_state["bz_results"])


def run_bz_optimization(view_params, periods, discount_rate, mining_cap, max_iters, proc_cap_factor):
    """Run the BZ optimization and store results in session state."""
    
    data_dir = Path("data")
    selected_path = data_dir / view_params["selected_name"]
    
    # Load block model
    with st.spinner("Loading block model..."):
        df_blocks = read_block_model_dataframe(str(selected_path))
    
    # Apply spatial filters from view parameters
    x_low, x_high = view_params["x_range"]
    y_low, y_high = view_params["y_range"]
    z_low, z_high = view_params["z_range"]
    
    df_filtered = df_blocks[
        (df_blocks["x"] >= x_low)
        & (df_blocks["x"] <= x_high)
        & (df_blocks["y"] >= y_low)
        & (df_blocks["y"] <= y_high)
        & (df_blocks["z"] >= z_low)
        & (df_blocks["z"] <= z_high)
    ]

    if df_filtered.empty:
        st.error("No blocks found in the selected range. Please adjust filters.")
        return
    
    # Convert DataFrame to Block objects
    with st.spinner("Building block graph..."):
        blocks_opt = []
        id_to_block = {}
        
        # First pass: create all blocks
        for _, row in df_filtered.iterrows():
            block_id = int(row["BlockID"])
            value = float(row["Value"])
            tonnage = float(row["Tonnage"])
            x = int(row["x"])
            y = int(row["y"])
            z = int(row["z"])
            grade = float(row.get("grade", 0.0))
            
            block = Block(block_id, x, y, z, tonnage, grade, value)
            blocks_opt.append(block)
            id_to_block[block_id] = block
        
        # Second pass: add predecessor relationships
        for _, row in df_filtered.iterrows():
            block_id = int(row["BlockID"])
            block = id_to_block.get(block_id)
            if block is None: continue
            
            pred_str = str(row.get("Predecessors", "")).strip()
            if pred_str:
                pred_ids = [int(x) for x in pred_str.split(";") if x.strip()]
                for pred_id in pred_ids:
                    pred_block = id_to_block.get(pred_id)
                    if pred_block is not None:
                        block.add_predecessor(pred_block)
    
    # Calculate processing capacity from ore tonnage
    proc_cap = None
    if proc_cap_factor > 0:
        total_ore_tonnage = sum(b.tonnage for b in blocks_opt if b.economic_value > 0)
        proc_cap = total_ore_tonnage * proc_cap_factor

    # Run BZ Solver
    t_start = time.time()
    with st.spinner("Solving with BZ Partition Refinement..."):
        bz_solver = BZSolver(
            blocks=blocks_opt,
            periods=periods,
            discount_rate=discount_rate,
            mining_capacity=mining_cap,
            processing_capacity=proc_cap,
        )
        final_schedule, logs = bz_solver.solve(max_iterations=max_iters)
        bz_npv = logs[-1][1] if logs else 0
    t_bz = time.time() - t_start

    # Run Baseline LP Solver for comparison
    t_start_lp = time.time()
    with st.spinner("Solving Baseline LP for comparison..."):
        lp_solver = LPSolver(
            blocks=blocks_opt,
            periods=periods,
            discount_rate=discount_rate,
            mining_capacity=mining_cap,
            processing_capacity=proc_cap,
        )
        _, lp_npv = lp_solver.solve()
    t_lp = time.time() - t_start_lp

    st.success("✓ Optimization complete")

    # Store results in session state
    st.session_state["bz_results"] = {
        "final_schedule": final_schedule,
        "logs": logs,
        "id_to_block": id_to_block,
        "periods": periods,
        "discount_rate": discount_rate,
        "mining_cap": mining_cap,
        "bz_npv": bz_npv,
        "lp_npv": lp_npv,
        "bz_duration": t_bz,
        "lp_duration": t_lp,
        "solver_name": "BZ vs Baseline Comparison"
    }


def display_bz_results(results):
    """Display optimization results from session state."""
    final_schedule = results["final_schedule"]
    logs = results["logs"]
    id_to_block = results["id_to_block"]
    periods = results["periods"]
    discount_rate = results["discount_rate"]
    mining_cap = results["mining_cap"]
    bz_npv = results.get("bz_npv", 0)
    lp_npv = results.get("lp_npv", 0)
    bz_duration = results.get("bz_duration", 0)
    lp_duration = results.get("lp_duration", 0)
    
    # Calculate final integer NPV
    int_npv = 0.0
    for bid, period in final_schedule.items():
        if period >= 0:
            df_factor = 1.0 / ((1.0 + discount_rate) ** (period + 1))
            int_npv += id_to_block[bid].economic_value * df_factor

    # Display results
    st.divider()
    st.subheader("Optimization Comparison: BZ vs Baseline LP")
    
    # Global LP Comparison Metrics
    col1, col2, col3 = st.columns(3)
    
    # Optimality Gap
    gap = (lp_npv - bz_npv) / lp_npv if lp_npv > 0 else 0
    col1.metric("Optimality Gap", f"{gap:.4%}", help="Gap between BZ Relaxed Bound and Global LP Optimal NPV")
    
    # NPV Comparison
    col2.metric("Integer Schedule NPV", f"${int_npv:,.0f}", help="Actual NPV after applying TopoSort heuristic")
    col3.metric("Baseline LP NPV", f"${lp_npv:,.0f}", help="Global optimal relaxed solution")
    
    # Timing Comparison
    col1, col2, col3 = st.columns(3)
    col1.metric("BZ (Relaxed) NPV", f"${bz_npv:,.0f}", help="Final NPV from the BZ Master LP")
    col2.metric("BZ Solve Time", f"{bz_duration:.2f}s")
    col3.metric("Baseline LP Time", f"{lp_duration:.2f}s")
    
    # Comparison Chart
    comp_df = pd.DataFrame({
        "Result Type": ["Integer (Feasible)", "BZ Master (Relaxed)", "Baseline LP (Global)"],
        "NPV ($)": [int_npv, bz_npv, lp_npv]
    })
    
    fig_comp = px.bar(
        comp_df,
        x="Result Type",
        y="NPV ($)",
        color="Result Type",
        title="Optimization Quality Benchmarks",
        text_auto='.2s'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Heuristic Schedule Metrics
    st.divider()
    st.subheader("Mining Schedule Analysis")
    
    # Metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    
    # Count scheduled blocks
    scheduled_count = sum(1 for p in final_schedule.values() if p >= 0)
    mcol1.metric("Blocks Scheduled", scheduled_count)
    
    # Calculate Iterations or Partitions
    if logs and len(logs) > 0:
        mcol2.metric("BZ Iterations", len(logs))
        mcol3.metric("Final Partitions", logs[-1][2])
    
    # Convergence chart
    if logs and len(logs) > 0:
        st.subheader("Convergence History")
        
        # Create DataFrames for plotting
        iterations = [l[0] for l in logs]
        lp_bounds = [l[1] for l in logs]
        
        # LP Bound chart using Plotly for proper axis control
        import plotly.graph_objects as go
        
        fig_lp = go.Figure()
        fig_lp.add_trace(go.Scatter(
            x=iterations,
            y=lp_bounds,
            mode='lines+markers',
            name='LP Bound',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        fig_lp.update_layout(
            xaxis_title='Iteration',
            yaxis_title='LP Bound',
            height=400,
            showlegend=False,
            xaxis=dict(
                tickmode='array',
                tickvals=iterations,
                ticktext=[str(int(i)) for i in iterations]
            )
        )
        st.plotly_chart(fig_lp, use_container_width=True)
        
        if logs[0] and len(logs[0]) >= 3:
            partitions = [l[2] for l in logs]
            
            fig_part = go.Figure()
            fig_part.add_trace(go.Bar(
                x=iterations,
                y=partitions,
                name='Partitions',
                marker=dict(color='lightblue')
            ))
            fig_part.update_layout(
                xaxis_title='Iteration',
                yaxis_title='Number of Partitions',
                height=400,
                showlegend=False,
                xaxis=dict(
                    tickmode='array',
                    tickvals=iterations,
                    ticktext=[str(int(i)) for i in iterations]
                )
            )
            st.plotly_chart(fig_part, use_container_width=True)
    
    # 3D Visualization by Destination
    st.subheader("Optimized Blocks by Destination")
    
    viz_data = []
    for bid, period in final_schedule.items():
        block = id_to_block[bid]
        destination = "Process (Ore)" if block.economic_value > 0 else "Waste"
        period_num = period + 1 if period >= 0 else -1
        viz_data.append({
            "x": block.x,
            "y": block.y,
            "z": block.z,
            "Destination": destination,
            "Period": period_num,
            "Tonnage": block.tonnage,
            "Grade": block.grade,
            "Value": block.economic_value
        })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Period filter
    col1, col2 = st.columns([2, 1])
    with col1:
        available_periods = sorted([p for p in viz_df["Period"].unique() if p > 0])
        if -1 in viz_df["Period"].unique():
            available_periods.append("Unscheduled")
        
        selected_periods = st.multiselect(
            "Select periods to display",
            options=available_periods,
            default=available_periods,
            key="viz_period_filter",
            help="Choose which periods to show in the 3D visualization"
        )
    
    with col2:
        color_by_viz = st.radio(
            "Color by",
            options=["Destination", "Period"],
            key="viz_color_by",
            help="Choose coloring scheme"
        )
    
    # Filter by selected periods
    if selected_periods:
        # Convert "Unscheduled" back to -1 for filtering
        filter_periods = [p if p != "Unscheduled" else -1 for p in selected_periods]
        viz_df_filtered = viz_df[viz_df["Period"].isin(filter_periods)].copy()
        
        # Format Period for display
        viz_df_filtered["Period_Display"] = viz_df_filtered["Period"].apply(
            lambda p: f"Period {p}" if p > 0 else "Unscheduled"
        )
        
        if color_by_viz == "Destination":
            fig_dest = px.scatter_3d(
                viz_df_filtered,
                x="x",
                y="y",
                z="z",
                color="Destination",
                color_discrete_map={
                    "Process (Ore)": "gold",
                    "Waste": "gray"
                },
                hover_data=["Period_Display", "Tonnage", "Grade", "Value"],
                title=f"Optimized Mining Schedule - Colored by Destination ({len(viz_df_filtered)} blocks)",
            )
        else:  # Color by Period
            fig_dest = px.scatter_3d(
                viz_df_filtered,
                x="x",
                y="y",
                z="z",
                color="Period_Display",
                hover_data=["Destination", "Tonnage", "Grade", "Value"],
                title=f"Optimized Mining Schedule - Colored by Period ({len(viz_df_filtered)} blocks)",
            )
        
        fig_dest.update_traces(marker=dict(size=4, opacity=0.7))
        fig_dest.update_layout(
            scene=dict(
                xaxis_title='X (Easting)',
                yaxis_title='Y (Northing)',
                zaxis_title='Z (Elevation)',
            ),
            height=600
        )
        
        st.plotly_chart(fig_dest, use_container_width=True)
    else:
        st.warning("Please select at least one period to visualize.")
    
    # Final schedule
    st.subheader("Final Mining Schedule")
    
    schedule_rows = []
    for bid, period in final_schedule.items():
        block = id_to_block[bid]
        destination = "Process" if block.economic_value > 0 else "Waste"
        period_display = period + 1 if period >= 0 else "Unscheduled"
        
        schedule_rows.append({
            "Block ID": bid,
            "Period": period_display,
            "Destination": destination,
            "Tonnage": block.tonnage,
            "Grade": f"{block.grade:.3f}",
            "Value": f"${block.economic_value:,.0f}"
        })
    
    schedule_df = pd.DataFrame(schedule_rows)
    st.dataframe(schedule_df, use_container_width=True)
    
    # Tonnage per period
    st.subheader("Mining Tonnage per Period")
    
    tons_per_t = np.zeros(periods)
    for bid, period in final_schedule.items():
        if period >= 0 and period < periods:
            tons_per_t[period] += id_to_block[bid].tonnage
    
    # Create tonnage chart with Plotly to show capacity line
    tonnage_df = pd.DataFrame({
        "Period": [f"P{t+1}" for t in range(periods)],
        "Mining Tonnage": tons_per_t.tolist()
    })
    
    fig_tonnage = px.bar(
        tonnage_df,
        x="Period",
        y="Mining Tonnage",
        title=f"Mining Tonnage vs Capacity ({mining_cap:,.0f} t/period)",
        color_discrete_sequence=["#1f77b4"]
    )
    
    fig_tonnage.add_hline(
        y=mining_cap, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Mining Capacity",
        annotation_position="bottom right"
    )
    
    st.plotly_chart(fig_tonnage, use_container_width=True)
    
    # Show capacity violations
    violations_detected = False
    for t in range(periods):
        if tons_per_t[t] > mining_cap:
            violations_detected = True
            st.warning(f"⚠ Period {t+1}: {tons_per_t[t]:,.0f} tons exceeds capacity of {mining_cap:,.0f} tons")
    
    if not violations_detected:
        st.success(f"✓ All periods respect the {mining_cap:,.0f} t/period capacity constraint")
    
    # NPV by period
    st.subheader("Cumulative NPV by Period")
    
    npv_cumulative = []
    npv_total = 0.0
    for bid, period in final_schedule.items():
        if period >= 0:
            df_factor = 1.0 / ((1.0 + discount_rate) ** (period + 1))
            npv_total += id_to_block[bid].economic_value * df_factor
        npv_cumulative.append(npv_total)
    
    st.line_chart(npv_cumulative)
