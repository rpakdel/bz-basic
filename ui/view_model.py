"""Block model visualization interface."""
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from deposit_utils import calculate_block_id, read_block_model_csv


def show_view_model_sidebar():
    """Show sidebar controls for block model visualization."""
    st.header("Block Model Visualization")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_files = sorted(p for p in data_dir.glob("*.csv"))
    csv_names = [p.name for p in csv_files]

    if csv_names:
        selected_name = st.selectbox("Pick CSV", options=csv_names, key="view_model_csv")
    else:
        selected_name = None
        st.warning("No CSV files found in data/. Add CSVs to proceed.")

    # Load preview to check available columns
    df_preview = None
    if selected_name:
        file_bytes_preview = (data_dir / selected_name).read_bytes()
        df_preview = pd.read_csv(io.BytesIO(file_bytes_preview))
    
    # Determine color options based on available columns
    color_options = ["tonnage", "grade", "economic_value"]
    if df_preview is not None and "reserve" in df_preview.columns:
        color_options.append("reserve")
    
    color_by = st.selectbox(
        "Color by",
        options=color_options,
        index=1,
        key="view_model_color",
    )

    df_preview = None
    if selected_name:
        file_bytes_preview = (data_dir / selected_name).read_bytes()
        df_preview = pd.read_csv(io.BytesIO(file_bytes_preview))

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

    x_range = st.slider("X range", min_value=x_min, max_value=x_max, value=(x_min, x_max), key="view_model_x")
    y_range = st.slider("Y range", min_value=y_min, max_value=y_max, value=(y_min, y_max), key="view_model_y")
    z_range = st.slider("Z range", min_value=z_min, max_value=z_max, value=(z_min, z_max), key="view_model_z")

    return {
        "selected_name": selected_name,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "color_by": color_by,
    }


def show_view_model_content(params):
    """Display block model visualization and statistics."""
    st.header("Block Model Data")

    if params["selected_name"] is None:
        st.error("Pick a CSV from the sidebar to view the block model.")
        return

    data_dir = Path("data")
    selected_path = data_dir / params["selected_name"]
    file_bytes = selected_path.read_bytes()
    df_blocks = pd.read_csv(io.BytesIO(file_bytes))

    # Load metadata for consistent color scale
    metadata_path = selected_path.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    # Apply spatial filters
    x_low, x_high = params["x_range"]
    y_low, y_high = params["y_range"]
    z_low, z_high = params["z_range"]
    df_filtered = df_blocks[
        (df_blocks["x"] >= x_low)
        & (df_blocks["x"] <= x_high)
        & (df_blocks["y"] >= y_low)
        & (df_blocks["y"] <= y_high)
        & (df_blocks["z"] >= z_low)
        & (df_blocks["z"] <= z_high)
    ]

    st.write(f"**Total blocks in file:** {len(df_blocks):,}")
    st.write(f"**Blocks in selected range:** {len(df_filtered):,}")

    # Show 3D visualization
    st.subheader("3D Block Model")
    
    # Special handling for reserve column (categorical)
    if params["color_by"] == "reserve" and "reserve" in df_filtered.columns:
        # Map reserve codes to labels
        df_filtered = df_filtered.copy()
        df_filtered['reserve_label'] = df_filtered['reserve'].map({
            1: 'Ore',
            0: 'Waste',
            -1: 'Overburden/Other'
        })
        
        fig3d = px.scatter_3d(
            df_filtered,
            x="x",
            y="y",
            z="z",
            color='reserve_label',
            color_discrete_map={
                'Ore': 'gold',
                'Waste': 'red',
                'Overburden/Other': 'lightgray'
            },
            title=f"Block Model - Colored by Reserve Type",
        )
    else:
        # Determine appropriate color scale based on data range
        data_min = df_filtered[params["color_by"]].min()
        data_max = df_filtered[params["color_by"]].max()
        
        # Get full range from metadata if available
        scale_min = data_min
        scale_max = data_max
        if metadata and "attributes" in metadata and params["color_by"] in metadata["attributes"]:
            color_range = metadata["attributes"][params["color_by"]]
            if "min" in color_range and "max" in color_range:
                scale_min = color_range["min"]
                scale_max = color_range["max"]
        
        # Use diverging color scale for data with negative values (like economic_value)
        color_scale = "Viridis"
        if scale_min < 0:
            color_scale = "RdYlGn"  # Red (negative) - Yellow (zero) - Green (positive)
        
        # Create figure
        fig3d = px.scatter_3d(
            df_filtered,
            x="x",
            y="y",
            z="z",
            color=params["color_by"],
            color_continuous_scale=color_scale,
            title=f"Block Model - Colored by {params['color_by'].capitalize()}",
        )
        
        # Explicitly set color axis range and midpoint
        fig3d.update_coloraxes(
            cmin=scale_min,
            cmax=scale_max,
            cmid=0 if scale_min < 0 else None
        )
    
    # Update marker properties
    fig3d.update_traces(marker=dict(size=3, opacity=0.6))
    
    st.plotly_chart(fig3d, use_container_width=True)

    # Show block statistics
    st.subheader("Block Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tonnage", f"{df_filtered['tonnage'].sum():,.0f} t")
    col2.metric("Avg Grade", f"{df_filtered['grade'].mean():.3f}")
    col3.metric("Total Economic Value", f"${df_filtered['economic_value'].sum():,.0f}")

    # Show data table
    st.subheader("Block Data Table")
    st.dataframe(df_filtered, use_container_width=True)
    
    # Precedence visualization section
    st.divider()
    st.header("Precedence Graph Visualization")
    
    # Precedence controls in main content area
    show_precedence = st.checkbox(
        "Show Precedence Relationships", 
        value=False, 
        key="view_model_precedence",
        help="Visualize block dependencies based on 1:3 slope constraints"
    )
    
    if show_precedence:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            precedence_mode = st.radio(
                "Visualization Mode",
                options=["Single Block", "Region View"],
                key="precedence_mode",
                help="Single Block: Focus on one block and its predecessors\nRegion View: Show all precedence in filtered range"
            )
        
        with col2:
            if precedence_mode == "Single Block":
                precedence_block_id = st.number_input(
                    "Block ID to visualize",
                    min_value=0,
                    value=1000,
                    step=1,
                    key="precedence_block_id",
                    help="Enter block ID to see its predecessors (blocks that must be mined first)"
                )
            else:
                st.info("Using X, Y, Z ranges from sidebar to define region")
        
        # Build params dict for precedence visualization
        precedence_params = {
            **params,
            "show_precedence": show_precedence,
            "precedence_mode": precedence_mode,
            "precedence_block_id": precedence_block_id if precedence_mode == "Single Block" else None,
        }
        
        show_precedence_visualization(precedence_params, selected_path, df_filtered)


def show_precedence_visualization(params, csv_path, df_filtered):
    """Display interactive precedence graph visualization."""
    
    try:
        # Load full block model with precedence
        with st.spinner("Loading block model with precedence relationships..."):
            blocks, x_size, y_size, z_size = read_block_model_csv(str(csv_path))
        
        id_to_block = {b.id: b for b in blocks}
        
        st.success(f"✓ Loaded {len(blocks)} blocks with precedence relationships")
        
        # Statistics
        total_predecessors = sum(len(b.predecessors) for b in blocks)
        surface_blocks = sum(1 for b in blocks if len(b.predecessors) == 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Precedence Links", f"{total_predecessors:,}")
        col2.metric("Surface Blocks (no predecessors)", f"{surface_blocks:,}")
        col3.metric("Avg Predecessors/Block", f"{total_predecessors / len(blocks):.2f}")
        
        st.divider()
        
        if params["precedence_mode"] == "Single Block":
            show_single_block_precedence(params, blocks, id_to_block, x_size, y_size, z_size)
        else:  # Region View
            show_region_precedence(params, blocks, id_to_block, x_size, y_size, z_size, df_filtered)
            
    except Exception as e:
        st.error(f"Error loading precedence data: {e}")


def show_single_block_precedence(params, blocks, id_to_block, x_size, y_size, z_size):
    """Visualize a single block and its predecessors in 3D."""
    st.subheader("Single Block Precedence View")
    
    block_id = params["precedence_block_id"]
    
    if block_id not in id_to_block:
        st.error(f"Block ID {block_id} not found. Valid range: 0 to {len(blocks) - 1}")
        return
    
    target_block = id_to_block[block_id]
    
    # Display block info
    st.info(
        f"**Block {block_id}** at coordinates ({target_block.x}, {target_block.y}, {target_block.z})\n\n"
        f"- Tonnage: {target_block.tonnage:,.0f} t\n"
        f"- Grade: {target_block.grade:.4f}\n"
        f"- Economic Value: ${target_block.economic_value:,.2f}\n"
        f"- **Predecessors: {len(target_block.predecessors)}**"
    )
    
    if len(target_block.predecessors) == 0:
        st.success("This is a **surface block** - can be mined first (no predecessors)")
    
    # Create 3D plotly figure
    fig = go.Figure()
    
    # Add target block (red)
    fig.add_trace(go.Scatter3d(
        x=[target_block.x],
        y=[target_block.y],
        z=[target_block.z],
        mode='markers',
        name='Target Block',
        marker=dict(size=12, color='red', symbol='circle', line=dict(color='black', width=2)),
        text=[f"Block {block_id}<br>({target_block.x}, {target_block.y}, {target_block.z})"],
        hoverinfo='text'
    ))
    
    # Add predecessors (blue)
    if target_block.predecessors:
        pred_x = [p.x for p in target_block.predecessors]
        pred_y = [p.y for p in target_block.predecessors]
        pred_z = [p.z for p in target_block.predecessors]
        pred_labels = [
            f"Block {p.id}<br>({p.x}, {p.y}, {p.z})<br>Predecessor"
            for p in target_block.predecessors
        ]
        
        fig.add_trace(go.Scatter3d(
            x=pred_x,
            y=pred_y,
            z=pred_z,
            mode='markers',
            name='Predecessors',
            marker=dict(size=10, color='blue', symbol='diamond', line=dict(color='black', width=1.5)),
            text=pred_labels,
            hoverinfo='text'
        ))
        
        # Add arrows from predecessors to target
        for pred in target_block.predecessors:
            fig.add_trace(go.Scatter3d(
                x=[pred.x, target_block.x],
                y=[pred.y, target_block.y],
                z=[pred.z, target_block.z],
                mode='lines',
                line=dict(color='gray', width=4, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Layout
    fig.update_layout(
        title=f"Block {block_id} Precedence Graph<br><sub>Blue diamonds = predecessors (must mine first)</sub>",
        scene=dict(
            xaxis_title='X (Easting)',
            yaxis_title='Y (Northing)',
            zaxis_title='Z (Elevation)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show predecessor details in a table
    if target_block.predecessors:
        st.subheader("Predecessor Details")
        pred_data = []
        for pred in target_block.predecessors:
            pred_data.append({
                "Block ID": pred.id,
                "X": pred.x,
                "Y": pred.y,
                "Z": pred.z,
                "Tonnage": pred.tonnage,
                "Grade": pred.grade,
                "Value": pred.economic_value
            })
        
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True)


def show_region_precedence(params, blocks, id_to_block, x_size, y_size, z_size, df_filtered):
    """Visualize precedence relationships in the filtered region."""
    st.subheader("Region Precedence View")
    
    # Filter blocks to the selected region
    x_low, x_high = params["x_range"]
    y_low, y_high = params["y_range"]
    z_low, z_high = params["z_range"]
    
    region_blocks = [
        b for b in blocks
        if (x_low <= b.x <= x_high and
            y_low <= b.y <= y_high and
            z_low <= b.z <= z_high)
    ]
    
    if not region_blocks:
        st.warning("No blocks in selected region")
        return
    
    # Check block count limit for performance
    if len(region_blocks) > 100:
        st.error(
            f"⚠️ **Too many blocks to visualize precedence graph!**\n\n"
            f"Selected region contains **{len(region_blocks)} blocks**, but the limit is **100 blocks** "
            f"to maintain application performance.\n\n"
            f"**Please narrow your selection using the X, Y, Z range sliders in the sidebar.**\n\n"
            f"**Suggestions:**\n"
            f"- Reduce Z range (e.g., show only 3-4 elevation levels)\n"
            f"- Reduce X or Y range to focus on a smaller area\n"
            f"- Current range: X[{x_low}-{x_high}], Y[{y_low}-{y_high}], Z[{z_low}-{z_high}]"
        )
        return
    
    st.info(f"Showing {len(region_blocks)} blocks in region: X[{x_low}-{x_high}], Y[{y_low}-{y_high}], Z[{z_low}-{z_high}]")
    
    region_ids = {b.id for b in region_blocks}
    
    # Count precedence links within region
    internal_links = 0
    for block in region_blocks:
        for pred in block.predecessors:
            if pred.id in region_ids:
                internal_links += 1
    
    st.metric("Precedence Links in Region", internal_links)
    
    # Create 3D plotly figure
    fig = go.Figure()
    
    # Color blocks by Z level
    z_values = [b.z for b in region_blocks]
    colors = [(z - min(z_values)) / (max(z_values) - min(z_values) + 1) for z in z_values]
    
    # Add blocks
    block_labels = [
        f"Block {b.id}<br>({b.x}, {b.y}, {b.z})<br>"
        f"Grade: {b.grade:.4f}<br>Value: ${b.economic_value:,.0f}<br>"
        f"Predecessors: {len(b.predecessors)}"
        for b in region_blocks
    ]
    
    fig.add_trace(go.Scatter3d(
        x=[b.x for b in region_blocks],
        y=[b.y for b in region_blocks],
        z=[b.z for b in region_blocks],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            colorscale='Viridis',
            opacity=0.8,
            line=dict(color='black', width=0.5)
        ),
        text=block_labels,
        hoverinfo='text',
        name='Blocks'
    ))
    
    # Add precedence arrows (only within region)
    for block in region_blocks:
        for pred in block.predecessors:
            if pred.id in region_ids:
                fig.add_trace(go.Scatter3d(
                    x=[pred.x, block.x],
                    y=[pred.y, block.y],
                    z=[pred.z, block.z],
                    mode='lines',
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Layout
    fig.update_layout(
        title=f"Region Precedence Graph<br><sub>{len(region_blocks)} blocks, {internal_links} precedence links</sub>",
        scene=dict(
            xaxis_title='X (Easting)',
            yaxis_title='Y (Northing)',
            zaxis_title='Z (Elevation)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=700,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics by Z level
    st.subheader("Predecessor Statistics by Z Level")
    z_stats = {}
    for z in range(z_low, z_high + 1):
        z_blocks = [b for b in region_blocks if b.z == z]
        if z_blocks:
            z_stats[z] = {
                "Z Level": z,
                "Block Count": len(z_blocks),
                "Avg Predecessors": sum(len(b.predecessors) for b in z_blocks) / len(z_blocks),
                "Surface Blocks": sum(1 for b in z_blocks if len(b.predecessors) == 0)
            }
    
    if z_stats:
        st.dataframe(pd.DataFrame(z_stats.values()), use_container_width=True)
