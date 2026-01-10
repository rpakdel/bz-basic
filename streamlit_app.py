"""
BZ Scheduler Streamlit Application
Main entry point - unified UI for block model visualization and BZ optimization.
"""
import matplotlib
matplotlib.use("Agg")  # headless-friendly backend

import streamlit as st

from ui.view_model import show_view_model_sidebar, show_view_model_content, show_precedence_visualization
from ui.bz_demo import show_bz_demo_section


def main():
    """Main application entry point."""
    st.title("BZ Scheduler Demo (Streamlit)")
    st.markdown("Visualize block models and run BZ optimization.")

    # Sidebar: Block model selection and visualization options
    with st.sidebar:
        view_params = show_view_model_sidebar()

    # Main content area
    if view_params["selected_name"] is None:
        st.error("Pick a CSV from the sidebar to get started.")
        return

    # Show block model visualization and statistics
    show_view_model_content(view_params)

    # Precedence Graph Visualization section
    st.divider()
    st.header("Precedence Graph Visualization")
    
    show_precedence = st.checkbox(
        "Show Precedence Relationships", 
        value=False, 
        key="main_precedence",
        help="Visualize block dependencies based on 1:3 slope constraints"
    )
    
    if show_precedence:
        from pathlib import Path
        data_dir = Path("data")
        selected_path = data_dir / view_params["selected_name"]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            precedence_mode = st.radio(
                "Visualization Mode",
                options=["Single Block", "Region View"],
                key="main_precedence_mode",
                help="Single Block: Focus on one block and its predecessors\nRegion View: Show all precedence in filtered range"
            )
        
        with col2:
            precedence_block_id = None
            if precedence_mode == "Single Block":
                precedence_block_id = st.number_input(
                    "Block ID to visualize",
                    min_value=0,
                    value=1000,
                    step=1,
                    key="main_precedence_block_id",
                    help="Enter block ID to see its predecessors (blocks that must be mined first)"
                )
            else:
                st.info("Using X, Y, Z ranges from sidebar to define region")
        
        # Build params dict for precedence visualization
        precedence_params = {
            **view_params,
            "show_precedence": show_precedence,
            "precedence_mode": precedence_mode,
            "precedence_block_id": precedence_block_id if precedence_mode == "Single Block" else None,
        }
        
        show_precedence_visualization(precedence_params, selected_path)

    # BZ Demo optimization section
    st.divider()
    show_bz_demo_section(view_params)


if __name__ == "__main__":
    main()

