"""
BZ Scheduler Streamlit Application
Main entry point with modular UI components.
"""
import matplotlib
matplotlib.use("Agg")  # headless-friendly backend

import streamlit as st

from ui.view_model import show_view_model_sidebar, show_view_model_content
from ui.synthetic_mode import show_synthetic_sidebar, show_synthetic_results
from ui.csv_mode import show_csv_sidebar, show_csv_results


def main():
    """Main application entry point."""
    st.title("BZ Scheduler Demo (Streamlit)")
    st.markdown("Compare the simple BZ demo vs. generalized CSV-based BZ with partition refinement.")

    with st.sidebar:
        st.header("Mode")
        mode = st.selectbox(
            "Algorithm",
            ["View Block Model", "Simple Demo (Synthetic)", "Generalized BZ (CSV)"]
        )

        # Mode-specific sidebar controls
        if mode == "View Block Model":
            view_params = show_view_model_sidebar()
        elif mode == "Simple Demo (Synthetic)":
            periods = st.number_input("Periods", min_value=2, max_value=10, value=4, step=1)
            discount_rate = st.slider("Discount Rate", min_value=0.0, max_value=0.25, value=0.10, step=0.01)
            st.header("Capacity Settings")
            mining_cap = st.number_input("Mining Capacity (t/period)", min_value=1000, max_value=200_000, value=30_000, step=1000)
            synth_params = show_synthetic_sidebar()
        else:  # Generalized BZ (CSV)
            periods = st.number_input("Periods", min_value=2, max_value=10, value=4, step=1)
            discount_rate = st.slider("Discount Rate", min_value=0.0, max_value=0.25, value=0.10, step=0.01)
            st.header("Capacity Settings")
            mining_cap = st.number_input("Mining Capacity (t/period)", min_value=1000, max_value=200_000, value=30_000, step=1000)
            csv_params = show_csv_sidebar(periods, mining_cap)

    # Main content area
    if mode == "View Block Model":
        show_view_model_content(view_params)
    elif mode == "Simple Demo (Synthetic)":
        if st.button("Run Optimization", type="primary"):
            show_synthetic_results(synth_params, periods, discount_rate, mining_cap)
    elif mode == "Generalized BZ (CSV)":
        if st.button("Run Optimization", type="primary"):
            show_csv_results(csv_params, periods, discount_rate, mining_cap)
    else:
        st.info("Set parameters on the left, select algorithm, then click Run Optimization.")


if __name__ == "__main__":
    main()

