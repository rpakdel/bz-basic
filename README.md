# BZ-Basic: Mining Schedule Optimization

`bz-basic` is a high-performance demonstration of the **Bienstock-Zuckerberg (BZ) algorithm** applied to Open-Pit Mine Production Scheduling.

Optimization in mining involves deciding *when* to mine each block to maximize Net Present Value (NPV), subject to:
1. **Precedence Constraints**: You must mine the blocks above before you can mine the blocks below.
2. **Resource Constraints**: Limits on total tonnage mined (Mining Capacity) and ore tonnage processed (Processing Capacity) per period.

## Features
- **BZ Solver**: Implementation using Partition Refinement and Max-Weight Closure.
- **Global LP Baseline**: Integrated `scipy.optimize.linprog` (HiGHS) solver for ground-truth comparison and optimality gap analysis.
- **Interactive UI**: Streamlit-based dashboard for visualizing block models and side-by-side optimization benchmarking.
- **Synthetic Generator**: Create realistic block models (multi-blob) using Perlin noise.
- **Precedence Visualization**: Interactive 3D visualization of block dependencies.

## Getting Started

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

### Generating Data
If you want to create a new synthetic block model:
```bash
python generate_block_model.py
```
This will create a CSV and JSON metadata file in the `data/` directory.

## How it Works
The BZ algorithm decomposes the massive scheduling problem into two parts:
1. A **Master LP** that works on "partitions" (groups of blocks) to find approximate dual prices for the resources.
2. A **Pricing Subproblem** that solves a Maximum Weight Closure problem to find the optimal set of blocks to mine, given those prices.

The algorithm iteratively refines the partitions until it converges on a near-optimal solution. Finally, a heuristic (Topological Sort) is used to convert the fractional solution into a practical, period-by-period mining schedule.

## Project Structure
- `bz_partition_refinement.py`: The core BZ solver logic.
- `streamlit_app.py`: Main UI entry point.
- `ui/`: Modular UI components for different modes.
- `data/`: Sample CSV block models.
- `utils/`: Plotting and data processing utilities.

## License
MIT

