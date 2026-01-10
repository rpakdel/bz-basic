"""
Comparison tool for BZ vs Full LP (Optimal) scheduling.
"""
import time
import numpy as np
from bz_partition_refinement import BZSolver
from baseline_lp_solver import LPSolver
from deposit_utils import read_block_model_csv

def main():
    # Load a small model for fast comparison
    # If the user doesn't have 10_10_10, use whatever is in data/
    csv_path = "data/block_model_10_10_10.csv"
    try:
        blocks, x_size, y_size, z_size = read_block_model_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} not found. Running generate_block_model.py first...")
        import subprocess
        subprocess.run(["python", "generate_block_model.py"])
        # We need to find the name of the file generated or use the default
        blocks, x_size, y_size, z_size = read_block_model_csv("data/block_model_25_25_10.csv")
        # Take a subset for the full LP to not hang
        blocks = blocks[:500]

    PERIODS = 3
    DISCOUNT = 0.1
    MINING_CAP = 30000
    PROC_CAP = 15000

    print(f"--- Comparing Solvers ({len(blocks)} blocks, {PERIODS} periods) ---")

    # 1. Run BZ Solver
    print("\n[BZ Solver]")
    bz_solver = BZSolver(
        blocks=blocks,
        periods=PERIODS,
        discount_rate=DISCOUNT,
        mining_capacity=MINING_CAP,
        processing_capacity=PROC_CAP
    )
    
    t0 = time.time()
    bz_schedule, bz_logs = bz_solver.solve(max_iterations=20)
    t_bz = time.time() - t0
    
    # Calculate BZ LP bound from logs (the last iteration)
    bz_lp_bound = bz_logs[-1][1] if bz_logs else 0

    print(f"BZ Time: {t_bz:.2f}s")
    print(f"BZ Upper Bound (Relaxed): ${bz_lp_bound:,.2f}")

    # 2. Run Full LP Solver
    print("\n[Full LP Solver (HiGHS)]")
    lp_solver = LPSolver(
        blocks=blocks,
        periods=PERIODS,
        discount_rate=DISCOUNT,
        mining_capacity=MINING_CAP,
        processing_capacity=PROC_CAP
    )
    
    t0 = time.time()
    lp_solution, lp_npv = lp_solver.solve()
    t_lp = time.time() - t0

    print(f"Full LP Time: {t_lp:.2f}s")
    print(f"Full LP Optimal NPV (Relaxed): ${lp_npv:,.2f}")

    # 3. Comparison
    gap = (lp_npv - bz_lp_bound) / lp_npv if lp_npv > 0 else 0
    print("\n[Comparison Summary]")
    print(f"Optimality Gap (Relaxed): {gap:.4%}")
    print(f"Speedup Factor: {t_lp / t_bz if t_bz > 0 else 0:.1f}x")

    if gap < 0.001:
        print("✅ BZ implementation is highly consistent with full LP solution.")
    else:
        print("⚠️ Significant gap detected. Check partition refinement settings.")

if __name__ == "__main__":
    main()
