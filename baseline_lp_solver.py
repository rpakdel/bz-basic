"""
Baseline Linear Programming (LP) Solver for Production Scheduling.
Uses scipy.optimize.linprog (HiGHS) to solve the exact PCPSP problem.
Provides a standard baseline to compare BZ results against the global optimal relaxed solution.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog
from block import Block

class LPSolver:
    """
    Standard LP Solver for the Precedence-Constrained Production Scheduling Problem (PCPSP).
    This solver handles every block individually in the LP, which is the mathematically 
    optimal "relaxed" solution but scales poorly for very large models.
    """

    def __init__(
        self,
        blocks: List[Block],
        periods: int,
        discount_rate: float,
        mining_capacity: float,
        processing_capacity: Optional[float] = None,
    ):
        self.blocks = blocks
        self.periods = periods
        self.discount_rate = discount_rate
        self.mining_capacity = mining_capacity
        self.processing_capacity = processing_capacity if processing_capacity is not None else 0.0
        
        # Map block IDs to internal indices
        self.id_to_idx = {b.id: i for i, b in enumerate(blocks)}
        self.n = len(blocks)
        self.T = periods

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Solve the full PCPSP using a multi-period LP formulation.
        
        Variables: x_{b, t} is the fraction of block b mined in period t.
        Total variables = N_blocks * T_periods
        """
        N = self.n
        T = self.T
        num_vars = N * T

        # 1. Objective Function (Maximize NPV)
        # Minimize -NPV
        c = np.zeros(num_vars)
        for i, b in enumerate(self.blocks):
            for t in range(T):
                df = 1.0 / ((1.0 + self.discount_rate) ** (t + 1))
                idx = i * T + t
                c[idx] = -b.economic_value * df

        A_ub_list = []
        b_ub_list = []

        # 2. Exclusivity Constraints: sum_t x_{b, t} <= 1
        for i in range(N):
            row = np.zeros(num_vars)
            for t in range(T):
                row[i * T + t] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(1.0)

        # 3. Mining Capacity per Period: sum_b tonnage_b * x_{b, t} <= mining_cap
        for t in range(T):
            row = np.zeros(num_vars)
            for i in range(N):
                row[i * T + t] = self.blocks[i].tonnage
            A_ub_list.append(row)
            b_ub_list.append(self.mining_capacity)

        # 4. Processing Capacity per Period: sum_b tonnage_b * x_{b, t} <= proc_cap (if ore)
        if self.processing_capacity > 0:
            for t in range(T):
                row = np.zeros(num_vars)
                for i in range(N):
                    if self.blocks[i].economic_value > 0:
                        row[i * T + t] = self.blocks[i].tonnage
                A_ub_list.append(row)
                b_ub_list.append(self.processing_capacity)

        # 5. Precedence Constraints: 
        # For each block b and its predecessors p:
        # MinedBy_b,t <= MinedBy_p,t
        # sum_{tau=1}^t x_{b, tau} <= sum_{tau=1}^t x_{p, tau}
        # sum_{tau=1}^t x_{b, tau} - sum_{tau=1}^t x_{p, tau} <= 0
        for i, b in enumerate(self.blocks):
            for p in b.predecessors:
                if p.id in self.id_to_idx:
                    p_idx = self.id_to_idx[p.id]
                    for t in range(T):
                        row = np.zeros(num_vars)
                        for tau in range(t + 1):
                            row[i * T + tau] = 1.0
                            row[p_idx * T + tau] = -1.0
                        A_ub_list.append(row)
                        b_ub_list.append(0.0)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Bounds: 0 <= x_{b, t} <= 1
        bounds = [(0.0, 1.0)] * num_vars

        # Solve using HiGHS
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            print(f"LP Solver Error: {res.message}")
            return np.zeros((N, T)), 0.0

        # Reshape solution to (N, T)
        solution = res.x.reshape((N, T))
        npv = -res.fun
        
        return solution, npv

def decode_lp_to_schedule(solution: np.ndarray, b_ids: List[int], threshold: float = 0.5) -> Dict[int, int]:
    """
    Simple rounding heuristic to convert fractional LP solution to integer schedule.
    Note: Highly unlikely to maintain precedence perfectly. 
    Mainly used for visualization of 'where' the LP thinks blocks belong.
    """
    schedule = {}
    N, T = solution.shape
    for i in range(N):
        assigned = False
        # Find period with highest extraction
        t_max = np.argmax(solution[i, :])
        if solution[i, t_max] > 1e-3:
            schedule[b_ids[i]] = t_max
        else:
            schedule[b_ids[i]] = -1
    return schedule
