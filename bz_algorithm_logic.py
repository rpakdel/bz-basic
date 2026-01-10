from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import time

from block import Block
from bz_graph import build_closure_graph


class BZScheduler:
    """
    Implements a simplified version of the Bienstock-Zuckerberg (BZ) approach
    for educational purposes.

    The Core Logic:
    1. The 'Hard' problem is the Resource Constraint (e.g., Mill Capacity per year).
    2. We relax this constraint using Lagrangian Multipliers (penalties).
    3. The problem transforms into a 'Maximum Weight Closure' problem.
    4. We solve the Closure problem efficiently using Min-Cut / Max-Flow.
    5. We iteratively update the multipliers (penalties) until the solution converges
       or satisfies constraints reasonably well.
    """
    def __init__(
        self,
        blocks: List[Block],
        periods: int,
        discount_rate: float,
        mining_capacity: float,
        processing_capacity: float,
    ) -> None:
        self.blocks = blocks
        self.periods = periods
        self.discount_rate = discount_rate
        self.mining_limit = mining_capacity
        self.processing_limit = processing_capacity

        # Lagrangian Multipliers: One for each period
        # Initialize with 0 penalty
        self.lambda_mining = np.zeros(periods)
        self.lambda_processing = np.zeros(periods)

    def solve(
        self,
        max_iterations: int = 20,
        step_size_factor: float = 0.00005,
    ) -> Tuple[Dict[int, int], List[Tuple[int, float, float]]]:
        """
        The Main BZ Loop.
        """
        print(f"--- Starting BZ Optimization ({self.periods} periods) ---")
        start_time = time.time()

        best_objective = float('inf')
        history: List[Tuple[int, float, float]] = []

        for k in range(1, max_iterations + 1):
            # 1. Build Graph with current multipliers
            G, source, sink = build_closure_graph(
                self.blocks,
                self.periods,
                self.discount_rate,
                self.lambda_mining,
                self.lambda_processing,
            )

            # 2. Solve Min-Cut (Equivalent to Max Weight Closure)
            # value is the capacity of the cut
            cut_value, partition = nx.minimum_cut(G, source, sink)

            # Partition[0] is the set of nodes reachable from Source (The set we keep/mine)
            reachable = partition[0]

            # 3. Decode Solution
            # x[b][t] = 1 if (b,t) is in reachable set
            current_schedule: Dict[int, int] = {} # block_id -> period extracted (or None)

            # Track resource usage per period to calculate violations
            mining_usage = np.zeros(self.periods)
            processing_usage = np.zeros(self.periods)

            mined_by_matrix = np.zeros((len(self.blocks), self.periods))

            for block in self.blocks:
                for t in range(self.periods):
                    if (block.id, t) in reachable:
                        mined_by_matrix[block.id, t] = 1

            # Convert "Mined By" to "Mined In"
            # Mined In t = (Mined By t) - (Mined By t-1)
            actual_profit = 0

            for block in self.blocks:
                extract_period = -1
                for t in range(self.periods):
                    is_mined_by_t = mined_by_matrix[block.id, t]
                    is_mined_by_prev = mined_by_matrix[block.id, t-1] if t > 0 else 0

                    if is_mined_by_t == 1 and is_mined_by_prev == 0:
                        extract_period = t
                        # Tally usage
                        mining_usage[t] += block.tonnage
                        if block.economic_value > 0:
                            processing_usage[t] += block.tonnage

                        # Add to NPV
                        df = 1 / ((1 + self.discount_rate) ** (t + 1))
                        actual_profit += block.economic_value * df
                        break

                current_schedule[block.id] = extract_period

            # 4. Calculate Subgradients (Violations)
            # Gradient > 0 implies we used too much resource -> Increase Penalty
            # Gradient < 0 implies we have slack -> Decrease Penalty
            grad_mining = mining_usage - self.mining_limit
            grad_processing = processing_usage - self.processing_limit

            # 5. Update Multipliers (Subgradient Method)
            # Step size usually decays (1/k) or uses Polyak's rule.
            # Using a simple decaying step size for demo stability.
            step = step_size_factor / (k**0.5)

            self.lambda_mining = np.maximum(0, self.lambda_mining + step * grad_mining)
            self.lambda_processing = np.maximum(0, self.lambda_processing + step * grad_processing)

            # Log status
            total_violation = np.sum(np.maximum(0, grad_mining)) + np.sum(np.maximum(0, grad_processing))
            history.append((k, actual_profit, total_violation))

            print(f"Iter {k:02d} | Profit: ${actual_profit:,.0f} | Violations: {total_violation:,.0f} tons")

            if total_violation == 0 and k > 5:
                print("Converged to feasible solution.")
                break

        print(f"--- Finished in {time.time() - start_time:.2f}s ---")
        return current_schedule, history
