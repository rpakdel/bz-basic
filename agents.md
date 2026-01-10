# BZ-Basic: AI Agent Technical Guide

This document provides a technical overview of the BZ-Basic project for AI agents. It describes the architecture, the Bienstock-Zuckerberg (BZ) algorithm implementation, and how the different components interact.

## Project Context
`bz-basic` is a prototype for Open-Pit Mine Production Scheduling Optimization using the Bienstock-Zuckerberg algorithm. This algorithm is designed to solve Large-Scale Precedence-Constrained Production Scheduling Problems (PCPSP).

## Core Algorithm: Bienstock-Zuckerberg (BZ)
The BZ algorithm implemented in [bz_partition_refinement.py](bz_partition_refinement.py) follows a decomposition approach:

### 1. Master Problem (Master LP)
- **Purpose**: Solve a relaxed version of the scheduling problem using aggregated groups of blocks called "partitions".
- **Formulation**: By-period. $x_{P,t}$ is the fraction of partition $P$ mined specifically in period $t$.
- **Constraints**: 
    - Mining Capacity (Total tonnage).
    - Processing Capacity (Ore tonnage).
    - Mass balance ($\sum_t x_{P,t} \le 1$).
- **Output**: Dual variables (shadow prices) $\mu_{mining,t}$ and $\mu_{processing,t}$ which represent the cost of constraint violations.

### 2. Pricing Subproblem (Max-Weight Closure)
- **Purpose**: Find a set of blocks that maximizes profit minus dual penalties while respecting precedence.
- **Formulation**: By-variable (Cumulative). $y_{b,t}=1$ if block $b$ is mined *by* period $t$.
- **Weights**: The coefficient for $(b, t)$ is derived from the marginal profit of mining in $t$ vs $t+1$, minus the resource penalties.
- **Implementation**: uses `networkx.minimum_cut` to solve the Max-Weight Closure problem.
- **Graph Structure**:
    - **Nodes**: $(block, t)$ for each period.
    - **Precedence Edges**: $(block, t) \to (predecessor, t)$ (infinite capacity).
    - **Temporal Edges**: $(block, t) \to (block, t+1)$ (infinite capacity).

### 3. Partition Refinement
- If the pricing subproblem selects a set of blocks $C$, every existing partition $P$ is checked.
- If $P$ contains blocks both in $C$ and not in $C$, it is split into two new partitions.
- This allows the Master LP to become increasingly granular only where the optimal solution is likely to change.

### 4. Integer Heuristic (TopoSort)
- Since the BZ solution can be fractional, [bz_partition_refinement.py](bz_partition_refinement.py) uses a **Topological Sort Greedy Heuristic**.
- It calculates an "Expected Period" $E_b$ for each block based on the fractional solution.
- It then processes blocks in increasing order of $E_b$, respecting precedence and hard capacity limits to produce a feasible mining schedule.

## Key Files
- [block.py](block.py): Data structure for a single mining block.
- [bz_partition_refinement.py](bz_partition_refinement.py): Primary implementation of the BZ algorithm.
- [deposit_utils.py](deposit_utils.py): Utilities for reading CSV/JSON block models and building precedence.
- [streamlit_app.py](streamlit_app.py): Entry point for the UI.
- [generate_block_model.py](generate_block_model.py): Synthetic data generator using Perlin noise.

## Implementation Details for Agents
- When modifying the solver, ensure the dual price extraction from `scipy.optimize.linprog` (HiGHS) remains consistent. Scipy returns non-positive marginals for `A_ub @ x <= b`, so they are negated to get positive shadow prices.
- The closure graph profit calculation handles both mining and processing duals. If `economic_value > 0`, the block is considered ore and subject to processing constraints.
- The current implementation assumes a 1:1 slope (mining a block requires 3-5 blocks above it) which is handled in [deposit_utils.py](deposit_utils.py).

## Testing
Unit tests are available in:
- [test_bz_partition_refinement.py](test_bz_partition_refinement.py)
- [test_precedence.py](test_precedence.py)
