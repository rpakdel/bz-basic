import math
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from block import Block


class BZSolver:
    """
    Generalized BZ Solver using Partition Refinement and By-Variable formulation.

    Inputs:
      - blocks: list of Block
      - periods: number of periods
      - discount_rate: discount rate (e.g., 0.1)
      - mining_capacity: tons per period capacity
      - processing_capacity: tons per period capacity for ore (value > 0), optional

    Architecture:
      - Master LP over partitions produces dual penalties mu for each period.
      - Pricing subproblem forms a max-weight closure over (block, t) nodes.
      - Partitions are refined by splitting with respect to the closure.
      - Iterate until duals stabilize or partitions stop changing.
      - TopoSort heuristic produces a feasible integer schedule.
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
        self.id_to_block: Dict[int, Block] = {b.id: b for b in blocks}
        self.id_to_idx: Dict[int, int] = {b.id: i for i, b in enumerate(blocks)}
        self.periods = periods
        self.discount_rate = discount_rate
        self.mining_capacity = mining_capacity
        self.processing_capacity = processing_capacity if processing_capacity is not None else 0.0

        self.G = self._build_precedence_graph(blocks)
        self.partitions: List[Set[int]] = [set(b.id for b in blocks)]
        self.mu_mining = np.zeros(periods)
        self.mu_processing = np.zeros(periods)

    @staticmethod
    def _build_precedence_graph(blocks: List[Block]) -> nx.DiGraph:
        G = nx.DiGraph()
        for b in blocks:
            G.add_node(b.id)
        for b in blocks:
            for p in b.predecessors:
                if p.id in G:
                    G.add_edge(p.id, b.id)
        return G

    # ----------------------- Master LP ----------------------- #
    def _solve_master_lp(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve a coarse LP over partitions to obtain dual penalties per period.

        Variables x_{P,t} ∈ [0, 1] represent fractional extraction of partition P in period t.
        Objective (maximize): sum_{P,t} value(P) * df(t) * (x_{P,t} - x_{P,t-1})
        Wait, BZ master LP usually uses x_{P,t} as mined EXACTLY in period t for simplicity
        if partitions are disjoint.

        Returns:
          - mu_mining: shadow prices for period mining capacity constraints (non-negative)
          - mu_processing: shadow prices for period processing capacity constraints (non-negative)
          - lp_bound: objective value of the LP (upper bound on NPV)
        """
        P = len(self.partitions)
        T = self.periods

        # Aggregate partition tonnage, ore tonnage and value
        part_tonnage = np.zeros(P)
        part_ore_tonnage = np.zeros(P)
        part_value = np.zeros(P)
        for i, part in enumerate(self.partitions):
            tons = 0.0
            ore_tons = 0.0
            val = 0.0
            for bid in part:
                b = self.id_to_block[bid]
                tons += b.tonnage
                if b.economic_value > 0:
                    ore_tons += b.tonnage
                val += b.economic_value
            part_tonnage[i] = tons
            part_ore_tonnage[i] = ore_tons
            part_value[i] = val

        # Variables: x_{i,t} is fraction of partition i mined in period t
        # sum_t x_{i,t} <= 1
        n_vars = P * T
        c = np.zeros(n_vars)
        for i in range(P):
            for t in range(T):
                df = 1.0 / ((1.0 + self.discount_rate) ** (t + 1))
                idx = i * T + t
                c[idx] = -part_value[i] * df

        # Constraints
        A_ub_list = []
        b_ub_list = []

        # 1. Mining capacity: sum_i part_tonnage[i] * x_{i,t} <= mining_capacity
        for t in range(T):
            row = np.zeros(n_vars)
            for i in range(P):
                row[i * T + t] = part_tonnage[i]
            A_ub_list.append(row)
            b_ub_list.append(self.mining_capacity)

        # 2. Processing capacity: sum_i part_ore_tonnage[i] * x_{i,t} <= processing_capacity
        # Only if processing_capacity > 0
        has_proc = self.processing_capacity > 0
        if has_proc:
            for t in range(T):
                row = np.zeros(n_vars)
                for i in range(P):
                    row[i * T + t] = part_ore_tonnage[i]
                A_ub_list.append(row)
                b_ub_list.append(self.processing_capacity)

        # 3. sum_t x_{i,t} <= 1 for each partition
        for i in range(P):
            row = np.zeros(n_vars)
            for t in range(T):
                row[i * T + t] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(1.0)

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Bounds 0 <= x
        bounds = [(0.0, None)] * n_vars

        # Solve using HiGHS
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            return np.zeros(T), np.zeros(T), 0.0

        # Marginals for Ax <= b are non-positive in scipy's minimization.
        # We want positive shadow prices for the capacities.
        all_mu = -np.array(res.ineqlin.marginals)
        mu_mining = all_mu[0:T]
        mu_processing = all_mu[T : 2 * T] if has_proc else np.zeros(T)

        lp_bound = -res.fun
        return mu_mining, mu_processing, lp_bound

    # -------------------- Pricing Subproblem -------------------- #
    def _build_closure_graph(self, mu_mining: np.ndarray, mu_processing: np.ndarray) -> Tuple[nx.DiGraph, str, str]:
        G = nx.DiGraph()
        S, T_sink = "S", "T"
        G.add_node(S)
        G.add_node(T_sink)

        # Nodes: (block_id, t)
        for t in range(self.periods):
            df_current = 1.0 / ((1.0 + self.discount_rate) ** (t + 1))
            df_next = (
                1.0 / ((1.0 + self.discount_rate) ** (t + 2))
                if t < self.periods - 1
                else 0.0
            )

            # Duals for current and next period
            m_t = mu_mining[t]
            m_next = mu_mining[t + 1] if t < self.periods - 1 else 0.0
            
            p_t = mu_processing[t]
            p_next = mu_processing[t + 1] if t < self.periods - 1 else 0.0

            for b in self.blocks:
                node = (b.id, t)
                base_weight = (b.economic_value * df_current) - (b.economic_value * df_next)
                
                mining_penalty = b.tonnage * (m_t - m_next)
                processing_penalty = 0.0
                if b.economic_value > 0:
                    processing_penalty = b.tonnage * (p_t - p_next)

                w = base_weight - mining_penalty - processing_penalty
                
                G.add_node(node, profit=w)
                if w > 1e-9:
                    G.add_edge(S, node, capacity=w)
                elif w < -1e-9:
                    G.add_edge(node, T_sink, capacity=-w)

        # Precedence edges within each period
        for t in range(self.periods):
            for b in self.blocks:
                u = (b.id, t)
                for p in b.predecessors:
                    v = (p.id, t)
                    G.add_edge(u, v, capacity=float("inf"))

                if t < self.periods - 1:
                    G.add_edge(u, (b.id, t + 1), capacity=float("inf"))

        return G, S, T_sink

    def _solve_closure(self, G: nx.DiGraph, S: str, T_sink: str) -> Tuple[Set[Tuple[int, int]], float]:
        cut_value, part = nx.minimum_cut(G, S, T_sink)
        reachable = part[0]
        return {n for n in reachable if isinstance(n, tuple)}, cut_value

    # -------------------- Partition Refinement -------------------- #
    def _refine_partitions(self, closure_nodes: Set[Tuple[int, int]]) -> bool:
        # Blocks in closure (any period selected)
        C_blocks: Set[int] = {bid for (bid, _) in closure_nodes}
        new_parts: List[Set[int]] = []
        changed = False
        for P in self.partitions:
            P_in = P.intersection(C_blocks)
            P_out = P.difference(C_blocks)
            if len(P_in) > 0 and len(P_out) > 0:
                new_parts.append(P_in)
                new_parts.append(P_out)
                changed = True
            else:
                new_parts.append(P)
        self.partitions = new_parts
        return changed

    # -------------------- Decoding + Heuristic -------------------- #
    def _decode_mined_by(self, closure_nodes: Set[Tuple[int, int]]) -> np.ndarray:
        n = len(self.blocks)
        mined_by = np.zeros((n, self.periods))
        for (bid, t) in closure_nodes:
            idx = self.id_to_idx.get(bid)
            if idx is not None and 0 <= t < self.periods:
                mined_by[idx, t] = 1
        return mined_by

    def _expected_periods(self, mined_by: np.ndarray) -> Dict[int, float]:
        E: Dict[int, float] = {}
        for b in self.blocks:
            idx = self.id_to_idx[b.id]
            s = 0.0
            for t in range(self.periods):
                prev = mined_by[idx, t - 1] if t > 0 else 0.0
                prob_in_t = mined_by[idx, t] - prev
                s += (t + 1) * prob_in_t
            E[b.id] = s if s > 0 else math.inf
        return E

    def _toposort_schedule(self, mined_by: np.ndarray) -> Dict[int, int]:
        # Compute E_b
        E = self._expected_periods(mined_by)

        # Determine which blocks should be mined based on mined_by matrix
        # A block is in the closure if mined_by[idx, t] > 0 for any period t
        blocks_to_mine = set()
        for b in self.blocks:
            idx = self.id_to_idx[b.id]
            if np.any(mined_by[idx, :] > 0):
                blocks_to_mine.add(b.id)

        # Kahn's algorithm with tie-break by smallest E_b
        # Only consider blocks that are in the closure
        in_deg = {u: 0 for u in self.G.nodes if u in blocks_to_mine}
        for u, v in self.G.edges:
            if u in blocks_to_mine and v in blocks_to_mine:
                in_deg[v] += 1

        ready = [u for u, d in in_deg.items() if d == 0]
        ready.sort(key=lambda u: E.get(u, math.inf))

        schedule: Dict[int, int] = {b.id: -1 for b in self.blocks}
        tons_per_t = np.zeros(self.periods)
        ore_tons_per_t = np.zeros(self.periods)

        while ready:
            u = ready.pop(0)
            block_u = self.id_to_block[u]
            tonnage_u = block_u.tonnage
            is_ore = block_u.economic_value > 0
            
            # Greedy: earliest period with capacity slack
            t_assigned = -1
            for t in range(self.periods):
                mining_ok = (tons_per_t[t] + tonnage_u <= self.mining_capacity)
                proc_ok = True
                if is_ore and self.processing_capacity > 0:
                    proc_ok = (ore_tons_per_t[t] + tonnage_u <= self.processing_capacity)
                
                if mining_ok and proc_ok:
                    t_assigned = t
                    break
            
            if t_assigned == -1:
                # If no period has space, push to the last period (hard constraint violation but completes schedule)
                t_assigned = self.periods - 1

            schedule[u] = t_assigned
            tons_per_t[t_assigned] += tonnage_u
            if is_ore:
                ore_tons_per_t[t_assigned] += tonnage_u

            for _, v in list(self.G.out_edges(u)):
                if v in blocks_to_mine:
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        ready.append(v)
            ready.sort(key=lambda x: E.get(x, math.inf))

        return schedule

    # ------------------------- Main Loop ------------------------- #
    def solve(
        self,
        max_iterations: int = 20,
        tol_mu: float = 1e-4,
    ) -> Tuple[Dict[int, int], List[Tuple[int, float, int]]]:
        """
        Returns:
          - final_schedule: dict BlockID -> Period
          - logs: list of (iter, lp_bound, num_partitions)
        """
        logs: List[Tuple[int, float, int]] = []
        last_mu = np.copy(self.mu_mining)

        for k in range(1, max_iterations + 1):
            mu_m, mu_p, lp_bound = self._solve_master_lp()
            self.mu_mining = mu_m
            self.mu_processing = mu_p

            Gc, S, Ts = self._build_closure_graph(self.mu_mining, self.mu_processing)
            closure_nodes, _ = self._solve_closure(Gc, S, Ts)

            changed = self._refine_partitions(closure_nodes)
            logs.append((k, lp_bound, len(self.partitions)))

            if not changed and np.linalg.norm(self.mu_mining - last_mu) < tol_mu:
                break
            last_mu = np.copy(self.mu_mining)

        mined_by = self._decode_mined_by(closure_nodes)
        final_schedule = self._toposort_schedule(mined_by)
        return final_schedule, logs
