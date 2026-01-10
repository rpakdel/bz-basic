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
    def _solve_master_lp(self) -> Tuple[np.ndarray, float]:
        """
        Solve a coarse LP over partitions to obtain dual penalties per period.

        Variables x_{P,t} ∈ [0, 1] represent fractional extraction of partition P in period t.
        Objective (maximize): sum_{P,t} value(P) * df(t) * x_{P,t}
        Subject to per-period mining capacity: sum_{P} tonnage(P) * x_{P,t} <= cap_t

        Returns:
          - mu_mining: shadow prices for period capacity constraints
          - lp_bound: objective value of the LP (upper bound on NPV)
        """
        P = len(self.partitions)
        T = self.periods

        # Aggregate partition tonnage and value
        part_tonnage = np.zeros(P)
        part_value = np.zeros(P)
        for i, part in enumerate(self.partitions):
            tons = 0.0
            val = 0.0
            for bid in part:
                b = self.id_to_block[bid]
                tons += b.tonnage
                val += b.economic_value
            part_tonnage[i] = tons
            part_value[i] = val

        # Build LP in canonical form for linprog (minimize c^T x)
        # We maximize value by minimizing -value
        n_vars = P * T
        c = np.zeros(n_vars)
        for i in range(P):
            for t in range(T):
                df = 1.0 / ((1.0 + self.discount_rate) ** (t + 1))
                idx = i * T + t
                c[idx] = -part_value[i] * df

        # A_ub x <= b_ub (capacity per period)
        A_ub = np.zeros((T, n_vars))
        b_ub = np.full(T, self.mining_capacity)
        for t in range(T):
            for i in range(P):
                idx = i * T + t
                A_ub[t, idx] = part_tonnage[i]

        # Bounds 0 <= x <= 1
        bounds = [(0.0, 1.0)] * n_vars

        # Solve using HiGHS to obtain duals
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            # Fallback: zero duals, zero bound
            return np.zeros(T), 0.0

        # HiGHS returns marginals for inequality constraints in res.ineqlin.marginals
        try:
            mu = np.array(res.ineqlin.marginals)
        except Exception:
            mu = np.zeros(T)

        lp_bound = -res.fun
        return mu, lp_bound

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

            for b in self.blocks:
                node = (b.id, t)
                base_weight = (b.economic_value * df_current) - (b.economic_value * df_next)

                penalty = b.tonnage * mu_mining[t]
                if b.economic_value > 0 and mu_processing is not None and len(mu_processing) == self.periods:
                    penalty += b.tonnage * mu_processing[t]

                w = base_weight - penalty
                G.add_node(node, profit=w)
                if w > 0:
                    G.add_edge(S, node, capacity=w)
                elif w < 0:
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

        # Kahn's algorithm with tie-break by smallest E_b
        in_deg = {u: 0 for u in self.G.nodes}
        for u, v in self.G.edges:
            in_deg[v] += 1

        ready = [u for u, d in in_deg.items() if d == 0]
        ready.sort(key=lambda u: E.get(u, math.inf))

        schedule: Dict[int, int] = {b.id: -1 for b in self.blocks}
        tons_per_t = np.zeros(self.periods)

        while ready:
            u = ready.pop(0)
            # Greedy: earliest period with capacity slack
            t_assigned = -1
            tonnage_u = self.id_to_block[u].tonnage
            for t in range(self.periods):
                if tons_per_t[t] + tonnage_u <= self.mining_capacity:
                    t_assigned = t
                    break
            if t_assigned == -1:
                t_assigned = self.periods - 1

            schedule[u] = t_assigned
            tons_per_t[t_assigned] += tonnage_u

            for _, v in list(self.G.out_edges(u)):
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
            mu, lp_bound = self._solve_master_lp()
            self.mu_mining = mu

            # Optional: simple processing duals equal to mining duals for demo
            self.mu_processing = np.copy(self.mu_mining) if self.processing_capacity > 0 else np.zeros(self.periods)

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
