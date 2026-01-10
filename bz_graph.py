from typing import List, Tuple

import networkx as nx
import numpy as np

from block import Block


def build_closure_graph(
    blocks: List[Block],
    periods: int,
    discount_rate: float,
    multipliers_mining: np.ndarray,
    multipliers_processing: np.ndarray,
) -> Tuple[nx.DiGraph, str, str]:
    """Construct the flow network for the closure problem.

    Nodes are (block_id, period_index) using the "mined by" formulation.
    Returns the directed graph plus source and sink identifiers.
    """
    G = nx.DiGraph()

    source = "S"
    sink = "T"
    G.add_node(source)
    G.add_node(sink)

    for t in range(periods):
        df_current = 1 / ((1 + discount_rate) ** (t + 1))
        df_next = 1 / ((1 + discount_rate) ** (t + 2)) if t < periods - 1 else 0

        current_mining_penalty = multipliers_mining[t]
        current_proc_penalty = multipliers_processing[t]

        for block in blocks:
            node_id = (block.id, t)

            npv_t = block.economic_value * df_current
            npv_next = block.economic_value * df_next if t < periods - 1 else 0
            base_weight = npv_t - npv_next

            resource_penalty = block.tonnage * current_mining_penalty
            if block.economic_value > 0:
                resource_penalty += block.tonnage * current_proc_penalty

            weight = base_weight - resource_penalty

            G.add_node(node_id, profit=weight)
            if weight > 0:
                G.add_edge(source, node_id, capacity=weight)
            elif weight < 0:
                G.add_edge(node_id, sink, capacity=-weight)

    for t in range(periods):
        for block in blocks:
            u = (block.id, t)
            for pred in block.predecessors:
                v = (pred.id, t)
                G.add_edge(u, v, capacity=float("inf"))

            if t < periods - 1:
                next_period_node = (block.id, t + 1)
                G.add_edge(u, next_period_node, capacity=float("inf"))

    return G, source, sink
