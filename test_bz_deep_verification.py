import numpy as np
import pytest
from block import Block
from bz_partition_refinement import BZSolver
from unittest.mock import MagicMock

def test_dual_extraction_consistency():
    """
    Verify that _solve_master_lp correctly extracts duals from scipy.optimize.linprog.
    In scipy, marginals for Ax <= b are non-positive (shadow prices for maximization 
    should be positive).
    """
    # Create 1 block, 1 period, 1 capacity
    b1 = Block(1, 0, 0, 0, tonnage=100, grade=1.0, value=1000)
    solver = BZSolver([b1], periods=1, discount_rate=0.0, mining_capacity=50) # Tonnage > Capacity
    
    # We shouldn't rely on actually solving, since we want to test the *extraction* logic
    # But let's see what it does
    mu_mining, mu_processing, lp_bound = solver._solve_master_lp()
    
    # Since tonnage (100) > capacity (50), the constraint is binding.
    # Shadow price should be the profit per ton = 1000 / 100 = 10.
    # Note: Scipy's highs solver might return slightly different depending on formulation,
    # but it should definitely be positive.
    assert mu_mining[0] > 0
    assert np.isclose(mu_mining[0], 10.0, atol=1e-5)

def test_subproblem_weight_math():
    """
    Verify the weight of nodes in the closure graph correctly accounts for:
    1. Discounted profit (and its delta across periods)
    2. Mining duals (and their delta)
    3. Processing duals (and their delta)
    """
    # Block: 100 tons, $1000 value (ore)
    b1 = Block(1, 0, 0, 0, tonnage=100, grade=1.0, value=1000)
    # 2 periods, no discount for simplicity
    solver = BZSolver([b1], periods=2, discount_rate=0.0, mining_capacity=1000)
    
    # Mock duals:
    # Period 0: high penalty (mu=10)
    # Period 1: low penalty (mu=2)
    mu_mining = np.array([10.0, 2.0])
    mu_processing = np.array([5.0, 1.0])
    
    G, S, T_sink = solver._build_closure_graph(mu_mining, mu_processing)
    
    # Weight for Period 0 (node 1, 0)
    # Base weight = Profit(t=0) - Profit(t=1) = 1000 - 1000 = 0
    # Mining penalty = Tonnage * (mu_m[0] - mu_m[1]) = 100 * (10 - 2) = 800
    # Processing penalty = Tonnage * (mu_p[0] - mu_p[1]) = 100 * (5 - 1) = 400
    # Total Weight = 0 - 800 - 400 = -1200
    
    node0 = (1, 0)
    assert G.nodes[node0]['profit'] == -1200.0

    # Weight for Period 1 (node 1, 1) - Last period
    # Base weight = Profit(t=1) - Profit(t=2) = 1000 - 0 = 1000
    # Mining penalty = Tonnage * (mu_m[1] - mu_m[2]) = 100 * (2 - 0) = 200
    # Processing penalty = Tonnage * (mu_p[1] - mu_p[2]) = 100 * (1 - 0) = 100
    # Total Weight = 1000 - 200 - 100 = 700
    
    node1 = (1, 1)
    assert G.nodes[node1]['profit'] == 700.0

def test_closure_graph_edges():
    """
    Verify that the closure graph contains correct precedence and temporal edges.
    """
    b1 = Block(1, 0, 0, 1, tonnage=100, grade=1.0, value=1000) # Below
    b2 = Block(2, 0, 0, 0, tonnage=100, grade=1.0, value=1000) # Above
    b1.add_predecessor(b2)
    
    solver = BZSolver([b1, b2], periods=2, discount_rate=0.0, mining_capacity=1000)
    G, S, T_sink = solver._build_closure_graph(np.zeros(2), np.zeros(2))
    
    # Spatial Precedence: (b1, t) -> (b2, t)
    assert G.has_edge((1, 0), (2, 0))
    assert G.has_edge((1, 1), (2, 1))
    assert G.get_edge_data((1, 0), (2, 0))['capacity'] == float('inf')
    
    # Temporal Precedence: (b, t) -> (b, t+1)
    assert G.has_edge((1, 0), (1, 1))
    assert G.has_edge((2, 0), (2, 1))
    assert G.get_edge_data((1, 0), (1, 1))['capacity'] == float('inf')

@pytest.mark.parametrize("has_proc", [True, False])
def test_master_lp_objective(has_proc):
    """
    Verify that the Master LP objective value matches the sum of 
    partition values when capacity is infinite.
    """
    b1 = Block(1, 0, 0, 0, tonnage=100, grade=1.0, value=1000)
    b2 = Block(2, 0, 0, 0, tonnage=100, grade=1.0, value=2000)
    
    # Infinity capacity, no discount
    proc_cap = 5000 if has_proc else None
    solver = BZSolver([b1, b2], periods=1, discount_rate=0.0, mining_capacity=5000, processing_capacity=proc_cap)
    
    mu_m, mu_p, lp_bound = solver._solve_master_lp()
    
    assert np.isclose(lp_bound, 3000.0)
    assert np.all(mu_m == 0)
    assert np.all(mu_p == 0)

def test_partition_refinement_splitting():
    """
    Test that _refine_partitions correctly splits partitions based on closure.
    """
    b1 = Block(1, 0, 0, 0, tonnage=100, grade=1.0, value=10)
    b2 = Block(2, 0, 0, 0, tonnage=100, grade=1.0, value=10)
    b3 = Block(3, 0, 0, 0, tonnage=100, grade=1.0, value=10)
    
    solver = BZSolver([b1, b2, b3], periods=1, discount_rate=0.0, mining_capacity=1000)
    
    # Initial: [{1, 2, 3}]
    assert len(solver.partitions) == 1
    
    # Case 1: Partial split. Closure nodes: {(1, 0), (2, 0)} - Block 1 and 2 are in.
    closure = {(1, 0), (2, 0)}
    changed = solver._refine_partitions(closure)
    
    assert changed is True
    assert len(solver.partitions) == 2
    # Verify contents (order might differ but sets should be {1, 2} and {3})
    partition_sets = [set(p) for p in solver.partitions]
    assert {1, 2} in partition_sets
    assert {3} in partition_sets
    
    # Case 2: No split. Closure nodes {(1, 0)} only contains blocks from ONE current partition {1, 2}
    # It should split {1, 2} into {1} and {2}.
    closure2 = {(1, 0)}
    changed2 = solver._refine_partitions(closure2)
    assert changed2 is True
    assert len(solver.partitions) == 3
    
    # Case 3: All blocks included. No split.
    closure_all = {(1, 0), (2, 0), (3, 0)}
    changed3 = solver._refine_partitions(closure_all)
    assert changed3 is False
    assert len(solver.partitions) == 3
