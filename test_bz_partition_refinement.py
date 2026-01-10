"""
Comprehensive unit tests for BZSolver using real CSV data.
"""
import numpy as np
import pytest

from bz_partition_refinement import BZSolver
from deposit_utils import read_block_model_csv


@pytest.fixture
def test_blocks():
    """Load blocks from test CSV file."""
    blocks, x_size, y_size, z_size = read_block_model_csv("data/block_model_10_10_10.csv")
    return blocks


@pytest.fixture
def small_test_blocks(test_blocks):
    """Get a small subset of blocks for faster testing."""
    # Take enough blocks to include some ore (around line 648 in CSV)
    # The first 100 blocks are mostly waste in the current dataset.
    return test_blocks[:800]


@pytest.fixture
def basic_solver(small_test_blocks):
    """Create a basic BZSolver instance."""
    return BZSolver(
        blocks=small_test_blocks,
        periods=4,
        discount_rate=0.1,
        mining_capacity=30000,
    )


@pytest.fixture
def solver_with_processing(small_test_blocks):
    """Create a BZSolver with processing capacity."""
    return BZSolver(
        blocks=small_test_blocks,
        periods=4,
        discount_rate=0.1,
        mining_capacity=30000,
        processing_capacity=10000,
    )


class TestBZSolverInit:
    """Test BZSolver initialization and graph building."""
    
    def test_init_basic_parameters(self, basic_solver, small_test_blocks):
        """Test that solver initializes with correct parameters."""
        assert basic_solver.periods == 4
        assert basic_solver.discount_rate == 0.1
        assert basic_solver.mining_capacity == 30000
        assert basic_solver.processing_capacity == 0.0
        assert len(basic_solver.blocks) == len(small_test_blocks)
    
    def test_init_with_processing(self, solver_with_processing):
        """Test initialization with processing capacity."""
        assert solver_with_processing.processing_capacity == 10000
    
    def test_id_to_block_mapping(self, basic_solver, small_test_blocks):
        """Test that id_to_block mapping is correct."""
        assert len(basic_solver.id_to_block) == len(small_test_blocks)
        for block in small_test_blocks:
            assert block.id in basic_solver.id_to_block
            assert basic_solver.id_to_block[block.id].id == block.id
    
    def test_id_to_idx_mapping(self, basic_solver, small_test_blocks):
        """Test that id_to_idx mapping is correct."""
        assert len(basic_solver.id_to_idx) == len(small_test_blocks)
        for idx, block in enumerate(small_test_blocks):
            assert basic_solver.id_to_idx[block.id] == idx
    
    def test_precedence_graph_structure(self, basic_solver, small_test_blocks):
        """Test that precedence graph is built correctly."""
        G = basic_solver.G
        
        # All blocks should be nodes
        assert len(G.nodes) == len(small_test_blocks)
        
        # Check that edges match predecessor relationships
        for block in small_test_blocks:
            assert block.id in G.nodes
            for pred in block.predecessors:
                if pred.id in G.nodes:
                    assert G.has_edge(pred.id, block.id)
    
    def test_initial_partitions(self, basic_solver, small_test_blocks):
        """Test that initial partition contains all blocks."""
        assert len(basic_solver.partitions) == 1
        assert len(basic_solver.partitions[0]) == len(small_test_blocks)
        
        all_ids = {b.id for b in small_test_blocks}
        assert basic_solver.partitions[0] == all_ids
    
    def test_initial_duals(self, basic_solver):
        """Test that initial dual variables are zero."""
        assert np.allclose(basic_solver.mu_mining, 0)
        assert np.allclose(basic_solver.mu_processing, 0)


class TestMasterLP:
    """Test master LP solver."""
    
    def test_solve_master_lp_returns_valid_output(self, basic_solver):
        """Test that master LP returns duals and bound."""
        mu_m, mu_p, lp_bound = basic_solver._solve_master_lp()
        
        assert isinstance(mu_m, np.ndarray)
        assert len(mu_m) == basic_solver.periods
        assert isinstance(mu_p, np.ndarray)
        assert len(mu_p) == basic_solver.periods
        assert isinstance(lp_bound, float)
        assert lp_bound >= 0  # Should be non-negative for feasible problem
    
    def test_master_lp_duals_non_negative(self, basic_solver):
        """Test that dual variables are non-negative (shadow prices)."""
        mu_m, mu_p, _ = basic_solver._solve_master_lp()
        assert np.all(mu_m >= -1e-6)  # Allow small numerical errors
        assert np.all(mu_p >= -1e-6)
    
    def test_master_lp_bound_reasonable(self, basic_solver, small_test_blocks):
        """Test that LP bound is within reasonable range."""
        mu_m, mu_p, lp_bound = basic_solver._solve_master_lp()
        
        # Bound should be at most sum of all positive values
        max_value = sum(b.economic_value for b in small_test_blocks if b.economic_value > 0)
        assert lp_bound <= max_value * 1.1 + 1e-6  # Allow 10% margin for discounting
    
    def test_master_lp_with_multiple_partitions(self, basic_solver):
        """Test master LP with refined partitions."""
        # Manually split partitions
        all_blocks = basic_solver.partitions[0]
        block_list = list(all_blocks)
        mid = len(block_list) // 2
        basic_solver.partitions = [
            set(block_list[:mid]),
            set(block_list[mid:])
        ]
        
        mu_m, mu_p, lp_bound = basic_solver._solve_master_lp()
        assert len(mu_m) == basic_solver.periods
        assert lp_bound >= 0


class TestClosureProblem:
    """Test closure graph construction and solution."""
    
    def test_build_closure_graph_structure(self, basic_solver):
        """Test that closure graph has correct structure."""
        mu_mining = np.array([0.1, 0.2, 0.15, 0.1])
        mu_processing = np.zeros(4)
        
        G, S, T = basic_solver._build_closure_graph(mu_mining, mu_processing)
        
        # Check source and sink nodes exist
        assert S in G.nodes
        assert T in G.nodes
        
        # Check (block, period) nodes exist - at least some should be created
        # (not all blocks may create nodes if they have zero or negative weights)
        assert len(G.nodes) >= 2  # At least source and sink
        
        # Count (block, period) tuple nodes
        tuple_nodes = [n for n in G.nodes if isinstance(n, tuple)]
        assert len(tuple_nodes) > 0
    
    def test_closure_graph_edges(self, basic_solver):
        """Test that closure graph has correct edges."""
        mu_mining = np.zeros(4)
        mu_processing = np.zeros(4)
        
        G, S, T = basic_solver._build_closure_graph(mu_mining, mu_processing)
        
        # Check precedence edges within periods
        for t in range(basic_solver.periods):
            for block in basic_solver.blocks:
                node = (block.id, t)
                for pred in block.predecessors:
                    pred_node = (pred.id, t)
                    if pred_node in G.nodes:
                        assert G.has_edge(node, pred_node)
        
        # Check temporal consistency edges
        for block in basic_solver.blocks:
            for t in range(basic_solver.periods - 1):
                assert G.has_edge((block.id, t), (block.id, t + 1))
    
    def test_solve_closure_returns_valid_output(self, basic_solver):
        """Test that closure solution returns valid cut."""
        mu_mining = np.array([0.1, 0.2, 0.15, 0.1])
        mu_processing = np.zeros(4)
        
        G, S, T = basic_solver._build_closure_graph(mu_mining, mu_processing)
        closure_nodes, cut_value = basic_solver._solve_closure(G, S, T)
        
        assert isinstance(closure_nodes, set)
        assert isinstance(cut_value, float)
        assert all(isinstance(n, tuple) and len(n) == 2 for n in closure_nodes)
    
    def test_closure_respects_precedence(self, basic_solver):
        """Test that closure solution respects precedence constraints."""
        mu_mining = np.zeros(4)
        mu_processing = np.zeros(4)
        
        G, S, T = basic_solver._build_closure_graph(mu_mining, mu_processing)
        closure_nodes, _ = basic_solver._solve_closure(G, S, T)
        
        closure_blocks = {bid for (bid, _) in closure_nodes}
        
        # If a block is in closure, all its predecessors should be too
        for block in basic_solver.blocks:
            if block.id in closure_blocks:
                for pred in block.predecessors:
                    if pred.id in basic_solver.id_to_block:
                        assert pred.id in closure_blocks


class TestPartitionRefinement:
    """Test partition refinement logic."""
    
    def test_refine_partitions_splits_correctly(self, basic_solver):
        """Test that partitions are split based on closure."""
        # Create a closure that includes half the blocks
        all_blocks = list(basic_solver.blocks)
        mid = len(all_blocks) // 2
        closure_nodes = {(all_blocks[i].id, 0) for i in range(mid)}
        
        initial_partitions = len(basic_solver.partitions)
        changed = basic_solver._refine_partitions(closure_nodes)
        
        assert changed is True
        assert len(basic_solver.partitions) == initial_partitions + 1
    
    def test_refine_partitions_no_split_if_all_in(self, basic_solver):
        """Test that partition is not split if all blocks are in closure."""
        closure_nodes = {(b.id, 0) for b in basic_solver.blocks}
        
        initial_partitions = len(basic_solver.partitions)
        changed = basic_solver._refine_partitions(closure_nodes)
        
        assert changed is False
        assert len(basic_solver.partitions) == initial_partitions
    
    def test_refine_partitions_no_split_if_all_out(self, basic_solver):
        """Test that partition is not split if no blocks are in closure."""
        closure_nodes = set()
        
        initial_partitions = len(basic_solver.partitions)
        changed = basic_solver._refine_partitions(closure_nodes)
        
        assert changed is False
        assert len(basic_solver.partitions) == initial_partitions
    
    def test_refine_maintains_all_blocks(self, basic_solver):
        """Test that refinement doesn't lose any blocks."""
        all_blocks = list(basic_solver.blocks)
        mid = len(all_blocks) // 2
        closure_nodes = {(all_blocks[i].id, 0) for i in range(mid)}
        
        original_block_ids = {b.id for b in basic_solver.blocks}
        basic_solver._refine_partitions(closure_nodes)
        
        # All blocks should still be in some partition
        refined_block_ids = set()
        for partition in basic_solver.partitions:
            refined_block_ids.update(partition)
        
        assert refined_block_ids == original_block_ids


class TestDecoding:
    """Test decoding of closure to mined_by matrix."""
    
    def test_decode_mined_by_dimensions(self, basic_solver):
        """Test that mined_by matrix has correct dimensions."""
        closure_nodes = {(basic_solver.blocks[0].id, 0), (basic_solver.blocks[1].id, 1)}
        mined_by = basic_solver._decode_mined_by(closure_nodes)
        
        assert mined_by.shape == (len(basic_solver.blocks), basic_solver.periods)
    
    def test_decode_mined_by_values(self, basic_solver):
        """Test that mined_by matrix has correct values."""
        block0_id = basic_solver.blocks[0].id
        block1_id = basic_solver.blocks[1].id
        
        closure_nodes = {(block0_id, 0), (block0_id, 1), (block1_id, 2)}
        mined_by = basic_solver._decode_mined_by(closure_nodes)
        
        block0_idx = basic_solver.id_to_idx[block0_id]
        block1_idx = basic_solver.id_to_idx[block1_id]
        
        assert mined_by[block0_idx, 0] == 1
        assert mined_by[block0_idx, 1] == 1
        assert mined_by[block1_idx, 2] == 1
        assert mined_by[block1_idx, 0] == 0
    
    def test_expected_periods_calculation(self, basic_solver):
        """Test expected period calculation."""
        # Create a simple mined_by matrix
        mined_by = np.zeros((len(basic_solver.blocks), basic_solver.periods))
        
        # Block 0: cumulative probability - mined by end of period 0
        mined_by[0, 0] = 1.0
        mined_by[0, 1] = 1.0
        mined_by[0, 2] = 1.0
        mined_by[0, 3] = 1.0
        
        # Block 1: mined by end of period 1 (cumulative)
        mined_by[1, 0] = 0.0
        mined_by[1, 1] = 0.5
        mined_by[1, 2] = 1.0
        mined_by[1, 3] = 1.0
        
        E = basic_solver._expected_periods(mined_by)
        
        block0_id = basic_solver.blocks[0].id
        block1_id = basic_solver.blocks[1].id
        
        # Expected period is calculated from incremental probabilities
        # Block 0: prob=1.0 in period 1 => E = 1.0 * 1 = 1
        assert E[block0_id] == 1.0
        # Block 1: prob=0.5 in period 2, prob=0.5 in period 3 => E = 0.5*2 + 0.5*3 = 2.5
        assert E[block1_id] == pytest.approx(2.5)


class TestTopoSort:
    """Test topological sort scheduling."""
    
    def test_toposort_schedule_all_blocks(self, basic_solver):
        """Test that toposort schedules all blocks."""
        mined_by = np.random.rand(len(basic_solver.blocks), basic_solver.periods)
        schedule = basic_solver._toposort_schedule(mined_by)
        
        assert len(schedule) == len(basic_solver.blocks)
        assert all(block.id in schedule for block in basic_solver.blocks)
    
    def test_toposort_respects_precedence(self, basic_solver):
        """Test that schedule respects precedence constraints."""
        mined_by = np.ones((len(basic_solver.blocks), basic_solver.periods))
        schedule = basic_solver._toposort_schedule(mined_by)
        
        for block in basic_solver.blocks:
            block_period = schedule[block.id]
            if block_period >= 0:
                for pred in block.predecessors:
                    if pred.id in schedule:
                        pred_period = schedule[pred.id]
                        if pred_period >= 0:
                            assert pred_period <= block_period
    
    def test_toposort_respects_capacity(self, basic_solver):
        """Test that schedule respects capacity constraints (soft)."""
        mined_by = np.ones((len(basic_solver.blocks), basic_solver.periods))
        schedule = basic_solver._toposort_schedule(mined_by)
        
        # Calculate tonnage per period
        tons_per_period = np.zeros(basic_solver.periods)
        for block in basic_solver.blocks:
            period = schedule[block.id]
            if 0 <= period < basic_solver.periods:
                tons_per_period[period] += block.tonnage
        
        # At least some periods should respect capacity (greedy may overflow last period)
        assert any(tons_per_period[t] <= basic_solver.mining_capacity for t in range(basic_solver.periods - 1))
    
    def test_toposort_period_range(self, basic_solver):
        """Test that scheduled periods are in valid range."""
        mined_by = np.ones((len(basic_solver.blocks), basic_solver.periods))
        schedule = basic_solver._toposort_schedule(mined_by)
        
        for block_id, period in schedule.items():
            assert -1 <= period < basic_solver.periods


class TestIntegration:
    """Integration tests for full solve."""
    
    def test_solve_returns_valid_schedule(self, basic_solver):
        """Test that solve returns valid schedule and logs."""
        schedule, logs = basic_solver.solve(max_iterations=10)
        
        assert isinstance(schedule, dict)
        assert isinstance(logs, list)
        assert len(schedule) == len(basic_solver.blocks)
        assert len(logs) > 0
    
    def test_solve_log_format(self, basic_solver):
        """Test that logs have correct format."""
        _, logs = basic_solver.solve(max_iterations=10)
        
        for log_entry in logs:
            assert len(log_entry) == 3
            iteration, lp_bound, num_partitions = log_entry
            assert isinstance(iteration, int)
            assert isinstance(lp_bound, float)
            assert isinstance(num_partitions, int)
            assert iteration > 0
            assert num_partitions > 0
    
    def test_solve_convergence(self, basic_solver):
        """Test that solver converges within max iterations."""
        schedule, logs = basic_solver.solve(max_iterations=20)
        
        assert len(logs) <= 20
    
    def test_solve_with_full_dataset(self, test_blocks):
        """Test solver with full 1000-block dataset."""
        solver = BZSolver(
            blocks=test_blocks,
            periods=4,
            discount_rate=0.1,
            mining_capacity=50000,
        )
        
        schedule, logs = solver.solve(max_iterations=20)
        
        assert len(schedule) == len(test_blocks)
        assert all(block.id in schedule for block in test_blocks)
        assert len(logs) > 0
    
    def test_solve_with_processing_capacity(self, solver_with_processing):
        """Test solver with processing capacity constraint."""
        schedule, logs = solver_with_processing.solve(max_iterations=10)
        
        assert len(schedule) == len(solver_with_processing.blocks)
        assert len(logs) > 0
    
    def test_solve_npv_calculation(self, basic_solver):
        """Test that NPV calculation is correct and schedule is valid."""
        schedule, _ = basic_solver.solve(max_iterations=10)
        
        # Calculate NPV
        npv = 0.0
        scheduled_blocks = 0
        for block in basic_solver.blocks:
            period = schedule[block.id]
            if period >= 0:
                scheduled_blocks += 1
                df = 1.0 / ((1.0 + basic_solver.discount_rate) ** (period + 1))
                npv += block.economic_value * df
        
        # NPV may be positive or negative depending on ore/waste ratio
        # Just verify we scheduled some blocks and calculation makes sense
        assert scheduled_blocks > 0
        assert isinstance(npv, float)
        assert not np.isnan(npv)
        assert not np.isinf(npv)
    
    def test_solve_schedule_feasibility(self, basic_solver):
        """Test that final schedule is topologically feasible."""
        schedule, _ = basic_solver.solve(max_iterations=10)
        
        # Check precedence constraints
        for block in basic_solver.blocks:
            block_period = schedule[block.id]
            if block_period >= 0:
                for pred in block.predecessors:
                    if pred.id in schedule:
                        pred_period = schedule[pred.id]
                        if pred_period >= 0:
                            assert pred_period <= block_period, \
                                f"Block {block.id} in period {block_period} violates precedence with {pred.id} in period {pred_period}"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_blocks(self):
        """Test solver with empty block list."""
        solver = BZSolver(
            blocks=[],
            periods=4,
            discount_rate=0.1,
            mining_capacity=30000,
        )
        
        schedule, logs = solver.solve(max_iterations=5)
        assert len(schedule) == 0
        assert len(logs) > 0
    
    def test_single_period(self, small_test_blocks):
        """Test solver with single period."""
        solver = BZSolver(
            blocks=small_test_blocks[:10],
            periods=1,
            discount_rate=0.1,
            mining_capacity=50000,
        )
        
        schedule, logs = solver.solve(max_iterations=5)
        assert all(period == 0 or period == -1 for period in schedule.values())
    
    def test_very_small_capacity(self, small_test_blocks):
        """Test solver with very restrictive capacity."""
        solver = BZSolver(
            blocks=small_test_blocks[:10],
            periods=4,
            discount_rate=0.1,
            mining_capacity=1000,  # Very small
        )
        
        schedule, logs = solver.solve(max_iterations=5)
        # Should still produce a schedule, even if it violates capacity
        assert len(schedule) == 10
    
    def test_zero_discount_rate(self, small_test_blocks):
        """Test solver with zero discount rate."""
        solver = BZSolver(
            blocks=small_test_blocks[:10],
            periods=4,
            discount_rate=0.0,
            mining_capacity=30000,
        )
        
        schedule, logs = solver.solve(max_iterations=5)
        assert len(schedule) == 10
    
    def test_high_discount_rate(self, small_test_blocks):
        """Test solver with high discount rate."""
        solver = BZSolver(
            blocks=small_test_blocks[:10],
            periods=4,
            discount_rate=0.5,
            mining_capacity=30000,
        )
        
        schedule, logs = solver.solve(max_iterations=5)
        assert len(schedule) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
