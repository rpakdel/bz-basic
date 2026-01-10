"""
Unit tests for slope-based precedence calculation (1:3 slope).

Tests the automatic generation of block dependencies based on
standard mining slope constraints.
"""

import pytest

from deposit_utils import (
    calculate_block_id,
    calculate_coordinates_from_block_id,
    calculate_precedence_graph_1_3,
    get_slope_predecessors_1_3,
)


class TestSlopePredecessors:
    """Test get_slope_predecessors_1_3 function."""

    def test_surface_block_no_predecessors(self):
        """Surface blocks (z_max) have no predecessors."""
        x_size, y_size, z_size = 25, 25, 25
        # Block at z = 24 (top)
        preds = get_slope_predecessors_1_3(5, 5, 24, x_size, y_size, z_size)
        assert preds == []

    def test_interior_block_three_predecessors(self):
        """Interior block has 3 predecessors in standard case."""
        x_size, y_size, z_size = 25, 25, 25
        # Block at (5, 5, 10) - interior block
        preds = get_slope_predecessors_1_3(5, 5, 10, x_size, y_size, z_size)
        assert len(preds) == 3
        # Should be centered at (5, 5, 11)
        assert (4, 5, 11) in preds
        assert (5, 5, 11) in preds
        assert (6, 5, 11) in preds

    def test_x_boundary_left_block(self):
        """Block at x=0 (left edge) has 2 predecessors (missing left neighbor)."""
        x_size, y_size, z_size = 25, 25, 25
        # Block at (0, 5, 10)
        preds = get_slope_predecessors_1_3(0, 5, 10, x_size, y_size, z_size)
        assert len(preds) == 2
        assert (0, 5, 11) in preds
        assert (1, 5, 11) in preds
        # (-1, 5, 11) should not be included (out of bounds)
        assert (-1, 5, 11) not in preds

    def test_x_boundary_right_block(self):
        """Block at x=x_max (right edge) has 2 predecessors."""
        x_size, y_size, z_size = 25, 25, 25
        # Block at (24, 5, 10)
        preds = get_slope_predecessors_1_3(24, 5, 10, x_size, y_size, z_size)
        assert len(preds) == 2
        assert (23, 5, 11) in preds
        assert (24, 5, 11) in preds

    def test_origin_block_bottom(self):
        """Block at origin (0, 0, 0) has 2 predecessors above."""
        x_size, y_size, z_size = 25, 25, 25
        # Block at (0, 0, 0) - corner at bottom
        preds = get_slope_predecessors_1_3(0, 0, 0, x_size, y_size, z_size)
        assert len(preds) == 2
        assert (0, 0, 1) in preds
        assert (1, 0, 1) in preds

    def test_just_below_surface(self):
        """Block at z=z_max-1 (just below surface) has 3 predecessors."""
        x_size, y_size, z_size = 25, 25, 25
        # Block at (12, 12, 23)
        preds = get_slope_predecessors_1_3(12, 12, 23, x_size, y_size, z_size)
        assert len(preds) == 3
        assert (11, 12, 24) in preds
        assert (12, 12, 24) in preds
        assert (13, 12, 24) in preds

    def test_y_boundaries_not_affecting_predecessors(self):
        """Y-boundaries don't restrict predecessors (slope is in X-Z plane)."""
        x_size, y_size, z_size = 25, 25, 25
        # Blocks at y=0 and y=24 should both have 3 interior predecessors
        preds_y0 = get_slope_predecessors_1_3(5, 0, 10, x_size, y_size, z_size)
        preds_y_max = get_slope_predecessors_1_3(5, 24, 10, x_size, y_size, z_size)
        
        assert len(preds_y0) == 3
        assert len(preds_y_max) == 3
        
        # Same y-coordinate in predecessors
        assert all(py == 0 for _, py, _ in preds_y0)
        assert all(py == 24 for _, py, _ in preds_y_max)

    def test_different_model_sizes(self):
        """Test precedence calculation for different model sizes."""
        test_cases = [
            (10, 10, 10),
            (20, 20, 20),
            (5, 8, 12),
        ]
        
        for x_size, y_size, z_size in test_cases:
            # Interior block should have 3 predecessors
            preds = get_slope_predecessors_1_3(
                x_size // 2, y_size // 2, z_size // 2,
                x_size, y_size, z_size
            )
            assert len(preds) == 3


class TestPrecedenceGraph:
    """Test calculate_precedence_graph_1_3 function."""

    def test_graph_completeness(self):
        """Graph contains entry for every block."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Should have exactly x_size * y_size * z_size entries
        assert len(graph) == x_size * y_size * z_size
        
        # All block IDs from 0 to total-1 should be present
        total_blocks = x_size * y_size * z_size
        for block_id in range(total_blocks):
            assert block_id in graph

    def test_surface_blocks_have_no_predecessors(self):
        """Surface blocks (z = z_max) have empty predecessor lists."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Surface level is z = z_max - 1 = 9
        for x in range(x_size):
            for y in range(y_size):
                block_id = calculate_block_id(x, y, z_size - 1, x_size, y_size)
                assert graph[block_id] == [], \
                    f"Surface block ({x}, {y}, {z_size-1}) should have no predecessors"

    def test_interior_blocks_have_predecessors(self):
        """Non-surface blocks have predecessors."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Block at (5, 5, 5) is interior
        block_id = calculate_block_id(5, 5, 5, x_size, y_size)
        preds = graph[block_id]
        
        # Should have 3 predecessors
        assert len(preds) == 3
        
        # Verify they correspond to the expected coordinates
        expected_pred_coords = [(4, 5, 6), (5, 5, 6), (6, 5, 6)]
        expected_pred_ids = [
            calculate_block_id(px, py, pz, x_size, y_size)
            for px, py, pz in expected_pred_coords
        ]
        assert set(preds) == set(expected_pred_ids)

    def test_boundary_blocks_have_fewer_predecessors(self):
        """Edge blocks have fewer predecessors (2 instead of 3)."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Block at (0, 5, 5) is at left edge
        block_id = calculate_block_id(0, 5, 5, x_size, y_size)
        preds = graph[block_id]
        assert len(preds) == 2

    def test_graph_25x25x25(self):
        """Test graph structure for actual model size."""
        x_size, y_size, z_size = 25, 25, 25
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Total blocks
        total = x_size * y_size * z_size
        assert len(graph) == total
        
        # Surface blocks (z=24) have no predecessors
        surface_count = 0
        for x in range(x_size):
            for y in range(y_size):
                block_id = calculate_block_id(x, y, 24, x_size, y_size)
                if graph[block_id] == []:
                    surface_count += 1
        assert surface_count == x_size * y_size

    def test_predecessor_ids_valid(self):
        """All predecessor IDs in graph are valid block IDs."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        total_blocks = x_size * y_size * z_size
        
        for block_id, pred_ids in graph.items():
            # Each predecessor ID should be valid
            for pred_id in pred_ids:
                assert 0 <= pred_id < total_blocks, \
                    f"Invalid predecessor ID {pred_id} for block {block_id}"

    def test_acyclic_graph(self):
        """Precedence graph has no cycles (DAG structure)."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Simple check: all predecessors should have higher z-levels
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    block_id = calculate_block_id(x, y, z, x_size, y_size)
                    pred_ids = graph[block_id]
                    
                    for pred_id in pred_ids:
                        pred_x, pred_y, pred_z = calculate_coordinates_from_block_id(
                            pred_id, x_size, y_size
                        )
                        # All predecessors must be at higher z level
                        assert pred_z > z, \
                            f"Block ({x}, {y}, {z}) has predecessor at lower z: " \
                            f"({pred_x}, {pred_y}, {pred_z})"

    def test_consistency_with_get_slope_predecessors(self):
        """Graph results match individual get_slope_predecessors calls."""
        x_size, y_size, z_size = 15, 15, 15
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Test a sample of blocks
        for x in [0, 5, 7, 14]:
            for y in [0, 5, 14]:
                for z in [0, 7, 14]:
                    block_id = calculate_block_id(x, y, z, x_size, y_size)
                    graph_preds = graph[block_id]
                    
                    # Get expected predecessors
                    pred_coords = get_slope_predecessors_1_3(x, y, z, x_size, y_size, z_size)
                    expected_ids = [
                        calculate_block_id(px, py, pz, x_size, y_size)
                        for px, py, pz in pred_coords
                    ]
                    
                    assert set(graph_preds) == set(expected_ids), \
                        f"Mismatch for block ({x}, {y}, {z})"


class TestPrecedenceLogic:
    """Test the mining logic implied by precedence constraints."""

    def test_mining_sequence_respects_slope(self):
        """A valid mining sequence respects all slope constraints."""
        x_size, y_size, z_size = 5, 5, 5
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Valid sequence: mine TOP-DOWN (surface first, then deeper levels)
        # Predecessors are ABOVE, so we must mine from high z to low z
        mining_order = []
        for z in range(z_size - 1, -1, -1):  # Start from z_max, go down to 0
            for y in range(y_size):
                for x in range(x_size):
                    mining_order.append(calculate_block_id(x, y, z, x_size, y_size))
        
        # Check that all precedence constraints are satisfied
        mined = set()
        for block_id in mining_order:
            pred_ids = graph[block_id]
            # All predecessors must already be mined
            assert all(pred_id in mined for pred_id in pred_ids), \
                f"Block {block_id} mined before its predecessors"
            mined.add(block_id)

    def test_top_block_can_be_mined_first(self):
        """Top surface block can be mined first (no dependencies)."""
        x_size, y_size, z_size = 10, 10, 10
        graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
        
        # Any surface block should have no predecessors
        for x in range(x_size):
            for y in range(y_size):
                block_id = calculate_block_id(x, y, z_size - 1, x_size, y_size)
                assert len(graph[block_id]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
