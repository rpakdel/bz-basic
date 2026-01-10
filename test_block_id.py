"""
Unit tests for block_id calculation functions.

Tests the conversion between 3D coordinates (x, y, z) and block IDs
to ensure consistent, reversible mapping for the block model.
"""

import pytest

from deposit_utils import calculate_block_id, calculate_coordinates_from_block_id


class TestBlockIDCalculation:
    """Test block_id calculation from coordinates."""

    def test_origin_block(self):
        """Test that (0, 0, 0) maps to block_id 0."""
        block_id = calculate_block_id(0, 0, 0, x_size=25, y_size=25)
        assert block_id == 0

    def test_single_x_increment(self):
        """Test that incrementing X by 1 increments block_id by 1."""
        x_size, y_size = 25, 25
        id_at_0 = calculate_block_id(0, 0, 0, x_size, y_size)
        id_at_1 = calculate_block_id(1, 0, 0, x_size, y_size)
        assert id_at_1 - id_at_0 == 1

    def test_y_increment_jumps_by_x_size(self):
        """Test that incrementing Y by 1 increments block_id by x_size."""
        x_size, y_size = 25, 25
        id_at_0 = calculate_block_id(0, 0, 0, x_size, y_size)
        id_at_1 = calculate_block_id(0, 1, 0, x_size, y_size)
        assert id_at_1 - id_at_0 == x_size

    def test_z_increment_jumps_by_xy_plane(self):
        """Test that incrementing Z by 1 increments block_id by x_size * y_size."""
        x_size, y_size = 25, 25
        id_at_0 = calculate_block_id(0, 0, 0, x_size, y_size)
        id_at_1 = calculate_block_id(0, 0, 1, x_size, y_size)
        assert id_at_1 - id_at_0 == x_size * y_size

    def test_max_x_coordinate(self):
        """Test the last block in X direction."""
        x_size, y_size = 25, 25
        block_id = calculate_block_id(24, 0, 0, x_size, y_size)
        assert block_id == 24

    def test_max_y_coordinate(self):
        """Test the last block in Y direction."""
        x_size, y_size = 25, 25
        block_id = calculate_block_id(0, 24, 0, x_size, y_size)
        assert block_id == 24 * x_size

    def test_max_z_coordinate(self):
        """Test the last block in Z direction (top surface)."""
        x_size, y_size, z_size = 25, 25, 25
        block_id = calculate_block_id(0, 0, 24, x_size, y_size)
        assert block_id == 24 * x_size * y_size

    def test_corner_block_top_right(self):
        """Test the top-right-back corner (max x, max y, max z)."""
        x_size, y_size = 25, 25
        block_id = calculate_block_id(24, 24, 24, x_size, y_size)
        # (24 + 24*25 + 24*25*25) = 24 + 600 + 15000 = 15624
        assert block_id == 15624

    def test_various_coordinates_25x25x25(self):
        """Test various coordinates in a 25x25x25 block model."""
        x_size, y_size = 25, 25
        
        test_cases = [
            ((0, 0, 0), 0),
            ((1, 0, 0), 1),
            ((5, 0, 0), 5),
            ((0, 1, 0), 25),
            ((0, 5, 0), 125),
            ((0, 0, 1), 625),
            ((0, 0, 5), 3125),
            ((3, 4, 2), 3 + 4*25 + 2*625),  # 3 + 100 + 1250 = 1353
        ]
        
        for (x, y, z), expected_id in test_cases:
            block_id = calculate_block_id(x, y, z, x_size, y_size)
            assert block_id == expected_id, f"({x}, {y}, {z}) should map to {expected_id}, got {block_id}"

    def test_different_model_sizes(self):
        """Test with different block model dimensions."""
        test_cases = [
            (10, 10, 100),  # 10x10 model
            (20, 20, 400),  # 20x20 model
            (30, 30, 900),  # 30x30 model
            (5, 8, 40),     # Non-square model
        ]
        
        for x_size, y_size, expected_plane_size in test_cases:
            # Test that Z increment matches xy_size
            id_z0 = calculate_block_id(0, 0, 0, x_size, y_size)
            id_z1 = calculate_block_id(0, 0, 1, x_size, y_size)
            assert id_z1 - id_z0 == expected_plane_size


class TestBlockIDInversion:
    """Test reverse conversion from block_id to coordinates."""

    def test_invert_origin(self):
        """Test that block_id 0 maps back to (0, 0, 0)."""
        x, y, z = calculate_coordinates_from_block_id(0, x_size=25, y_size=25)
        assert (x, y, z) == (0, 0, 0)

    def test_invert_simple_x(self):
        """Test inversion for simple X offsets."""
        x_size, y_size = 25, 25
        for test_x in [0, 1, 5, 10, 24]:
            x, y, z = calculate_coordinates_from_block_id(test_x, x_size, y_size)
            assert (x, y, z) == (test_x, 0, 0)

    def test_invert_y_boundary(self):
        """Test inversion for Y boundaries."""
        x_size, y_size = 25, 25
        block_id = 25  # (0, 1, 0)
        x, y, z = calculate_coordinates_from_block_id(block_id, x_size, y_size)
        assert (x, y, z) == (0, 1, 0)

    def test_invert_z_boundary(self):
        """Test inversion for Z boundaries."""
        x_size, y_size = 25, 25
        block_id = 625  # (0, 0, 1)
        x, y, z = calculate_coordinates_from_block_id(block_id, x_size, y_size)
        assert (x, y, z) == (0, 0, 1)

    def test_roundtrip_forward_then_backward(self):
        """Test that forward + backward conversion gives original coordinates."""
        x_size, y_size = 25, 25
        
        test_coords = [
            (0, 0, 0),
            (1, 0, 0),
            (24, 0, 0),
            (0, 1, 0),
            (5, 5, 5),
            (24, 24, 24),
            (12, 13, 7),
        ]
        
        for orig_x, orig_y, orig_z in test_coords:
            block_id = calculate_block_id(orig_x, orig_y, orig_z, x_size, y_size)
            x, y, z = calculate_coordinates_from_block_id(block_id, x_size, y_size)
            assert (x, y, z) == (orig_x, orig_y, orig_z), \
                f"Roundtrip failed for ({orig_x}, {orig_y}, {orig_z}): got ({x}, {y}, {z})"

    def test_roundtrip_backward_then_forward(self):
        """Test that backward + forward conversion gives original block_id."""
        x_size, y_size = 25, 25
        
        test_block_ids = [0, 1, 24, 25, 625, 1000, 15624]
        
        for orig_id in test_block_ids:
            x, y, z = calculate_coordinates_from_block_id(orig_id, x_size, y_size)
            block_id = calculate_block_id(x, y, z, x_size, y_size)
            assert block_id == orig_id, \
                f"Roundtrip failed for block_id {orig_id}: got {block_id}"

    def test_invert_various_coordinates_25x25x25(self):
        """Test inversion of various block IDs."""
        x_size, y_size = 25, 25
        
        test_cases = [
            (0, (0, 0, 0)),
            (1, (1, 0, 0)),
            (5, (5, 0, 0)),
            (25, (0, 1, 0)),
            (125, (0, 5, 0)),
            (625, (0, 0, 1)),
            (3125, (0, 0, 5)),
            (1353, (3, 4, 2)),  # 3 + 100 + 1250 = 1353
        ]
        
        for block_id, expected_coords in test_cases:
            x, y, z = calculate_coordinates_from_block_id(block_id, x_size, y_size)
            assert (x, y, z) == expected_coords, \
                f"block_id {block_id} should map to {expected_coords}, got ({x}, {y}, {z})"

    def test_invert_different_model_sizes(self):
        """Test inversion with different block model dimensions."""
        test_cases = [
            (10, 10),
            (20, 20),
            (5, 8),
        ]
        
        for x_size, y_size in test_cases:
            # Test various positions
            for x in range(0, min(x_size, 5)):
                for y in range(0, min(y_size, 5)):
                    for z in range(0, 3):
                        block_id = calculate_block_id(x, y, z, x_size, y_size)
                        x_inv, y_inv, z_inv = calculate_coordinates_from_block_id(block_id, x_size, y_size)
                        assert (x_inv, y_inv, z_inv) == (x, y, z), \
                            f"({x}, {y}, {z}) roundtrip failed in {x_size}x{y_size} model"


class TestBlockIDSequencing:
    """Test that block IDs are generated in expected sequential order."""

    def test_sequential_ordering_x_fastest(self):
        """Test that X varies fastest (changes every block)."""
        x_size, y_size = 25, 25
        ids = [calculate_block_id(x, 0, 0, x_size, y_size) for x in range(x_size)]
        # Should be [0, 1, 2, ..., 24]
        assert ids == list(range(x_size))

    def test_sequential_ordering_y_next(self):
        """Test that Y varies next (changes every x_size blocks)."""
        x_size, y_size = 25, 25
        ids = [calculate_block_id(0, y, 0, x_size, y_size) for y in range(y_size)]
        # Should be [0, 25, 50, 75, ...]
        assert ids == [y * x_size for y in range(y_size)]

    def test_sequential_ordering_z_slowest(self):
        """Test that Z varies slowest (changes every x_size*y_size blocks)."""
        x_size, y_size = 25, 25
        ids = [calculate_block_id(0, 0, z, x_size, y_size) for z in range(5)]
        # Should be [0, 625, 1250, 1875, 2500]
        expected = [z * x_size * y_size for z in range(5)]
        assert ids == expected

    def test_total_blocks_count(self):
        """Test that a full model has expected number of unique block IDs."""
        x_size, y_size, z_size = 25, 25, 25
        block_ids = set()
        
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    block_id = calculate_block_id(x, y, z, x_size, y_size)
                    block_ids.add(block_id)
        
        # Should have exactly x_size * y_size * z_size unique block IDs
        assert len(block_ids) == x_size * y_size * z_size
        # IDs should range from 0 to (total - 1)
        assert min(block_ids) == 0
        assert max(block_ids) == (x_size * y_size * z_size) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
