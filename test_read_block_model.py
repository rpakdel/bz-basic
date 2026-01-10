"""
Unit tests for read_block_model_csv function.

Tests automatic block_id assignment and precedence calculation
when parsing CSV files with spatial coordinates.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from deposit_utils import calculate_block_id, read_block_model_csv


class TestReadBlockModelCSV:
    """Test read_block_model_csv function."""

    def create_test_csv(self, x_size, y_size, z_size):
        """Helper: create a temporary CSV file for testing."""
        data = []
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    data.append({
                        "x": x,
                        "y": y,
                        "z": z,
                        "tonnage": 1000,
                        "grade": 0.5,
                        "economic_value": 10000
                    })
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        df.to_csv(temp_file.name, index=False)
        return temp_file.name

    def test_small_model_dimensions(self):
        """Test that dimensions are correctly determined from CSV."""
        csv_path = self.create_test_csv(5, 5, 5)
        try:
            blocks, x_size, y_size, z_size = read_block_model_csv(csv_path)
            
            assert x_size == 5
            assert y_size == 5
            assert z_size == 5
            assert len(blocks) == 125  # 5*5*5
        finally:
            Path(csv_path).unlink()

    def test_block_count_matches_dimensions(self):
        """Test that number of blocks matches model dimensions."""
        test_cases = [
            (3, 3, 3, 27),
            (10, 10, 5, 500),
            (5, 8, 4, 160),
        ]
        
        for x_size, y_size, z_size, expected_count in test_cases:
            csv_path = self.create_test_csv(x_size, y_size, z_size)
            try:
                blocks, _, _, _ = read_block_model_csv(csv_path)
                assert len(blocks) == expected_count
            finally:
                Path(csv_path).unlink()

    def test_block_ids_assigned_correctly(self):
        """Test that block_ids are calculated correctly from coordinates."""
        csv_path = self.create_test_csv(5, 5, 5)
        try:
            blocks, x_size, y_size, z_size = read_block_model_csv(csv_path)
            
            # Check a few specific blocks
            for block in blocks:
                expected_id = calculate_block_id(
                    block.x, block.y, block.z, x_size, y_size
                )
                assert block.id == expected_id, \
                    f"Block at ({block.x}, {block.y}, {block.z}) has wrong ID"
        finally:
            Path(csv_path).unlink()

    def test_coordinates_preserved(self):
        """Test that x, y, z coordinates are preserved in Block objects."""
        csv_path = self.create_test_csv(5, 5, 5)
        try:
            blocks, _, _, _ = read_block_model_csv(csv_path)
            
            # Create a set of all coordinates
            coords = {(b.x, b.y, b.z) for b in blocks}
            
            # Should have all combinations from 0-4 in each dimension
            expected_coords = {
                (x, y, z)
                for x in range(5)
                for y in range(5)
                for z in range(5)
            }
            
            assert coords == expected_coords
        finally:
            Path(csv_path).unlink()

    def test_attributes_preserved(self):
        """Test that tonnage, grade, and value are preserved."""
        csv_path = self.create_test_csv(3, 3, 3)
        try:
            blocks, _, _, _ = read_block_model_csv(csv_path)
            
            for block in blocks:
                assert block.tonnage == 1000
                assert block.grade == 0.5
                assert block.economic_value == 10000
        finally:
            Path(csv_path).unlink()

    def test_surface_blocks_no_predecessors(self):
        """Test that surface blocks (z_max) have no predecessors."""
        csv_path = self.create_test_csv(5, 5, 5)
        try:
            blocks, _, _, z_size = read_block_model_csv(csv_path)
            
            # Filter surface blocks
            surface_blocks = [b for b in blocks if b.z == z_size - 1]
            
            # All surface blocks should have no predecessors
            for block in surface_blocks:
                assert len(block.predecessors) == 0, \
                    f"Surface block at ({block.x}, {block.y}, {block.z}) has predecessors"
        finally:
            Path(csv_path).unlink()

    def test_interior_blocks_have_predecessors(self):
        """Test that interior blocks have correct number of predecessors."""
        csv_path = self.create_test_csv(10, 10, 10)
        try:
            blocks, x_size, _, z_size = read_block_model_csv(csv_path)
            
            # Find an interior block (not at any boundary)
            interior_block = None
            for block in blocks:
                if (0 < block.x < x_size - 1 and 
                    0 < block.z < z_size - 1):
                    interior_block = block
                    break
            
            assert interior_block is not None
            # Interior block should have 3 predecessors
            assert len(interior_block.predecessors) == 3
        finally:
            Path(csv_path).unlink()

    def test_edge_blocks_fewer_predecessors(self):
        """Test that edge blocks have 2 predecessors."""
        csv_path = self.create_test_csv(10, 10, 10)
        try:
            blocks, _, _, z_size = read_block_model_csv(csv_path)
            
            # Block at x=0 (left edge), not at surface
            edge_block = [b for b in blocks if b.x == 0 and b.z < z_size - 1][0]
            
            # Edge block should have 2 predecessors
            assert len(edge_block.predecessors) == 2
        finally:
            Path(csv_path).unlink()

    def test_predecessors_are_above(self):
        """Test that all predecessors are at higher z level."""
        csv_path = self.create_test_csv(5, 5, 5)
        try:
            blocks, _, _, _ = read_block_model_csv(csv_path)
            
            for block in blocks:
                for pred in block.predecessors:
                    assert pred.z > block.z, \
                        f"Block ({block.x}, {block.y}, {block.z}) has " \
                        f"predecessor at same/lower z: ({pred.x}, {pred.y}, {pred.z})"
        finally:
            Path(csv_path).unlink()

    def test_predecessor_coordinates_match_slope(self):
        """Test that predecessor coordinates match 1:3 slope pattern."""
        csv_path = self.create_test_csv(10, 10, 10)
        try:
            blocks, x_size, _, z_size = read_block_model_csv(csv_path)
            
            # Test an interior block
            test_block = [b for b in blocks 
                         if b.x == 5 and b.y == 5 and b.z == 5][0]
            
            assert len(test_block.predecessors) == 3
            
            # Predecessors should be at (4,5,6), (5,5,6), (6,5,6)
            pred_coords = {(p.x, p.y, p.z) for p in test_block.predecessors}
            expected = {(4, 5, 6), (5, 5, 6), (6, 5, 6)}
            
            assert pred_coords == expected
        finally:
            Path(csv_path).unlink()

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        # Create CSV with missing 'tonnage' column
        data = [{"x": 0, "y": 0, "z": 0, "economic_value": 1000}]
        df = pd.DataFrame(data)
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        df.to_csv(temp_file.name, index=False)
        
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                read_block_model_csv(temp_file.name)
        finally:
            Path(temp_file.name).unlink()

    def test_grade_optional_defaults_to_zero(self):
        """Test that missing 'grade' column defaults to 0."""
        # Create CSV without grade column
        data = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    data.append({
                        "x": x, "y": y, "z": z,
                        "tonnage": 1000,
                        "economic_value": 5000
                    })
        
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        df.to_csv(temp_file.name, index=False)
        
        try:
            blocks, _, _, _ = read_block_model_csv(temp_file.name)
            
            # All blocks should have grade = 0
            for block in blocks:
                assert block.grade == 0.0
        finally:
            Path(temp_file.name).unlink()

    def test_actual_data_file(self):
        """Test with the actual generated block model CSV."""
        csv_path = "data/block_model_25_25_25_2026_01_10_03_26.csv"
        
        # Check if file exists
        if not Path(csv_path).exists():
            pytest.skip(f"Test data file not found: {csv_path}")
        
        blocks, x_size, y_size, z_size = read_block_model_csv(csv_path)
        
        # Verify dimensions
        assert x_size == 25
        assert y_size == 25
        assert z_size == 25
        
        # Verify block count
        assert len(blocks) == 15625  # 25*25*25
        
        # Verify block IDs are unique
        block_ids = [b.id for b in blocks]
        assert len(set(block_ids)) == len(block_ids)
        
        # Verify IDs are in valid range
        assert min(block_ids) == 0
        assert max(block_ids) == 15624
        
        # Check surface blocks
        surface_blocks = [b for b in blocks if b.z == 24]
        assert len(surface_blocks) == 625  # 25*25
        assert all(len(b.predecessors) == 0 for b in surface_blocks)
        
        # Check a bottom block has predecessors
        bottom_blocks = [b for b in blocks if b.z == 0]
        assert all(len(b.predecessors) > 0 for b in bottom_blocks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
