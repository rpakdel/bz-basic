import io
from typing import List, Tuple

import numpy as np
import pandas as pd

from block import Block


def calculate_block_id(x: int, y: int, z: int, x_size: int, y_size: int) -> int:
    """
    Calculate a unique block ID from 3D coordinates.
    
    The block ID is computed using a row-major ordering:
      block_id = x + (y * x_size) + (z * x_size * y_size)
    
    This means:
      - X varies fastest (innermost loop)
      - Y varies next
      - Z varies slowest (outermost loop)
    
    Args:
        x: X coordinate (easting), 0 <= x < x_size
        y: Y coordinate (northing), 0 <= y < y_size
        z: Z coordinate (elevation), 0 <= z < z_size
        x_size: Total blocks in X direction
        y_size: Total blocks in Y direction
    
    Returns:
        Unique block_id (non-negative integer)
    
    Example:
        For a 25x25x25 model:
        - (0, 0, 0) -> ID 0 (bottom corner)
        - (1, 0, 0) -> ID 1 (next in X)
        - (0, 1, 0) -> ID 25 (next in Y)
        - (0, 0, 1) -> ID 625 (next in Z)
    """
    return x + (y * x_size) + (z * x_size * y_size)


def calculate_coordinates_from_block_id(block_id: int, x_size: int, y_size: int) -> Tuple[int, int, int]:
    """
    Inverse of calculate_block_id: recover coordinates from block_id.
    
    Args:
        block_id: The block identifier
        x_size: Total blocks in X direction
        y_size: Total blocks in Y direction
    
    Returns:
        Tuple of (x, y, z) coordinates
    
    Example:
        For a 25x25x25 model:
        - ID 0 -> (0, 0, 0)
        - ID 1 -> (1, 0, 0)
        - ID 625 -> (0, 0, 1)
    """
    xy_size = x_size * y_size
    z = block_id // xy_size
    remainder = block_id % xy_size
    y = remainder // x_size
    x = remainder % x_size
    return x, y, z


def get_slope_predecessors_1_3(
    x: int, y: int, z: int, x_size: int, y_size: int, z_size: int
) -> List[Tuple[int, int, int]]:
    """
    Calculate predecessor blocks for a 1:3 slope constraint.
    
    Standard 1:3 slope rule: to mine block (x, y, z), you must first mine
    the 3 blocks directly above it at level z+1:
      - (x-1, y, z+1)  [left]
      - (x,   y, z+1)  [center]
      - (x+1, y, z+1)  [right]
    
    Assumes Z is elevation (positive upward, surface is z_max).
    
    Args:
        x, y, z: Current block coordinates
        x_size: Total blocks in X direction
        y_size: Total blocks in Y direction
        z_size: Total blocks in Z direction
    
    Returns:
        List of (x, y, z) coordinates of predecessor blocks that are valid
        (within bounds).
    
    Example:
        For block (5, 5, 10) in a 25x25x25 model:
        Returns: [(4, 5, 11), (5, 5, 11), (6, 5, 11)]
    """
    predecessors = []
    
    # Blocks above this one (z+1)
    if z < z_size - 1:
        # Three blocks in X direction at the level above
        for dx in [-1, 0, 1]:
            pred_x = x + dx
            pred_y = y
            pred_z = z + 1
            
            # Check if within bounds
            if 0 <= pred_x < x_size and 0 <= pred_y < y_size:
                predecessors.append((pred_x, pred_y, pred_z))
    
    return predecessors


def calculate_precedence_graph_1_3(x_size: int, y_size: int, z_size: int) -> dict:
    """
    Build complete precedence graph for a block model using 1:3 slope constraints.
    
    Returns a mapping: block_id -> list of predecessor block_ids
    
    Args:
        x_size: Total blocks in X direction
        y_size: Total blocks in Y direction
        z_size: Total blocks in Z direction
    
    Returns:
        Dictionary mapping block_id to list of predecessor block_ids.
        Blocks with no predecessors (surface blocks at z_max) map to empty list.
    
    Example:
        For a 25x25x25 model:
        >>> graph = calculate_precedence_graph_1_3(25, 25, 25)
        >>> graph[625]  # Block at (0, 0, 1)
        [1250, 1875, 0]  # block_ids for (0, 0, 2), (1, 0, 2), (−1, 0, 2)
        # Note: (−1, 0, 2) is out of bounds, so only valid predecessors included
    """
    precedence_graph = {}
    
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                block_id = calculate_block_id(x, y, z, x_size, y_size)
                
                # Get predecessor coordinates
                pred_coords = get_slope_predecessors_1_3(x, y, z, x_size, y_size, z_size)
                
                # Convert coordinates to block IDs
                pred_ids = [
                    calculate_block_id(px, py, pz, x_size, y_size)
                    for px, py, pz in pred_coords
                ]
                
                precedence_graph[block_id] = pred_ids
    
    return precedence_graph


def generate_2d_deposit(width, depth):
    """
    Generates a synthetic 2D block model.
    Value increases with depth (simulating a deep ore body)
    but overburden (waste) is on top.
    """
    blocks = []
    grid = {}  # To easily find neighbors

    block_id_counter = 0

    for y in range(depth):  # y is depth (0 is surface)
        for x in range(width):
            # Procedural generation of Ore vs Waste
            # Ore is a blob in the middle/bottom

            center_x = width // 2
            center_y = depth // 2 + 2
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            tonnage = 1000  # fixed tonnages per block

            # Grade distribution
            if dist < width / 4:
                grade = 1.5 + np.random.normal(0, 0.2)  # High grade ore
                val_per_ton = (grade * 50) - 20  # Revenue - Processing Cost
                value = val_per_ton * tonnage
                block_type = "Ore"
            elif dist < width / 2.5:
                grade = 0.6 + np.random.normal(0, 0.1)  # Low grade ore
                val_per_ton = (grade * 50) - 20
                value = val_per_ton * tonnage
                block_type = "LowGrade"
            else:
                grade = 0.0  # Waste
                value = -5 * tonnage  # Mining cost only (waste)
                block_type = "Waste"

            _ = block_type  # Quiet unused variable to mirror notebook intent

            b = Block(block_id_counter, x, y, 0, tonnage, grade, value)
            blocks.append(b)
            grid[(x, y)] = b
            block_id_counter += 1

    # Add precedence (slope constraints)
    # 1-to-3 pattern: To mine (x, y), you need (x-1, y-1), (x, y-1), (x+1, y-1)
    for b in blocks:
        if b.y > 0:  # If not on surface
            predecessors_coords = [
                (b.x - 1, b.y - 1),
                (b.x, b.y - 1),
                (b.x + 1, b.y - 1),
            ]
            for px, py in predecessors_coords:
                if (px, py) in grid:
                    b.add_predecessor(grid[(px, py)])

    return blocks, width, depth


def read_block_model_csv(csv_path: str) -> Tuple[List[Block], int, int, int]:
    """
    Parse a block model CSV with spatial coordinates and automatically assign
    block IDs and precedence relationships based on 1:3 slope constraints.
    
    Expected CSV columns:
      - x: X coordinate (easting)
      - y: Y coordinate (northing)
      - z: Z coordinate (elevation, surface is z_max)
      - tonnage: Block tonnage
      - grade: Ore grade (optional, defaults to 0)
      - economic_value: Net present value per block
    
    Automatically calculates:
      - block_id from coordinates using row-major ordering
      - Predecessors using 1:3 slope constraint (blocks above must be mined first)
    
    Args:
        csv_path: Path to CSV file (string or Path object)
    
    Returns:
        Tuple of (blocks, x_size, y_size, z_size) where:
        - blocks: List of Block objects with predecessors populated
        - x_size, y_size, z_size: Dimensions of the block model
    
    Example:
        >>> blocks, x_size, y_size, z_size = read_block_model_csv("data/block_model.csv")
        >>> print(f"Loaded {len(blocks)} blocks in {x_size}x{y_size}x{z_size} model")
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = {"x", "y", "z", "tonnage", "economic_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Determine model dimensions from data
    x_size = int(df["x"].max()) + 1
    y_size = int(df["y"].max()) + 1
    z_size = int(df["z"].max()) + 1
    
    # Calculate precedence graph for entire model
    precedence_graph = calculate_precedence_graph_1_3(x_size, y_size, z_size)
    
    # Create blocks with assigned IDs
    blocks = []
    id_to_block = {}
    
    for _, row in df.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        z = int(row["z"])
        tonnage = float(row["tonnage"])
        grade = float(row.get("grade", 0.0))
        value = float(row["economic_value"])
        
        # Calculate block_id from coordinates
        block_id = calculate_block_id(x, y, z, x_size, y_size)
        
        # Create block
        block = Block(block_id, x, y, z, tonnage, grade, value)
        blocks.append(block)
        id_to_block[block_id] = block
    
    # Add predecessor relationships
    for block in blocks:
        pred_ids = precedence_graph.get(block.id, [])
        for pred_id in pred_ids:
            pred_block = id_to_block.get(pred_id)
            if pred_block is not None:
                block.add_predecessor(pred_block)
    
    return blocks, x_size, y_size, z_size


def read_blocks_csv(file_bytes: bytes) -> List[Block]:
    """
    Parse a CSV with columns: BlockID, Value, Tonnage, Predecessors.
    Predecessors are semicolon-separated block ids. Coordinates/grade are
    not provided; set to zero for compatibility with Block.
    """

    df = pd.read_csv(io.BytesIO(file_bytes))

    blocks = []
    pred_map = {}

    for _, row in df.iterrows():
        raw = str(row.get("Predecessors", "")).strip()
        preds = []
        if raw:
            preds = [int(x) for x in str(raw).split(";") if str(x).strip()]

        block_id = int(row["BlockID"])
        value = float(row["Value"])
        tonnage = float(row["Tonnage"])

        b = Block(block_id, 0, 0, 0, tonnage, 0.0, value)
        blocks.append(b)
        pred_map[block_id] = preds

    id_to_block = {b.id: b for b in blocks}
    for b in blocks:
        for pid in pred_map.get(b.id, []):
            pred_block = id_to_block.get(pid)
            if pred_block is not None:
                b.add_predecessor(pred_block)

    return blocks
