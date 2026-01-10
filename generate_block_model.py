"""
Generate a block model CSV with columns:

    x, y, z, tonnage, grade, economic_value

Where:
    x = easting (horizontal direction 1)
    y = northing (horizontal direction 2)
    z = elevation (vertical, positive upward)

Uses Perlin noise for realistic spatial variation with a high-grade ore blob
centered in the model. Run:
    python generate_block_model.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from perlin_noise import PerlinNoise

# Block model dimensions
X_SIZE = 10  # Easting extent (blocks)
Y_SIZE = 10  # Northing extent (blocks)
Z_SIZE = 10  # Elevation extent (blocks)

TONNAGE = 1000

# Reserve types
RESERVE_ORE = 1
RESERVE_WASTE = 0
RESERVE_OVERBURDEN = -1


def generate_output_filename(x_size: int, y_size: int, z_size: int) -> Path:
    """Generate output filename with dimensions and timestamp."""    
    filename = f"block_model_{x_size}_{y_size}_{z_size}.csv"
    return Path("data") / filename


def grade_value(x: int, y: int, z: int, noise_gen, x_size: int, y_size: int, z_size: int) -> tuple[float, float]:
    """Generate grade and economic value with high-grade ore blob centered in the model."""
    # Scale coordinates to perlin noise space for randomness
    scale = 0.08
    
    # Generate multiple layers of Perlin noise for high variation
    noise1 = noise_gen([x * scale, y * scale, z * scale])
    
    # Create a spatial distance function - closer to center = higher grade potential
    center_x, center_y, center_z = x_size / 2, y_size / 2, z_size / 2
    dist_to_center = np.sqrt(
        ((x - center_x) / (x_size / 2)) ** 2 +
        ((y - center_y) / (y_size / 2)) ** 2 +
        ((z - center_z) / (z_size / 2)) ** 2
    )
    
    # Distance-based factor: max at center (0), decreases with distance
    # Uses exponential decay to create a sharp blob of high grades
    distance_factor = np.exp(-2.0 * dist_to_center)
    
    # Perlin noise (ranges -1 to 1, scale to 0-1)
    noise_component = (noise1 + 1.0) / 2.0
    
    # Grade: base high value in center, modulated by noise for randomness
    # Range 0-5 with higher concentrations in center
    grade = 2.0 * distance_factor + 2.5 * noise_component * distance_factor
    grade = max(grade, 0.0)
    grade = min(grade, 5.0)  # Cap at 5
    
    economic_value = (grade * 50 - 20) * TONNAGE
    return round(grade, 4), round(economic_value, 2)


def generate_contiguous_region(x_size: int, y_size: int, z_size: int, target_size: int = 15, seed_point: tuple = None) -> set:
    """
    Generate a random contiguous 3D region using flood-fill approach.
    
    Args:
        x_size, y_size, z_size: Model dimensions
        target_size: Target number of blocks in region
        seed_point: Optional (x, y, z) starting point, otherwise random
    
    Returns:
        Set of (x, y, z) coordinates forming a contiguous region
    """
    if seed_point is None:
        # Pick random starting point
        seed_point = (
            random.randint(0, x_size - 1),
            random.randint(0, y_size - 1),
            random.randint(0, z_size - 1)
        )
    
    region = {seed_point}
    candidates = deque([seed_point])
    
    while len(region) < target_size and candidates:
        current = candidates.popleft()
        x, y, z = current
        
        # Check all 6 neighbors (±1 in each direction)
        neighbors = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1)
        ]
        
        # Randomize order to create irregular shapes
        random.shuffle(neighbors)
        
        for nx, ny, nz in neighbors:
            # Check bounds
            if (0 <= nx < x_size and 0 <= ny < y_size and 0 <= nz < z_size):
                if (nx, ny, nz) not in region:
                    # Probabilistic growth (90% chance to add neighbor)
                    if random.random() < 0.9:
                        region.add((nx, ny, nz))
                        candidates.append((nx, ny, nz))
                        
                        if len(region) >= target_size:
                            break
    
    return region


def generate_reserve_regions(x_size: int, y_size: int, z_size: int, min_size: int = 50) -> dict:
    """
    Generate three contiguous reserve regions: ore, waste, and overburden.
    Ore and waste regions are placed close to each other.
    
    Args:
        x_size, y_size, z_size: Model dimensions
        min_size: Minimum blocks per region
    
    Returns:
        Dictionary mapping (x, y, z) coordinates to reserve type (1=ore, 0=waste, -1=overburden)
    """
    reserve_map = {}
    
    # Generate ore region first (center-ish, middle depth)
    ore_seed = (
        random.randint(x_size // 4, 3 * x_size // 4),
        random.randint(y_size // 4, 3 * y_size // 4),
        random.randint(z_size // 4, 3 * z_size // 4)
    )
    ore_region = generate_contiguous_region(x_size, y_size, z_size, 
                                            target_size=random.randint(min_size, min_size + 10), 
                                            seed_point=ore_seed)
    
    for coord in ore_region:
        reserve_map[coord] = RESERVE_ORE
    
    # Generate waste region near ore (pick a point adjacent to ore region)
    ore_boundary = list(ore_region)
    waste_seed = random.choice(ore_boundary)
    # Offset slightly to ensure it's adjacent but not overlapping
    wx, wy, wz = waste_seed
    waste_seed = (
        max(0, min(x_size - 1, wx + random.randint(-2, 2))),
        max(0, min(y_size - 1, wy + random.randint(-2, 2))),
        max(0, min(z_size - 1, wz + random.randint(-1, 1)))
    )
    
    waste_region = generate_contiguous_region(x_size, y_size, z_size, 
                                              target_size=random.randint(min_size, min_size + 10),
                                              seed_point=waste_seed)
    
    # Remove any overlap with ore
    waste_region = waste_region - ore_region
    
    for coord in waste_region:
        reserve_map[coord] = RESERVE_WASTE
    
    # Generate overburden region (typically near surface, higher Z)
    overburden_seed = (
        random.randint(0, x_size - 1),
        random.randint(0, y_size - 1),
        random.randint(max(0, z_size - z_size // 3), z_size - 1)  # Upper third
    )
    
    overburden_region = generate_contiguous_region(x_size, y_size, z_size,
                                                   target_size=random.randint(min_size, min_size + 10),
                                                   seed_point=overburden_seed)
    
    # Remove any overlap with ore and waste
    overburden_region = overburden_region - ore_region - waste_region
    
    for coord in overburden_region:
        reserve_map[coord] = RESERVE_OVERBURDEN
    
    return reserve_map


def adjust_grade_for_reserve(base_grade: float, base_value: float, reserve_type: int) -> tuple[float, float]:
    """
    Adjust grade and economic value based on reserve type.
    
    Args:
        base_grade: Original calculated grade
        base_value: Original calculated economic value
        reserve_type: 1 (ore), 0 (waste), -1 (overburden/other)
    
    Returns:
        Adjusted (grade, economic_value) tuple
    """
    if reserve_type == RESERVE_ORE:
        # Boost ore blocks: increase grade by 50-100%
        multiplier = random.uniform(1.5, 2.0)
        grade = min(5.0, base_grade * multiplier)
        economic_value = (grade * 50 - 20) * TONNAGE
    elif reserve_type == RESERVE_WASTE:
        # Waste blocks: very low grade, negative economic value
        grade = random.uniform(0.0, 0.15)
        economic_value = (grade * 50 - 20) * TONNAGE  # Will be negative
    else:  # RESERVE_OVERBURDEN or not in any region
        # Keep original values
        grade = base_grade
        economic_value = base_value
    
    return round(grade, 4), round(economic_value, 2)


def write_csv(x_size: int = X_SIZE, y_size: int = Y_SIZE, z_size: int = Z_SIZE) -> tuple[Path, Path]:
    """Generate and write the block model CSV with Perlin noise variation.
    
    Also generates a companion JSON metadata file with attribute statistics.
    
    Args:
        x_size: Size in easting direction
        y_size: Size in northing direction
        z_size: Size in elevation direction
    
    Returns:
        Tuple of (CSV path, JSON metadata path)
    """
    output_path = generate_output_filename(x_size, y_size, z_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize Perlin noise generator with a seed for reproducibility
    noise_gen = PerlinNoise(octaves=4, seed=42)
    
    # Generate reserve regions
    print("Generating reserve regions...")
    reserve_map = generate_reserve_regions(x_size, y_size, z_size, min_size=10)
    
    ore_count = sum(1 for v in reserve_map.values() if v == RESERVE_ORE)
    waste_count = sum(1 for v in reserve_map.values() if v == RESERVE_WASTE)
    overburden_count = sum(1 for v in reserve_map.values() if v == RESERVE_OVERBURDEN)
    
    print(f"  Ore blocks: {ore_count}")
    print(f"  Waste blocks: {waste_count}")
    print(f"  Overburden blocks: {overburden_count}")
    
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "tonnage", "grade", "economic_value", "reserve"])

        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    # Get base grade and value from Perlin noise
                    base_grade, base_value = grade_value(x, y, z, noise_gen, x_size, y_size, z_size)
                    
                    # Check if block is in a reserve region
                    coord = (x, y, z)
                    reserve_type = reserve_map.get(coord, None)
                    
                    # Adjust grade/value if in a reserve region
                    if reserve_type is not None:
                        grade, value = adjust_grade_for_reserve(base_grade, base_value, reserve_type)
                    else:
                        grade, value = base_grade, base_value
                        reserve_type = -1  # Default: not in any special region
                    
                    writer.writerow([x, y, z, TONNAGE, grade, value, reserve_type])
    
    # Generate metadata JSON file with attribute statistics
    df = pd.read_csv(output_path)
    metadata = {
        "dimensions": {
            "x": {"min": int(df["x"].min()), "max": int(df["x"].max())},
            "y": {"min": int(df["y"].min()), "max": int(df["y"].max())},
            "z": {"min": int(df["z"].min()), "max": int(df["z"].max())},
        },
        "attributes": {
            "tonnage": {
                "min": float(df["tonnage"].min()),
                "max": float(df["tonnage"].max()),
                "mean": float(df["tonnage"].mean()),
                "unit": "tons",
            },
            "grade": {
                "min": float(df["grade"].min()),
                "max": float(df["grade"].max()),
                "mean": float(df["grade"].mean()),
            },
            "economic_value": {
                "min": float(df["economic_value"].min()),
                "max": float(df["economic_value"].max()),
                "mean": float(df["economic_value"].mean()),
                "unit": "$",
            },
            "reserve": {
                "ore_blocks": int((df["reserve"] == 1).sum()),
                "waste_blocks": int((df["reserve"] == 0).sum()),
                "overburden_blocks": int((df["reserve"] == -1).sum()),
                "description": "1=ore, 0=waste, -1=overburden/other"
            }
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    # Create JSON file with same name
    json_path = output_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    return output_path, json_path


if __name__ == "__main__":
    csv_file, json_file = write_csv()
    print(f"Wrote CSV to {csv_file.resolve()}")
    print(f"Wrote metadata to {json_file.resolve()}")


