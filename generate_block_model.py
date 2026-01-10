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


def generate_output_filename(x_size: int, y_size: int, z_size: int) -> Path:
    """Generate output filename with dimensions and timestamp."""
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M")
    filename = f"block_model_{x_size}_{y_size}_{z_size}_{timestamp}.csv"
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
    
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "tonnage", "grade", "economic_value"])

        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    grade, value = grade_value(x, y, z, noise_gen, x_size, y_size, z_size)
                    writer.writerow([x, y, z, TONNAGE, grade, value])
    
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


