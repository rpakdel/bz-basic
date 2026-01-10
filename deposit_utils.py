import numpy as np
from bz_algorithm_logic import Block


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
