class Block:
    """
    Represents a single mining block in the model.
    """

    def __init__(self, id, x, y, z, tonnage, grade, value):
        self.id = id
        self.x = x  # Horizontal
        self.y = y  # Vertical (Depth)
        self.z = z  # 3rd Dimension (optional, 0 for 2D)
        self.tonnage = tonnage
        self.grade = grade
        self.economic_value = value
        self.predecessors = []  # Blocks that must be mined before this one (physically above)

    def add_predecessor(self, block):
        self.predecessors.append(block)
