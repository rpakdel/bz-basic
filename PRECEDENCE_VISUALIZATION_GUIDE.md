# Precedence Visualization in Streamlit - User Guide

## New Feature: Interactive Precedence Graph Visualization

The Streamlit app now includes **interactive precedence visualization** directly in the "View Block Model" mode!

## How to Use

### 1. **Navigate to "View Block Model" Mode**
   - Open the Streamlit app
   - In the sidebar, select **"View Block Model"** from the Algorithm dropdown

### 2. **Enable Precedence Visualization**
   - In the sidebar, find the **"Precedence Visualization"** section
   - Check the box **"Show Precedence Relationships"**

### 3. **Choose Visualization Mode**

#### **Single Block Mode**
Perfect for understanding dependencies of specific blocks:
- Select **"Single Block"** mode
- Enter a **Block ID** (e.g., 1000, 625, 15000)
- See the block (red circle) and its predecessors (blue diamonds)
- Arrows show which blocks must be mined first
- Hover over blocks for detailed info

**Tip:** Try different block IDs:
- Low IDs (0-624): Bottom blocks, 2-3 predecessors
- Mid IDs (5000-10000): Middle depth blocks
- High IDs (15000+): Surface blocks, 0 predecessors

#### **Region View Mode**
Visualize precedence relationships across a spatial region:
- Select **"Region View"** mode
- Use X, Y, Z sliders to filter the region
- See all blocks and precedence links in that region
- Blocks are colored by elevation (Z level)
- Gray lines show dependencies between blocks

**Tip:** For clearer visualization:
- Start with a small region (e.g., Z: 20-24 for surface)
- Gradually expand to see deeper levels
- Use smaller X/Y ranges for less clutter

### 4. **Understanding the Visualization**

**Color Coding:**
- 🔴 **Red Circle**: Target block (Single Block mode)
- 🔷 **Blue Diamonds**: Predecessor blocks (must mine these first)
- 🟢 **Gradient Colors**: Z-level (Region mode - lighter = higher elevation)

**Arrows/Lines:**
- Point from **predecessor → dependent block**
- Gray dashed lines show "must mine first" relationships

**Key Metrics Shown:**
- Total precedence links in dataset
- Surface blocks (no predecessors)
- Average predecessors per block

### 5. **Real-Time Interaction**
- **Rotate**: Click and drag to rotate 3D view
- **Zoom**: Scroll to zoom in/out
- **Pan**: Right-click and drag
- **Hover**: Mouse over blocks for details

## Example Use Cases

### Understanding Surface Mining:
1. Set Z range to (24, 24) - top level
2. Enable Region View
3. Notice: All blocks have 0 predecessors (can mine first!)

### Analyzing Slope Constraints:
1. Pick a mid-depth block (e.g., ID 7500)
2. Enable Single Block mode
3. See exactly which 3 blocks above must be mined first
4. Notice the 1:3 slope pattern (left, center, right)

### Visualizing a Mining Sequence:
1. Start with Z range (20, 24)
2. See surface blocks ready to mine
3. Gradually decrease Z to see deeper dependencies
4. Understand how mining progresses top-down

## Technical Details

**1:3 Slope Constraint:**
- Each block requires 3 predecessors above it
- Pattern: (x-1, y, z+1), (x, y, z+1), (x+1, y, z+1)
- Surface blocks (Z=max) have no predecessors
- Edge blocks may have only 2 predecessors

**Block ID Calculation:**
- `block_id = x + (y × x_size) + (z × x_size × y_size)`
- Sequential ordering: X fastest, then Y, then Z

**Performance:**
- Initial load may take a few seconds for large models
- Visualizations are interactive and responsive
- Filter regions for faster rendering with large datasets

## Tips for Best Experience

1. **Start Simple**: Begin with Single Block mode on a few blocks
2. **Use Filters**: Narrow X/Y/Z ranges for clearer Region View
3. **Check Surface First**: Set Z to maximum to see starting blocks
4. **Follow Dependencies**: Trace chains from surface to depth
5. **Compare Blocks**: Try corner blocks vs. interior blocks to see different predecessor patterns

Enjoy exploring your block model's precedence structure! 🎉
