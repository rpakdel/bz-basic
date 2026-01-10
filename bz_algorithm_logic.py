import networkx as nx
import numpy as np
import time

class Block:
    """
    Represents a single mining block in the model.
    """
    def __init__(self, id, x, y, z, tonnage, grade, value):
        self.id = id
        self.x = x # Horizontal
        self.y = y # Vertical (Depth)
        self.z = z # 3rd Dimension (optional, 0 for 2D)
        self.tonnage = tonnage
        self.grade = grade
        self.economic_value = value
        self.predecessors = [] # Blocks that must be mined before this one (physically above)

    def add_predecessor(self, block):
        self.predecessors.append(block)

class BZScheduler:
    """
    Implements a simplified version of the Bienstock-Zuckerberg (BZ) approach
    for educational purposes.

    The Core Logic:
    1. The 'Hard' problem is the Resource Constraint (e.g., Mill Capacity per year).
    2. We relax this constraint using Lagrangian Multipliers (penalties).
    3. The problem transforms into a 'Maximum Weight Closure' problem.
    4. We solve the Closure problem efficiently using Min-Cut / Max-Flow.
    5. We iteratively update the multipliers (penalties) until the solution converges
       or satisfies constraints reasonably well.
    """
    def __init__(self, blocks, periods, discount_rate, mining_capacity, processing_capacity):
        self.blocks = blocks
        self.periods = periods
        self.discount_rate = discount_rate
        self.mining_limit = mining_capacity
        self.processing_limit = processing_capacity

        # Lagrangian Multipliers: One for each period
        # Initialize with 0 penalty
        self.lambda_mining = np.zeros(periods)
        self.lambda_processing = np.zeros(periods)

    def build_graph(self, multipliers_mining, multipliers_processing):
        """
        Constructs the flow network for the closure problem.
        Nodes are tuples: (block_id, period_index).

        This formulation uses the "By-Period" variable transformation:
        x_{b,t} = 1 if block b is mined BY period t.
        """
        G = nx.DiGraph()

        # Source and Sink for Min-Cut
        source = 'S'
        sink = 'T'
        G.add_node(source)
        G.add_node(sink)

        # 1. Calculate "Lagrangian Weight" for each node
        # The weight represents the benefit of mining block 'b' specifically in period 't'
        # adjusted by the penalty for using resources in that period.

        for t in range(self.periods):
            # Discount factor for this period
            df_current = 1 / ((1 + self.discount_rate) ** (t + 1))
            df_next = 1 / ((1 + self.discount_rate) ** (t + 2)) if t < self.periods - 1 else 0

            current_mining_penalty = multipliers_mining[t]
            current_proc_penalty = multipliers_processing[t]

            # Penalties applies to the difference between extracting in t vs t+1
            # But in the "Mined By" formulation, logic is slightly different.
            # Simplified approach: We assign the Net Present Value increment to the node.

            for block in self.blocks:
                node_id = (block.id, t)

                # Pure Economic Value of mining in this period vs waiting (or never)
                # Ideally: Value = (Price - Cost) * Tonnage
                # Here we use the pre-calculated block.economic_value

                # NPV if mined in period t
                npv_t = block.economic_value * df_current

                # NPV if mined in period t+1 (Opportunity cost of doing it now)
                npv_next = block.economic_value * df_next if t < self.periods - 1 else 0

                # Marginal gain of mining in t instead of t+1
                base_weight = npv_t - npv_next

                # Subtract Penalties (Lagrangian Relaxation)
                # Penalty = Lambda * Resource_Usage
                resource_penalty = (block.tonnage * current_mining_penalty)

                # Apply penalty only if it's ore (for processing)
                if block.economic_value > 0:
                    resource_penalty += (block.tonnage * current_proc_penalty)

                weight = base_weight - resource_penalty

                # Add Node to Graph with Capacity Logic for Min-Cut
                # Picard's transformation:
                # If Weight > 0: Edge S -> Node (Capacity = Weight)
                # If Weight < 0: Edge Node -> T (Capacity = -Weight)

                G.add_node(node_id, profit=weight)

                if weight > 0:
                    G.add_edge(source, node_id, capacity=weight)
                elif weight < 0:
                    G.add_edge(node_id, sink, capacity=-weight)
                else:
                    # Weight is 0, no edge needed to S or T
                    pass

        # 2. Add Precedence Constraints (Infinite Capacity Edges)
        for t in range(self.periods):
            for block in self.blocks:
                u = (block.id, t)

                # Spatial Precedence:
                # If we mine Block A (lower) in period t, we must mine Block B (upper) in period t.
                # Graph Edge: A -> B (Infinite capacity)
                for pred in block.predecessors:
                    v = (pred.id, t)
                    G.add_edge(u, v, capacity=float('inf'))

                # Temporal Precedence (Consistency):
                # If we mine Block A by period t-1, we effectively have mined it by period t.
                # In closure logic: If we select (A, t-1), we imply (A, t)?
                # Actually, standard formulation links (Block, t) -> (Block, t-1).
                # Meaning: You cannot mine by 't' unless you mined by 't-1'? No.
                # Correct Logic: If we mine in Period 2, we must have "mined by Period 2".
                # The constraint is x_{b,t} <= x_{b, t+1} is wrong for Closure.
                # We want: If we DON'T mine by t+1, we CAN'T mine by t.
                # Edge: (Block, t) -> (Block, t+1)

                if t < self.periods - 1:
                    next_period_node = (block.id, t+1)
                    # If we commit to mining by t, we automatically commit to mining by t+1
                    # This ensures monotonicity of the "mined by" variable.
                    G.add_edge(u, next_period_node, capacity=float('inf'))

        return G, source, sink

    def solve(self, max_iterations=20, step_size_factor=0.00005):
        """
        The Main BZ Loop.
        """
        print(f"--- Starting BZ Optimization ({self.periods} periods) ---")
        start_time = time.time()

        best_objective = float('inf')
        history = []

        for k in range(1, max_iterations + 1):
            # 1. Build Graph with current multipliers
            G, source, sink = self.build_graph(self.lambda_mining, self.lambda_processing)

            # 2. Solve Min-Cut (Equivalent to Max Weight Closure)
            # value is the capacity of the cut
            cut_value, partition = nx.minimum_cut(G, source, sink)

            # Partition[0] is the set of nodes reachable from Source (The set we keep/mine)
            reachable = partition[0]

            # 3. Decode Solution
            # x[b][t] = 1 if (b,t) is in reachable set
            current_schedule = {} # block_id -> period extracted (or None)

            # Track resource usage per period to calculate violations
            mining_usage = np.zeros(self.periods)
            processing_usage = np.zeros(self.periods)

            mined_by_matrix = np.zeros((len(self.blocks), self.periods))

            for block in self.blocks:
                for t in range(self.periods):
                    if (block.id, t) in reachable:
                        mined_by_matrix[block.id, t] = 1

            # Convert "Mined By" to "Mined In"
            # Mined In t = (Mined By t) - (Mined By t-1)
            actual_profit = 0

            for block in self.blocks:
                extract_period = -1
                for t in range(self.periods):
                    is_mined_by_t = mined_by_matrix[block.id, t]
                    is_mined_by_prev = mined_by_matrix[block.id, t-1] if t > 0 else 0

                    if is_mined_by_t == 1 and is_mined_by_prev == 0:
                        extract_period = t
                        # Tally usage
                        mining_usage[t] += block.tonnage
                        if block.economic_value > 0:
                            processing_usage[t] += block.tonnage

                        # Add to NPV
                        df = 1 / ((1 + self.discount_rate) ** (t + 1))
                        actual_profit += block.economic_value * df
                        break

                current_schedule[block.id] = extract_period

            # 4. Calculate Subgradients (Violations)
            # Gradient > 0 implies we used too much resource -> Increase Penalty
            # Gradient < 0 implies we have slack -> Decrease Penalty
            grad_mining = mining_usage - self.mining_limit
            grad_processing = processing_usage - self.processing_limit

            # 5. Update Multipliers (Subgradient Method)
            # Step size usually decays (1/k) or uses Polyak's rule.
            # Using a simple decaying step size for demo stability.
            step = step_size_factor / (k**0.5)

            self.lambda_mining = np.maximum(0, self.lambda_mining + step * grad_mining)
            self.lambda_processing = np.maximum(0, self.lambda_processing + step * grad_processing)

            # Log status
            total_violation = np.sum(np.maximum(0, grad_mining)) + np.sum(np.maximum(0, grad_processing))
            history.append((k, actual_profit, total_violation))

            print(f"Iter {k:02d} | Profit: ${actual_profit:,.0f} | Violations: {total_violation:,.0f} tons")

            if total_violation == 0 and k > 5:
                print("Converged to feasible solution.")
                break

        print(f"--- Finished in {time.time() - start_time:.2f}s ---")
        return current_schedule, history
