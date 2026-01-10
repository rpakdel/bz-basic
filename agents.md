## BZ Algorithm Agent Guide

- Demonstrates a small Bienstock–Zuckerberg (BZ) scheduler for open-pit style block sequencing.
- Generates a synthetic 2D orebody, enforces simple slope precedence, relaxes mining/processing caps with Lagrangian multipliers, and solves a max-closure problem via min-cut.

### Streamlit UI

- Sidebar controls for geometry, periods, discount, max iterations, step size, mining cap, processing cap factor (% of ore tonnage), and RNG seed.
- Click **Run Optimization** to solve and view plots plus a convergence table.
- Processing cap is computed from the chosen factor after the first pass so both caps reflect the sidebar choices.

### Parameter notes

- Mining capacity: tons/period hard cap in the subgradient update; violations show in the convergence log.
- Processing capacity: applied only to ore-tonnage blocks (value > 0); set via fraction of total ore in Streamlit or fixed in the CLI.
- Discounting: NPV computed per block per period using $(1 + r)^{-(t+1)}$; weights compare mining now vs delaying one period.
- Precedence: 1:3 slope (block requires three blocks above). Temporal consistency enforced with infinite-capacity edges across periods.

### Outputs

- `bz_result.png` (CLI) or inline matplotlib figure (Streamlit) with extraction periods heatmap and NPV/violation history.
- Console/Streamlit logs show iteration, NPV, and total violation tons; stops early when violations reach 0 after a few iterations.

### Extending

- Deposit shape: edit grade/value logic or blob geometry in [deposit_utils.py](deposit_utils.py).
- Constraints: change mining/processing caps or add new resources by extending multiplier arrays and subgradient updates in [bz_algorithm_logic.py](bz_algorithm_logic.py).
- Tuning: adjust `max_iterations` and `step_size_factor` in callers; smaller steps improve stability, larger steps speed progress but risk oscillation.
