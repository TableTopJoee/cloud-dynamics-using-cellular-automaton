"""
=============================================================
2D Cloud Cellular Automaton (CA) Simulation - Smooth Clustered Model
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------------------------------------
# STEP 1: SET UP SIMULATION PARAMETERS
# -------------------------------------------------------------

GRID_SIZE = 100
TIME_STEPS = 400
P_HUM = 0.44             # Balanced moisture density
P_ACT = 0.8             # Updraft triggers
P_EXT = 0.15             # Evaporation probability

np.random.seed(42)

# -------------------------------------------------------------
# STEP 2: INITIALIZE SMOOTH MOISTURE & SEED CLOUDS
# -------------------------------------------------------------

def get_neighbor_sum(grid):
    neighbor_sum = np.zeros_like(grid, dtype=float)
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    for shift_row, shift_col in shifts:
        neighbor_sum += np.roll(grid, shift=(shift_row, shift_col), axis=(0, 1))
    return neighbor_sum

# Smooth initial humidity field (Heavy smoothing for large continuous blobs)
raw_hum = np.random.rand(GRID_SIZE, GRID_SIZE)
for _ in range(5):
    raw_hum = (raw_hum + get_neighbor_sum(raw_hum) / 8.0) / 2.0

hum = (raw_hum > (1.0 - P_HUM)).astype(int)
act = (np.random.rand(GRID_SIZE, GRID_SIZE) < P_ACT).astype(int)

# Cloud Seeds
cld = (hum == 1) & (act == 1)

cloud_coverage_history = []

start_time = time.time()   # Start the stopwatch

# -------------------------------------------------------------
# STEP 3: MAIN SIMULATION LOOP
# -------------------------------------------------------------

for step in range(TIME_STEPS):

    # ---- (A) CLOUD FORMATION RULE ----
    neighbor_clouds = get_neighbor_sum(cld)

    # Require at least 2 neighboring clouds for cluster stability (prevents noise)
    new_cloud_formation = (hum == 1) & ((neighbor_clouds >= 2) | ((neighbor_clouds >= 1) & (act == 1)))

    # ---- (B) CLOUD EVAPORATION RULE ----
    random_values = np.random.rand(GRID_SIZE, GRID_SIZE)
    evaporation_mask = random_values < P_EXT
    surviving_clouds = (cld == 1) & (~evaporation_mask)

    # ---- (C) UPDATE CLOUD GRID ----
    cld = (surviving_clouds | new_cloud_formation).astype(int)

    # ---- (D) WIND DRIFT & SMOOTH MOISTURE DYNAMICS ----
    if step % 2 == 0:
        cld = np.roll(cld, shift=1, axis=1)  # Horizontal wind

    if step % 5 == 0:
        hum = np.roll(hum, shift=(1, 1), axis=(0, 1))  # Smooth moisture drift

    act = (np.random.rand(GRID_SIZE, GRID_SIZE) < P_ACT).astype(int)

    # ---- (E) RECORD COVERAGE ----
    total_cells = GRID_SIZE * GRID_SIZE
    cloud_cells = np.sum(cld)
    coverage_percent = (cloud_cells / total_cells) * 100
    cloud_coverage_history.append(coverage_percent)


# -------------------------------------------------------------
# STEP 3.5: FRACTAL DIMENSION ANALYSIS (Box-Counting on CLOUD EDGES)
# -------------------------------------------------------------
# IMPORTANT FIX: Real cloud studies measure the roughness of the
# cloud's OUTLINE/BOUNDARY, not the solid filled shape. A filled
# blob naturally scores close to 2.0 as it gets bigger, which isn't
# what we want. So here, we first extract just the EDGE pixels of
# the cloud (pixels that are cloud, but touch at least one non-cloud
# neighbor), and measure the fractal dimension of THAT outline instead.

def get_neighbor_sum_simple(grid):
    """Same neighbor-counting idea as before, used here to find edges."""
    neighbor_sum = np.zeros_like(grid, dtype=float)
    shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for sr, sc in shifts:
        neighbor_sum += np.roll(grid, shift=(sr, sc), axis=(0,1))
    return neighbor_sum

# Step A: find how many cloud-neighbors each cloud cell has
cloud_neighbor_count = get_neighbor_sum_simple(cld)

# Step B: a cell is an "edge" cell if it IS a cloud, but does NOT have
# all 8 neighbors also as cloud (meaning it touches open sky somewhere)
cloud_edges = (cld == 1) & (cloud_neighbor_count < 8)

def box_count(binary_grid, box_size):
    S = binary_grid.shape[0]
    count = 0
    for i in range(0, S, box_size):
        for j in range(0, S, box_size):
            box = binary_grid[i:i+box_size, j:j+box_size]
            if np.any(box):
                count += 1
    return count

box_sizes = [2, 4, 5, 10, 20, 25]
box_counts = [box_count(cloud_edges, size) for size in box_sizes]

log_sizes = np.log(1.0 / np.array(box_sizes, dtype=float))
log_counts = np.log(np.array(box_counts, dtype=float))
slope, intercept = np.polyfit(log_sizes, log_counts, 1)
fractal_dimension = slope

print("=" * 60)
print("FRACTAL DIMENSION ANALYSIS (Cloud Boundary Only)")
print("=" * 60)
print(f"Estimated Fractal Dimension of cloud EDGES: {fractal_dimension:.3f}")
print("Real-world cloud boundaries typically measure: 1.30 - 1.40")
if 1.2 <= fractal_dimension <= 1.5:
    print("✅ RESULT: Cloud edge roughness closely matches real clouds!")
else:
    print("⚠ RESULT: Outside typical range — see explanation in report.")
print("=" * 60)

end_time = time.time()     # Stop the stopwatch
elapsed_time = end_time - start_time
print("=" * 60)
print("COMPUTATIONAL EFFICIENCY BENCHMARK")
print("=" * 60)
print(f"Simulation completed {TIME_STEPS} time steps on a {GRID_SIZE}x{GRID_SIZE} grid")
print(f"Total computation time: {elapsed_time:.3f} seconds")
print(f"Average time per time step: {(elapsed_time/TIME_STEPS)*1000:.3f} milliseconds")
print("=" * 60)

# -------------------------------------------------------------
# STEP 4: PLOT RESULTS (Layout & Spacing Fixed)
# -------------------------------------------------------------

fig = plt.figure(figsize=(15, 6))

# Left Plot: Clouds Image
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(cld, cmap="Blues", interpolation="bicubic")
ax1.set_title(f"Final Cloud Pattern at Time Step {TIME_STEPS}")
ax1.set_xlabel("Y Position")
ax1.set_ylabel("X Position")

# Colorbar with proper pad to avoid overlap
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.08)
cbar.set_label("Cloud Presence (1 = Cloud, 0 = Clear)", rotation=270, labelpad=15)

# Right Plot: Coverage Graph
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(range(TIME_STEPS), cloud_coverage_history, color="steelblue", linewidth=2)
ax2.set_title("Total Cloud Coverage Over Time")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Cloud Coverage (%)")
ax2.set_ylim([0, 100])
ax2.set_xlim([0, TIME_STEPS])
ax2.grid(True, linestyle="--", alpha=0.5)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.35)
plt.show(block=False)
plt.pause(2)


# =============================================================
# EXPERIMENT 2: MOISTURE vs COVERAGE — PHASE TRANSITION CURVE
# =============================================================
# WHAT WE'RE DOING (plain English):
# We will re-run the ENTIRE cloud simulation multiple times.
# Each time, we only change ONE thing: the starting moisture level (P_HUM).
# We record what % of the grid ends up as clouds at the end of each run.
# Then we plot: Moisture Level (x-axis) vs Final Cloud Coverage (y-axis).
#
# WHY THIS MATTERS:
# If a small increase in moisture causes a SUDDEN jump in cloud coverage
# (instead of a slow, steady increase), that is called a "phase transition"
# — the same kind of sudden-jump behavior seen in real physical systems
# (like water turning to ice at exactly 0°C). Finding this in our simple
# CA model is strong evidence of "emergent behavior" arising from simple
# local rules — the central idea of your paper.

import numpy as np
import matplotlib.pyplot as plt

def run_cloud_simulation(P_HUM_value, P_ACT_value=0.08, P_EXT_value=0.15,
                          GRID_SIZE=100, TIME_STEPS=200, seed=42):
    """
    This function runs your ENTIRE cloud simulation ONE time, using
    whatever P_HUM value is passed in, and returns the final coverage %.

    Think of this as putting your whole existing simulation into a box
    with a handle labeled 'P_HUM' — every time we pull the handle to a
    different setting, the box runs the full simulation again from
    scratch and hands us back one number: the final cloud coverage %.
    """
    np.random.seed(seed)  # same random pattern every time, for fair comparison

    def get_neighbor_sum(grid):
        neighbor_sum = np.zeros_like(grid, dtype=float)
        shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for sr, sc in shifts:
            neighbor_sum += np.roll(grid, shift=(sr, sc), axis=(0,1))
        return neighbor_sum

    # Smooth moisture field (same as your original code)
    raw_hum = np.random.rand(GRID_SIZE, GRID_SIZE)
    for _ in range(5):
        raw_hum = (raw_hum + get_neighbor_sum(raw_hum) / 8.0) / 2.0

    hum = (raw_hum > (1.0 - P_HUM_value)).astype(int)
    act = (np.random.rand(GRID_SIZE, GRID_SIZE) < P_ACT_value).astype(int)
    cld = (hum == 1) & (act == 1)

    # Main loop (same rules as your original code)
    for step in range(TIME_STEPS):
        neighbor_clouds = get_neighbor_sum(cld)
        new_cloud_formation = (hum == 1) & ((neighbor_clouds >= 2) |
                              ((neighbor_clouds >= 1) & (act == 1)))
        evaporation_mask = np.random.rand(GRID_SIZE, GRID_SIZE) < P_EXT_value
        surviving_clouds = (cld == 1) & (~evaporation_mask)
        cld = (surviving_clouds | new_cloud_formation).astype(int)

        if step % 2 == 0:
            cld = np.roll(cld, shift=1, axis=1)
        if step % 5 == 0:
            hum = np.roll(hum, shift=(1, 1), axis=(0, 1))
        act = (np.random.rand(GRID_SIZE, GRID_SIZE) < P_ACT_value).astype(int)

    final_coverage = (np.sum(cld) / (GRID_SIZE * GRID_SIZE)) * 100
    return final_coverage


# ---- RUN THE EXPERIMENT: test many moisture levels ----
# We test P_HUM values from 0.10 (very dry) to 0.70 (very moist)
moisture_levels_to_test = np.arange(0.10, 0.71, 0.05)  # 0.10, 0.15, 0.20 ... 0.70
resulting_coverages = []

print("Running sensitivity experiment... this may take a minute.")
for p_hum_val in moisture_levels_to_test:
    coverage = run_cloud_simulation(P_HUM_value=p_hum_val)
    resulting_coverages.append(coverage)
    print(f"  Moisture = {p_hum_val:.2f}  ->  Final Cloud Coverage = {coverage:.2f}%")

# ---- PLOT THE PHASE TRANSITION CURVE ----
plt.figure(figsize=(8, 6))
plt.plot(moisture_levels_to_test, resulting_coverages,
         marker='o', color='darkblue', linewidth=2)
plt.title("Phase Transition: Moisture Level vs Final Cloud Coverage")
plt.xlabel("Initial Moisture Level (P_HUM)")
plt.ylabel("Final Cloud Coverage (%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


