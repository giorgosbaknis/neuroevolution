"""Experiment 1. Search two fitness landscapes with a genetic algorithm.

Install dependencies with: python3 -m pip install numpy 'matplotlib>=3.8'
Run this file to recreate the plots from Figures 2.4 and 2.5 in results/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# Change one setting at a time. Try MUTATION_SCALE = 0 with the same seed
# to see how far crossover can get using only the original coordinate values.
SEED = 42
POPULATION_SIZE = 100  # Use at least two points.
GENERATIONS = 20
MUTATION_SCALE = 0.04  # Mutation standard deviation as a fraction of the map width.


# 1. Define the fitness landscapes.
# The shifts and map bounds are estimates based on Figure 2.4.
# Negate each cost so the algorithm can search for the highest fitness (0).
SCHAFFER_PEAK = (-200.0, 200.0)
RASTRIGIN_PEAK = (-5.4, 5.4)


def schaffer(x, y):
    # The sine term creates rings around the peak.
    radius = np.hypot(x - SCHAFFER_PEAK[0], y - SCHAFFER_PEAK[1])
    return -np.sqrt(radius) * (1 + np.sin(50 * radius**0.2) ** 2)


def rastrigin(x, y):
    # Independent ripples in x and y create a grid of peaks.
    dx = x - RASTRIGIN_PEAK[0]
    dy = y - RASTRIGIN_PEAK[1]
    ripples = np.sin(np.pi * dx) ** 2 + np.sin(np.pi * dy) ** 2
    return -(dx**2 + dy**2 + 20 * ripples)


# 2. Evolve a population. Each point [x, y] is one chromosome with two genes.
def run_genetic_algorithm(fitness, lower, upper):
    rng = np.random.default_rng(SEED)
    width = upper - lower
    mutation_std = MUTATION_SCALE * width
    elite_count = max(1, POPULATION_SIZE // 10)

    # Initialize generation 0 with random points near the middle of the map.
    center = (lower + upper) / 2
    initial_points = rng.normal(center, 0.04 * width, size=(POPULATION_SIZE, 2))
    candidates = np.clip(initial_points, lower, upper).tolist()
    elites = []
    history = []

    for generation in range(GENERATIONS + 1):
        # Evaluate and rank the population. fitness(*point) calls fitness(x, y).
        population = elites + candidates
        ranked = sorted(population, key=lambda point: fitness(*point), reverse=True)
        best = ranked[0]
        history.append((candidates, elites, best))  # Save points for plotting later.
        if generation == GENERATIONS:
            break

        # Select the top 10% as parents (truncation selection).
        # These parents also carry over to the next generation unchanged (elitism).
        elites = ranked[:elite_count]
        candidates = []
        for _ in range(POPULATION_SIZE - elite_count):
            # Uniform crossover. Pick two parents and flip one coin per gene.
            a, b = rng.integers(len(elites), size=2)
            parent_a, parent_b = elites[a], elites[b]
            x = parent_a[0] if rng.random() < 0.5 else parent_b[0]
            y = parent_a[1] if rng.random() < 0.5 else parent_b[1]

            # Mutation. Add a random offset to each gene and stay inside the map.
            dx, dy = rng.normal(0, mutation_std, size=2)
            child = np.clip([x + dx, y + dy], lower, upper).tolist()
            candidates.append(child)

    return history


# 3. Run the same search on both landscapes and plot a few generations.
snapshots = np.linspace(0, GENERATIONS, min(GENERATIONS + 1, 6), dtype=int)
maps, map_axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
progress, progress_axes = plt.subplots(
    2, len(snapshots), figsize=(2.4 * len(snapshots) + 0.8, 5.4),
    squeeze=False, layout="constrained",
)
maps.suptitle("2D fitness landscapes · lighter means higher fitness")
progress.suptitle(f"Genetic algorithm progress · seed {SEED}")

orange = LinearSegmentedColormap.from_list("schaffer", ["#e66017", "#ffe872"])
landscapes = [
    ("Schaffer F7", schaffer, SCHAFFER_PEAK, -300, 300, orange),
    ("Rastrigin", rastrigin, RASTRIGIN_PEAK, -8.8, 8.8, "afmhot"),
]

for row, (name, fitness, peak, lower, upper, colors) in enumerate(landscapes):
    history = run_genetic_algorithm(fitness, lower, upper)

    # Evaluate the landscape on a grid once. Each score becomes one colored pixel.
    step = (upper - lower) / 720
    coordinates = np.linspace(lower + step / 2, upper - step / 2, 720)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    grid_scores = fitness(grid_x, grid_y)
    heatmap = dict(origin="lower", extent=(lower, upper, lower, upper), cmap=colors, vmax=0)

    ax = map_axes[row]
    image = ax.imshow(grid_scores, **heatmap)
    ax.set(title=name, xlabel="x", ylabel="y")
    ax.axhline(0, color="black", alpha=0.35, linewidth=0.6)
    ax.axvline(0, color="black", alpha=0.35, linewidth=0.6)
    ax.annotate("Maximum", xy=peak, xytext=(0.48, 0.9), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->"})
    maps.colorbar(image, ax=ax, shrink=0.74, label="Fitness")

    for column, generation in enumerate(snapshots):
        ax = progress_axes[row, column]
        ax.imshow(grid_scores, **heatmap)

        # Draw the new candidates in blue, carried elites in green, and best in red.
        candidates, elites, best = history[generation]
        x, y = zip(*candidates)
        ax.scatter(x, y, s=9, color="tab:blue", alpha=0.6, label="Candidates")
        if elites:
            x, y = zip(*elites)
            ax.scatter(x, y, s=13, color="tab:green", label="Carried elites")
        ax.scatter(*best, s=70, color="tab:red", marker="*", edgecolors="white",
                   linewidths=0.5, label="Best so far")
        ax.set(xticks=[], yticks=[])
        if row == 0:
            ax.set_title(f"Generation {generation}", fontsize=10)
        if column == 0:
            ax.set_ylabel(name)

    _, _, initial_best = history[0]
    _, _, final_best = history[-1]
    print(f"{name}: fitness {fitness(*initial_best):.4f} -> {fitness(*final_best):.4f}")
    x, y = final_best
    distance = np.hypot(x - peak[0], y - peak[1])
    print(f"  Best point ({x:.3f}, {y:.3f}), distance to maximum {distance:.3f}")

# Save both figures next to this script so they are easy to find after a run.
handles, labels = progress_axes[-1, -1].get_legend_handles_labels()
progress.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
results = Path(__file__).with_name("results")
results.mkdir(exist_ok=True)
maps.savefig(results / "fitness_landscapes.png", dpi=150)
progress.savefig(results / "ga_progress.png", dpi=150)
print(f"Saved the plots in {results}")
