from dataclasses import dataclass
import math
from typing import Callable

from matplotlib import pyplot as plt


@dataclass
class Solution:
    algorithm: str
    radius: float
    centers: list

@dataclass
class Snapshot:
    points: list[tuple]
    solutions: list[Solution]


def simulate(streaming_algos: list[str, object], offline_algos: list[str, object], points: list[object], snapshots_count: int, d: Callable, k: int) -> list[Snapshot]:

    snapshots: list[Snapshot] = []
    inserted_points: list[object] = []

    step = max(1, len(points) // snapshots_count)

    for i, point in enumerate(points):

        for name, algo in streaming_algos:
            algo.insert(point)

        inserted_points.append(point)

        if i > k and i % step == 0:

            solutions: list[Solution] = []

            for name, algo in streaming_algos:
                r, centers = algo.query()
                solutions.append(Solution(name, r, list(centers)))

            for name, algo in offline_algos:
                r, centers = algo(k, d, inserted_points)
                solutions.append(Solution(name, r, list(centers)))

            snapshots.append(Snapshot(list(inserted_points), list(solutions)))

    return snapshots


def plot(snapshots: list[Snapshot]) -> None:
    n = len(snapshots)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    # Farben für verschiedene Algorithmen
    colors = {
        "Doubling": "tomato",
        "Gonzalez": "green",
        "Parallized": "orange"
    }

    for i, snap in enumerate(snapshots):
        ax = axes[i]

        # Punkte plotten
        if snap.points:
            px, py = zip(*snap.points)
            ax.scatter(px, py, s=15, color="steelblue", alpha=0.4, label="Punkte")

        # Alle Lösungen plotten
        legend_entries = []
        for solution in snap.solutions:
            color = colors.get(solution.algorithm, "gray")
            
            # Center als Stern
            for cx, cy in solution.centers:
                ax.scatter(cx, cy, s=80, color=color, marker="*", zorder=5)
                # Kreis um Center
                circle = plt.Circle((cx, cy), solution.radius, color=color, 
                                   fill=False, linestyle="--", alpha=0.6)
                ax.add_patch(circle)
            
            # Legende vorbereiten (nur ein Eintrag pro Algorithmus)
            if solution.algorithm not in legend_entries:
                ax.scatter([], [], color=color, marker="*", 
                          label=f"{solution.algorithm} r={solution.radius:.2f}")
                legend_entries.append(solution.algorithm)

        ax.set_title(f"Snapshot {i+1} | k={len(snap.solutions[0].centers) if snap.solutions else 0}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Unsichtbare Achsen verstecken
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()