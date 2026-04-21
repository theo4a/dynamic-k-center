import random
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from hochbaum_shmoys import hochbaum_shmoys
from verbessertes_doubling import DoublingClustering
from generators import cluster_by_cluster, cluster_by_cluster_fixed_radius, evenly_growing_fixed_radius, evenly_growing_k_cluster
from gonzalez import gonzalez

def euclidean_distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@dataclass
class Snapshot:
    radius: float
    centers: list[tuple]
    points: list[tuple]
    gonzalez_radius: float = 0.0
    gonzalez_centers: list[tuple] = None



def simulate(algo: object, points: list[object], snapshots_count: int, d: Callable, k: int) -> list[Snapshot]:

    snapshots: list[Snapshot] = []
    inserted_points: list[object] = []

    step = max(1, len(points) // snapshots_count)

    for i, point in enumerate(points):

        algo.insert(point)
        inserted_points.append(point)

        if i != 0 and i % step == 0:

            result = algo.query()

            if not result:
                continue

            r, centers = result

            g_r, g_centers = hochbaum_shmoys(inserted_points, d, k)

            snapshots.append(Snapshot(r, centers, list(inserted_points), g_r, g_centers))

    return snapshots


def plot(snapshots: list[Snapshot], algo_label: str = "Algo") -> None:

    n = len(snapshots)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, snap in enumerate(snapshots):
        ax = axes[i]

        if snap.points:
            px, py = zip(*snap.points)
            ax.scatter(px, py, s=15, color="steelblue", alpha=0.4, label="Punkte")

        # Algo-Lösung
        for cx, cy in snap.centers:
            ax.scatter(cx, cy, s=80, color="tomato", marker="*", zorder=5)
            circle = plt.Circle((cx, cy), snap.radius, color="tomato", fill=False, linestyle="--", alpha=0.6)
            ax.add_patch(circle)

        # Gonzalez-Lösung
        for cx, cy in snap.gonzalez_centers:
            ax.scatter(cx, cy, s=80, color="green", marker="*", zorder=5)
            circle = plt.Circle((cx, cy), snap.gonzalez_radius, color="green", fill=False, linestyle=":", alpha=0.6)
            ax.add_patch(circle)

        # Legende
        ax.scatter([], [], color="tomato", marker="*", label=f"{algo_label} r={snap.radius:.2f}")
        ax.scatter([], [], color="green",  marker="*", label=f"Gonzalez r={snap.gonzalez_radius:.2f}")

        ax.set_title(f"Snapshot {i+1} | k={len(snap.centers)}")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


k = 10
d = euclidean_distance
algo = DoublingClustering(k, d, 0.1)
points = cluster_by_cluster_fixed_radius(k, 1000, 10, 0)

snapshots = simulate(algo, points, 9, d, k)
plot(snapshots, algo_label="Dynamic k-center")