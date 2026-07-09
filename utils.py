import csv
import json
import math
import os
from typing import Callable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.ops import transform
from shapely import MultiPolygon, wkb as shapely_wkb

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import osmium
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer

from algorithms.online.streaming_k_center import StreamingKCenter
from config import FILE_PATH_1


# Datasets

def nrw_osm_data_to_metric_centroids_csv() -> None:

    # WGS84 -> UTM32N
    to_metric = Transformer.from_crs(
        4326,
        25832,
        always_xy=True
    )

    class HouseHandler(osmium.SimpleHandler):

        def __init__(self, writer):
            super().__init__()
            self.writer = writer
            self.wkbfab = osmium.geom.WKBFactory()

        def area(self, a):

            if a.tags.get("building") in {
                "house",
                "residential",
                "apartments",
                "detached",
                "semidetached_house",
                "terrace"
            }:

                try:

                    # OSM -> Shapely
                    wkb = self.wkbfab.create_multipolygon(a)
                    geom = shapely_wkb.loads(wkb, hex=True)

                    # MultiPolygon behandeln
                    if isinstance(geom, MultiPolygon):
                        geom = max(
                            geom.geoms,
                            key=lambda g: g.area
                        )

                    # Polygon in metrisches CRS transformieren
                    projected_geom = transform(
                        to_metric.transform,
                        geom
                    )

                    # centroid
                    centroid = projected_geom.centroid

                    # Speichern
                    self.writer.writerow([centroid.x, centroid.y])

                except Exception as e:
                    print(e)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        "nordrhein-westfalen-260703-building-centroids.csv"
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)
        writer.writerow(["Northing (m)", "Easting (m)"])

        handler = HouseHandler(writer)

        global FILE_PATH_1

        handler.apply_file(
            FILE_PATH_1,
            locations=True,
            idx="flex_mem"
        )

# Plots

def get_subplot() -> tuple[Figure, Axes]:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12
    })

    fig, ax = plt.subplots(figsize=(16, 8))

    fig.tight_layout()

    return fig, ax

def boxplot(data: dict,
            x_label: str,
            y_label: str,
            algorithms: dict,
    ) -> tuple[Figure, Axes]:

    # Cluster sortieren
    try:
        cluster_order = sorted(data.keys(), key=lambda x: float(x))
    except:
        cluster_order = sorted(data.keys(), key=lambda x: str(x))

    # Default-Farben, falls nicht übergeben
    default_colors = plt.get_cmap("tab10")
    colors = {
        name: default_colors(i % 10)
        for i, name in enumerate(algorithms.keys())
    }

    fig, ax = get_subplot()

    n = len(algorithms)
    base_positions = np.arange(len(cluster_order))

    # Breite & Abstand dynamisch je nach Anzahl Algorithmen
    total_width = 0.8  # Gesamtbreite pro Cluster-Gruppe (bleibt konstant)
    box_width = total_width / n
    # Offsets symmetrisch um 0 zentriert
    offsets = np.linspace(-(total_width - box_width) / 2, (total_width - box_width) / 2, n)

    positions_map = {
        name: base_positions + offsets[i]
        for i, name in enumerate(algorithms.keys())
    }

    # Plotten
    for _, (label, key) in enumerate(algorithms.items()):

        box_data = []
        positions = positions_map[label]
        color = colors[label]

        for cluster in cluster_order:
            values = data[cluster].get(key, [])
            box_data.append(values)

        ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width * 0.9,  # kleiner Abstand zwischen Boxen
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, color=color),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            medianprops=dict(color="black")
        )

    # X-Achse
    ax.set_xticks(base_positions, cluster_order)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Legend
    legend_handles = [
        Patch(facecolor=colors[name], label=name)
        for name in algorithms.keys()
    ]

    ax.legend(handles=legend_handles)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    return fig, ax

# Metrics

def euclidean_distance(a: object, b: object) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def manhattan_distance(a: object, b: object) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def chebyshev_distance(a: object, b: object) -> float:
    return max(abs(x - y) for x, y in zip(a, b))

# Experiments

def simulate_streaming(
    streaming_algo: StreamingKCenter,
    points: list[object]
) -> dict:

    for point in points:
        streaming_algo.insert(point)

    return streaming_algo.query()

def check_radius(
    d: Callable[[object, object], float],
    points: list[object],
    centers: list[object]
) -> float:

    max_dist = 0.0

    for p in points:

        # minimale Distanz von p zu einem Center
        min_dist = min(d(p, c) for c in centers)

        # worst-case (Radius)
        max_dist = max(max_dist, min_dist)

    return max_dist

# I/O

def write_json(file_name: str, dict: dict) -> None:

    # Zielpfad
    path = os.path.join(
        os.path.dirname(__file__),
        "results",
        "data",
        f"{file_name}.json"
    )

    # Ordner sicher erstellen
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # In Datei schreiben
    with open(path, "w", encoding="utf-8") as datei:
        json.dump(dict, datei, ensure_ascii=False, indent=4)

def load_dataset() -> list:
    
    "nordrhein-westfalen-260703-house-centroids.csv"

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "datasets", "nordrhein-westfalen-260703-building-centroids.csv"))
    return df[["Northing (m)", "Easting (m)"]].to_numpy().tolist()

# Point generators

def _select_farthest_centers(
    candidates: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    num_candidates = candidates.shape[0]
    first_idx = int(rng.integers(num_candidates))
    chosen_indices = [first_idx]
 
    # min_dist[i] = Abstand von candidates[i] zum nächstgelegenen bereits
    # gewählten Zentrum (initial: Abstand zum ersten Zentrum)
    min_dist = np.linalg.norm(candidates - candidates[first_idx], axis=1)
    min_dist[first_idx] = -1.0  # bereits gewählt -> für argmax sperren
 
    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dist))
        chosen_indices.append(next_idx)
        new_dist = np.linalg.norm(candidates - candidates[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[next_idx] = -1.0  # bereits gewählt -> für argmax sperren
 
    return candidates[chosen_indices]
 
def generate_clustered_points(
    k: int,
    n: int,
    cluster_std: float,
    dim: int,
    center_std: float,
    seed: int = None,
) -> list[object]:
 
    rng = np.random.default_rng(seed)
 
    # 1. Kandidatenzentren generieren und per Gonzalez-Prinzip auswählen
    num_candidates = 3 * k
    candidates = rng.normal(loc=0.0, scale=center_std, size=(num_candidates, dim))
    centers = _select_farthest_centers(candidates, k, rng)
 
    # 2. Punkte möglichst gleichmäßig auf die k Cluster verteilen
    base, remainder = divmod(n, k)
    sizes = [base + (1 if i < remainder else 0) for i in range(k)]
 
    points = np.empty((n, dim), dtype=float)
    offset = 0
    for center, size in zip(centers, sizes):
        points[offset: offset + size] = rng.normal(loc=center, scale=cluster_std, size=(size, dim))
        offset += size
 
    # 3. Reihenfolge durchmischen, damit nicht alle Punkte clusterweise sortiert sind
    points = points[rng.permutation(n)]
 
    return points.tolist()

def generate_uniform_points(min: float, max: float, dim: int, n: int, seed) -> list[object]:
    
    rng = np.random.default_rng(seed)
    
    return [
        [rng.uniform(min, max) for _ in range(dim)]
        for _ in range(n)
    ]

