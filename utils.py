import colorsys
import csv
import json
import math
import os
from typing import Callable
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
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

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.streaming_k_center_interface import StreamingKCenter
from config import FILE_PATH_1


# ----------------------
# Colors
# ----------------------

def hsva_to_rgba(h, s, v, a):
    # h: 0-360, s/v/a: 0-1
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return (
        r,
        g,
        b,
        a
    )



# ----------------------
# Plot Expamples
# ----------------------

def plot_example_cluster_separation_comparison() -> None:
    
    fig, axes = get_subplot(2, 3, (15, 10))
    axes = axes.flatten()

    cluster_sds = [0.25, 0.5, 1, 2, 4, 8]

    for ax, cluster_sd in zip(axes, cluster_sds):

        points = generate_clustered_points(
            k=10,
            n=500,
            cluster_std=cluster_sd,
            dim=2,
            center_std=10,
            seed=0
        )

        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            s=8,
            alpha=0.8
        )

        ax.set_title(rf"$\sigma_{{cluster}} = {cluster_sd}$")
        ax.set_aspect("equal")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.grid(True)

    plot_file_path = os.path.join(
        os.path.dirname(__file__),
        "plots",
        "plot_example_cluster_separation_comparison.jpg",
    )

    plt.savefig(plot_file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_example_k_cluster_comparison() -> None:
    
    fig, axes = get_subplot(2, 3, (15, 10))
    axes = axes.flatten()

    ks = [2, 4, 8, 16, 32, 64]

    for ax, k in zip(axes, ks):

        points = generate_clustered_points(
            k=k,
            n=500,
            cluster_std=1,
            dim=2,
            center_std=2*math.sqrt(k),
            seed=0
        )

        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            s=8,
            alpha=0.8
        )

        ax.set_title(rf"$k = {k}$")
        ax.set_aspect("equal")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.grid(True)

    plot_file_path = os.path.join(
        os.path.dirname(__file__),
        "plots",
        "plot_example_k_cluster_comparison.jpg",
    )

    plt.savefig(plot_file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_example_nrw_geo_points() -> None:

    points = load_dataset()
    
    n_samples = 50000
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(points), size=n_samples, replace=False)
    sample = np.array(points)[indices]

    
    fig, ax = get_subplot(figsize=(11, 10))

    # Plot
    ax.scatter(sample[:, 0], sample[:, 1], s=1, color=hsva_to_rgba(0, 0, 0.5, 0.5))

    ax.set_aspect('equal')
    ax.set_xlabel("Easting (m) - UTM32N / EPSG:25832")
    ax.set_ylabel("Northing (m) - UTM32N / EPSG:25832")

    plot_file_path = os.path.join(os.path.dirname(__file__), "plots", "plot_example_nrw_geo_points.jpg")
    plt.savefig(plot_file_path, dpi=200)
    plt.close(fig)

def plot_example_da_step_by_step() -> None:
        
    def plot_state(pointss, counter, title, r_prime, r_prime_label,
                   r=None, r_label=None):

        fig, ax = get_subplot(figsize=(18, 10))

        keys = list(pointss.keys())

        for i, key in enumerate(keys):
            point = pointss[key]
            x, y = point["position"]
            ptype = point["type"]
            label = point["label"]

            # Letztes hinzugefügtes Element hervorheben
            markersize = 20 if i == len(keys) - 1 else 10
            color = "red" if ptype == "center" else "blue"

            # Punkte erstellen
            ax.plot(
                x, y,
                marker="o",
                color=color,
                markersize=markersize,
                alpha=0.5
            )

            # Punktname erstellen
            ax.text(
                x + 0.25,
                y - 0.25,
                label,
                color = color,
                fontsize=20,
                ha="center",
                va="top",
                alpha=1
            )

            # Kreise erstellen
            if ptype == "center":
                # Zentrum zeichnen
                ax.plot(
                    x, y,
                    marker="o",
                    color="red",
                    markersize=markersize,
                    alpha=0.5
                )

                # r'-Kreis
                circle = plt.Circle(
                    (x, y),
                    4 * r_prime,
                    color="red",
                    fill=False,
                    linewidth=2.5,
                    linestyle="--",
                    alpha=0.5
                )
                ax.add_patch(circle)

                # optionaler r-Kreis
                if r is not None:
                    circle = plt.Circle(
                        (x, y),
                        r,
                        color="red",
                        fill=False,
                        linewidth=2.5,
                        alpha=0.5
                    )
                    ax.add_patch(circle)

        ax.set_xlim(-9, 23)
        ax.set_ylim(-9, 9)

        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(title)

        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                alpha=0.5,
                label="Centers"
            ),
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                alpha=0.5,
                label="Points"
            ),
            plt.Line2D(
                [0], [0],
                color="red",
                linestyle="--",
                alpha=0.5,
                label=r_prime_label
            ),
        ]

        if r is not None:
            legend_elements.append(
                plt.Line2D(
                    [0], [0],
                    color="red",
                    alpha=0.5,
                    label=r_label
                )
            )

        ax.legend(handles=legend_elements, loc="upper right")

        file_path = os.path.join(
            os.path.dirname(__file__),
            "plots",
            f"plot_example_da_step_by_step_{counter}.jpg"
        )

        fig.savefig(file_path, dpi=200)
        plt.close(fig)
    
    pointss: dict = {}
    points: list = []
    centers: list = []
    k: int = 3
    phase: int = 0
    r: float = 0
    
    plot_state(
        pointss=pointss,
        counter=0,
        r_prime=r,
        r_prime_label=r"$r' = 0$",
        title=r"$k = 3, \quad C = \emptyset, \quad r_0 = 0, \quad phase = 0$"
    )

    pointss["p_1"] = {
        "position": (0,0),
        "type": "center",
        "label": r"$p_1$"
    }
    
    plot_state(
        pointss=pointss,
        counter=1,
        r_prime=r,
        r_prime_label=r"$r' = 0$",
        title=r"$k = 3, \quad C = \{p_1\}, \quad r_0 = 0, \quad phase = 0$"
    )

    pointss["p_2"] = {
        "position": (1,0),
        "type": "center",
        "label": r"$p_2$"
    }
    
    plot_state(
        pointss=pointss,
        counter=2,
        r_prime=r,
        r_prime_label=r"$r' = 0$",
        title=r"$k = 3, \quad C = \{p_1, p_2\}, \quad r_0 = 0, \quad phase = 0$"
    )
    
    pointss["p_3"] = {
        "position": (4,0),
        "type": "center",
        "label": r"$p_3$"
    }

    plot_state(
        pointss=pointss,
        counter=3,
        r_prime=r,
        r_prime_label=r"$r' = 0$",
        title=r"$k = 3, \quad C = \{p_1, p_2, p_3\}, \quad r_0 = 0, \quad phase = 0$"
    )
    
    pointss["p_4"] = {
        "position": (6,0),
        "type": "center",
        "label": r"$p_4$"
    }
    
    plot_state(
        pointss=pointss,
        counter=4,
        r_prime=r,
        r_prime_label=r"$r' = 0$",
        title=r"$k = 3, \quad C = \{p_1, p_2, p_3, p_4\}, \quad r_0 = 0, \quad phase = 0$"
    )
    
    r = 0.5
    
    plot_state(
        pointss=pointss,
        counter=5,
        r_prime=r,
        r_prime_label=r"$r' = 2$",
        title=r"$k = 3, \quad C = \{p_1, p_2, p_3, p_4\, \quad r_1 = 0.5, \quad phase = 0$"
    )
    
    phase += 1
    
    plot_state(
        pointss=pointss,
        counter=6,
        r_prime=r,
        r_prime_label=r"$r' = 2$",
        title=r"$k = 3, \quad C = \{p_1, p_2, p_3, p_4\}, \quad r_1 = 0.5, \quad phase = 1$"
    )
    
    r *= 2
    
    plot_state(
        pointss=pointss,
        counter=7,
        r_prime=r,
        r_prime_label=r"$r' = 4$",
        title=r"$k = 3, \quad C = \{p_1, p_2, p_3, p_4\}, \quad r_2 = 1, \quad phase = 1$"
    )
    
    pointss["p_2"] = {
        "position": (1,0),
        "type": "point",
        "label": r"$p_2$"
    }
    pointss["p_4"] = {
        "position": (6,0),
        "type": "point",
        "label": r"$p_4$"
    }
    
    plot_state(
        pointss=pointss,
        counter=8,
        r_prime=r,
        r_prime_label=r"$r' = 4$",
        title=r"$k = 3, \quad C = \{p_1, p_3\}, \quad r_2 = 1, \quad phase = 1$"
    )

    pointss["p_5"] = {
        "position": (7,0),
        "type": "point",
        "label": r"$p_5$"
    }
    
    plot_state(
        pointss=pointss,
        counter=9,
        r_prime=r,
        r_prime_label=r"$r' = 4$",
        title=r"$k = 3, \quad C = \{p_1, p_3\}, \quad r_2 = 1, \quad phase = 1$"
    )

    pointss["p_6"] = {
        "position": (9,0),
        "type": "center",
        "label": r"$p_6$"
    }
    
    plot_state(
        pointss=pointss,
        counter=10,
        r_prime=r,
        r_prime_label=r"$r' = 4$",
        title=r"$k = 3, \quad C = \{p_1, p_3, p_6\}, \quad r_2 = 1, \quad phase = 1$"
    )
    
    pointss["p_7"] = {
        "position": (14,0),
        "type": "center",
        "label": r"$p_7$"
    }
    
    plot_state(
        pointss=pointss,
        counter=11,
        r_prime=r,
        r_prime_label=r"$r' = 4$",
        title=r"$k = 3, \quad C = \{p_1, p_3, p_6, p_7\}, \quad r_2 = 1, \quad phase = 1$"
    )
    
    phase += 1
    
    plot_state(
        pointss=pointss,
        counter=12,
        r_prime=r,
        r_prime_label=r"$r' = 4$",
        title=r"$k = 3, \quad C = \{p_1, p_3, p_6, p_7\}, \quad r_2 = 1, \quad phase = 2$"
    )
    
    r *= 2
    
    plot_state(
        pointss=pointss,
        counter=13,
        r_prime=r,
        r_prime_label=r"$r' = 8$",
        title=r"$k = 3, \quad C = \{p_1, p_3, p_6, p_7\}, \quad r_3 = 2, \quad phase = 2$"
    )
    
    pointss["p_3"] = {
        "position": (4,0),
        "type": "point",
        "label": r"$p_3$"
    }
    pointss["p_7"] = {
        "position": (14,0),
        "type": "point",
        "label": r"$p_7$"
    }
    
    plot_state(
        pointss=pointss,
        counter=14,
        r_prime=r,
        r_prime_label=r"$r' = 8$",
        title=r"$k = 3, \quad k = 3, \quad C = \{p_1, p_6\}, \quad r_3 = 2, \quad phase = 2$"
    )
    
    plot_state(
        pointss=pointss,
        counter=15,
        r_prime=r,
        r_prime_label=r"$r' = 8$",
        title=r"$k = 3, \quad C = \{p_1, p_6\}, \quad r_3 = 2, \quad phase = 2$",
        r=5,
        r_label=r"$R = 5$"
    )

def plot_example_gonzalez_worst_case() -> None:
    
    k = 10
    d = euclidean_distance

    cluster_origins = [
        (-4.5,  3.5),
        (-3.5, -1.5),
        (-3.0,  1.0),
        (-1.0, -4.0),
        ( -0.5,  2.0),
        ( 0, -0.5),
        ( 2.5,  4.5),
        ( 2.0,  0.5),
        ( 4.0, -3.0),
        ( 4.5,  3.0),
    ]

    n = 200
    radius = 1

    points = []

    for c in cluster_origins:
        # Gleichverteilte Punkte im Kreis
        theta = np.random.uniform(0, 2 * np.pi, n)
        r = radius * np.sqrt(np.random.uniform(0, 1, n))

        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)

        # Einzelne Koordinaten als Tuple speichern
        points.extend(zip(x, y))

    go_solution = gonzalez(k, d, points)
    go_centers = go_solution["centers"]
    go_radius = go_solution["radius"]

    fig, ax = get_subplot(figsize=(10, 10))

    # Punktwolke
    ax.scatter(
        [p[0] for p in points],
        [p[1] for p in points],
        s=1,
        alpha=0.5
    )

    for i, c in enumerate(go_centers):

        ax.scatter(
            c[0],
            c[1],
            color=hsva_to_rgba(0, 1, 0.5, 1),
            s=40
        )

        ax.text(
            c[0] + 0.05,
            c[1] + 0.05,
            rf"$c_{i}$",
            fontsize=32,
            color=hsva_to_rgba(0, 1, 0.5, 1)
        )

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    file_path = os.path.join(os.path.dirname(__file__), "plots", "plot_example_gonzalez_worst_case.jpg")

    plt.savefig(file_path, dpi=200)

def plot_example_clustered_points_k() -> None:
    
    cluster_origins = [
        (-4.5,  3.5),
        (-3.5, -1.5),
        (-2.0,  1.0),
        (-1.0, -4.0),
        ( 0.5,  3.0),
        ( 0, -0.5),
        ( 2.5,  4.5),
        ( 3.0,  0.5),
        ( 4.0, -3.0),
        ( 4.5,  3.0),
    ]

    radius = 1
    n_points = 200

    points = []
    for c in cluster_origins:
        # Gleichverteilte Punkte im Kreis
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        r = radius * np.sqrt(np.random.uniform(0, 1, n_points))

        x = c[0] + r * np.cos(theta)
        y = c[1] + r * np.sin(theta)
        points.append((x, y))

    centers = []
    offset_range = 0.25
    for i, c in enumerate(cluster_origins):

        # Erzeuge ein schlecht gewähltes Zentrum
        if i == 8:
            center = (c[0] + radius, c[1])
            centers.append(center)
            continue

        offset = np.random.uniform(-offset_range, offset_range, size=2)
        center = np.array(c) + offset
        centers.append(tuple(center))

    fig, ax = get_subplot(figsize=(10, 10))

    ax.scatter(
        [p[0] for p in points],
        [p[1] for p in points],
        color=hsva_to_rgba(0, 0, 0.75, 0.5),
        s=10
    )

    for i, c in enumerate(centers):

        ax.scatter(
            c[0],
            c[1],
            color=hsva_to_rgba(0, 1, 0.5, 1),
            s=20
        )

        ax.text(
            c[0] + 0.05,
            c[1] + 0.05,
            rf"$c_{i}$",
            fontsize=30,
            color=hsva_to_rgba(0, 1, 0.5, 1)
        )

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    file_path = os.path.join(os.path.dirname(__file__), "plots", "plot_example_clustered_points_k.jpg")

    plt.savefig(file_path, dpi=200)

def plot_example_cluster_separation() -> None:

    np.random.seed(4)

    base_cluster_origins = [
        (-4.5,  3.5),
        (-3.5, -1.5),
        (-2.0,  1.0),
        (-1.0, -4.0),
        ( 0.5,  3.0),
        ( 0.0, -0.5),
        ( 2.5,  4.5),
        ( 3.0,  0.5),
        ( 4.0, -3.0),
        ( 4.5,  3.0),
    ]

    cluster_origins_1 = [
        (x * 0.5, y * 0.5)
        for x, y in base_cluster_origins
    ]

    cluster_origins_2 = [
        (x * 1.5, y * 1.5)
        for x, y in base_cluster_origins
    ]

    cluster_configurations = [
        ("Geringe Cluster Separation", cluster_origins_1),
        ("Hohe Cluster Separation", cluster_origins_2),
    ]

    radius = 1
    points_per_cluster = 200

    # Gleiche zufällige Verschiebungen für beide Instanzen
    center_offset_range = 0.75
    center_offsets = [
        np.random.uniform(
            -center_offset_range,
            center_offset_range,
            size=2
        )
        for _ in range(len(base_cluster_origins) + 1)
    ]

    fig, ax = get_subplot(1, 2, figsize=(20, 10))

    for axis, (title, cluster_origins) in zip(
        ax,
        cluster_configurations
    ):
        # Punkte innerhalb der Cluster erzeugen
        points = []

        for origin in cluster_origins:
            angles = np.random.uniform(
                0,
                2 * np.pi,
                points_per_cluster
            )

            distances = radius * np.sqrt(
                np.random.uniform(0, 1, points_per_cluster)
            )

            x_points = origin[0] + distances * np.cos(angles)
            y_points = origin[1] + distances * np.sin(angles)

            points.extend(zip(x_points, y_points))

        # Cluster-Zentren mit gleichem Offset erzeugen
        centers = [
            tuple(np.array(origin) + offset)
            for origin, offset in zip(
                cluster_origins,
                center_offsets
            )
        ]

        # Zusätzliches Zentrum erzeugen
        centers.append(
            tuple(
                np.array(cluster_origins[0]) + center_offsets[-1]
            )
        )

        # Punkte plotten
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            color=hsva_to_rgba(0, 0, 0.75, 0.5),
            s=10
        )

        # Zentren plotten
        for index, center in enumerate(centers):
            axis.scatter(
                center[0],
                center[1],
                color=hsva_to_rgba(0, 1, 0.5, 1),
                s=40
            )

            axis.text(
                center[0] + 0.05,
                center[1] + 0.05,
                rf"$c_{{{index}}}$",
                fontsize=30,
                color=hsva_to_rgba(0, 1, 0.5, 1)
            )

        axis.set_title(title, fontsize=24)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    file_path = os.path.join(os.path.dirname(__file__), "plots", "plot_example_cluster_separation.jpg")

    fig.savefig(file_path, dpi=200)



# ----------------------
# Plot Functions
# ----------------------

def plot_function_psa_approximationfaktor() -> None:
    
    beta = np.linspace(1.05, 20, 500)
    m = np.linspace(1, 10, 500)
    
    B, M = np.meshgrid(beta, m)
    
    Z = (2 * B / (B - 1)) * (B ** (1 / M))
    
    # Wichtig: keine 0 oder negative Werte erlaubt!
    Z = np.clip(Z, 1e-6, None)
    
    fig, ax = get_subplot(figsize=(12, 10))

    heat = ax.pcolormesh(
        B, M, Z,
        shading='auto',
        cmap='viridis',
        norm=LogNorm(vmin=Z.min(), vmax=Z.max())
    )
    
    fig.colorbar(heat, label=r"$\eta_p(\beta, m)$")
    
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$m$")
    
    file_path = os.path.join(os.path.dirname(__file__), "plots", "plot_function_psa_aproximationfaktor.jpg")
    fig.savefig(file_path, dpi=200)

def plot_function_psa_approximationfaktor_by_m() -> None:
    def f(m):
        return (2 * (m + 1)) / m * (m + 1)**(1/m)

    # m-Werte definieren (m darf nicht 0 sein, da sonst Division durch 0)
    m = np.linspace(0.1, 256, 1000)
    y = f(m)

    fig, ax = get_subplot(figsize=(12, 10))

    ax.plot(m, y, label=r'$\eta_p(m, m+1) = \frac{2(m+1)^2}{m}\,(m+1)^{1/m}$', color='blue')
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\eta_p(m, m+1)$")
    ax.set_yscale("log", base=10)
    ax.legend()
    ax.grid(True)

    file_path = os.path.join(os.path.dirname(__file__), "plots", "plot_function_psa_approximationfaktor_by_m.jpg")

    fig.savefig(file_path, dpi=200)
    plt.close(fig)



# ----------------------
# Datasets
# ----------------------

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



# ----------------------
# Plots
# ----------------------

def get_subplot(n_rows: int = 1, n_cols: int = 1, figsize: tuple[float, float] = (20, 10)) -> tuple[Figure, Axes]:
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 20
    })

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

    fig.tight_layout(pad=2)

    return fig, ax

def boxplot(data: dict,
            x_label: str,
            y_label: str,
            algorithms: dict,
            colors: dict = None,
    ) -> tuple[Figure, Axes]:

    # Cluster sortieren
    try:
        cluster_order = sorted(data.keys(), key=lambda x: float(x))
    except:
        cluster_order = sorted(data.keys(), key=lambda x: str(x))

    # Default-Farben, falls nicht übergeben
    default_colors = plt.get_cmap("tab10")
    #colors = {
    #    name: default_colors(i % 10)
    #    for i, name in enumerate(algorithms.keys())
    #}

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
    for _, (key, value) in enumerate(algorithms.items()):

        box_data = []
        positions = positions_map[key]
        color = colors[key]

        for cluster in cluster_order:
            values = data[cluster].get(value, [])
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



# ----------------------
# Metrics
# ----------------------

def euclidean_distance(a: object, b: object) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def manhattan_distance(a: object, b: object) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def chebyshev_distance(a: object, b: object) -> float:
    return max(abs(x - y) for x, y in zip(a, b))



# ----------------------
# Experiments
# ----------------------

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



# ----------------------
# I/O
# ----------------------

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



# ----------------------
# Point generators
# ----------------------

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

