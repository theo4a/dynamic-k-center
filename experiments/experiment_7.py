import json
import os

from matplotlib import pyplot as plt
import numpy as np

from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import load_dataset, write_json, euclidean_distance



def experiment_7() -> None:
    """
    Using geopoints of households in hamburg as an input
    """
    
    k = 10
    d = euclidean_distance
    points = load_dataset()
    psa_m = 16

    da = DoublingKCenter(k=k, d=d)
    rda = RandomizedDoublingKCenter(k=k, d=d)
    psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)

    results = {}

    for i in range(len(points)):

        print(i)

        da.insert(points[i])
        rda.insert(points[i])
        psa.insert(points[i])

        if (i+1) % (len(points) // 10) == 0:

            results[str(i)] = {}
            
            da_solution = da.query()
            results[str(i)]["da_centers"] = da_solution["centers"]
            results[str(i)]["da_radius"] = da_solution["radius"]

            rda_solution = rda.query()
            results[str(i)]["rda_centers"] = rda_solution["centers"]
            results[str(i)]["rda_radius"] = rda_solution["radius"]

            psa_solution = psa.query()
            results[str(i)]["psa_centers"] = psa_solution["centers"]
            results[str(i)]["psa_radius"] = psa_solution["radius"]

    data = {
        "info": "",
        "results": results
    }

    write_json("experiment_7", data)


def plot_2_experiment_7() -> None:

    points = load_dataset()
    
    n_samples = 10000
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(points), size=n_samples, replace=False)
    sample = np.array(points)[indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)


    ax.set_aspect("equal")
    ax.set_title("OSM Building Centroids")
    ax.set_xlabel("Easting (m) – UTM32N / EPSG:25832")
    ax.set_ylabel("Northing (m) – UTM32N / EPSG:25832")

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_2_experiment_7.jpg")

    plt.savefig(plot_file_path)


def plot_1_experiment_7() -> None:

    points = load_dataset()
    

    n_samples = 10000
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(points), size=n_samples, replace=False)
    sample = np.array(points)[indices]

    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_7.json")

    with open(file_path, "r") as f:
        data = json.load(f)

    centers = np.array(data["results"]["779049"]["da_centers"])  # Liste von Listen -> NumPy-Array
    radius = data["results"]["779049"]["da_radius"]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)

    # Zentren als rote Punkte
    ax.scatter(centers[:, 0], centers[:, 1], s=20, color="red", zorder=3, label="Centers")

    # Kreise um die Zentren mit gegebenem Radius
    for cx, cy in centers:
        circle = plt.Circle((cx, cy), radius, color="red", fill=False, linewidth=1, alpha=0.7)
        ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_title("OSM Residential Building Centroids")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_1_experiment_7.jpg")

    plt.savefig(plot_file_path)