import json
import os

from matplotlib import pyplot as plt
import numpy as np

from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import get_subplot, hsva_to_rgba, load_dataset, write_json, euclidean_distance



def experiment_nrw_geo_points() -> None:
    """
    Using geopoints of households in hamburg as an input
    """
    
    k = 16
    d = euclidean_distance
    points = load_dataset()
    psa_m = 64

    da = DoublingKCenter(k=k, d=d)
    rda = RandomizedDoublingKCenter(k=k, d=d)
    psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)

    results = {}

    snapshot_count = 10000
    bias = 2

    snapshot_indices = (np.linspace(0, 1, snapshot_count) ** bias * (len(points) - 1)).astype(int)

    for i in range(len(points)):

        print(i)

        da.insert(points[i])
        rda.insert(points[i])
        psa.insert(points[i])

        if i > k and i in snapshot_indices:

            results[str(i)] = {}
            
            da_solution = da.query()
            results[str(i)]["r_prime_DA"] = da_solution["radius"]

            rda_solution = rda.query()
            results[str(i)]["r_prime_RDA"] = rda_solution["radius"]

            psa_solution = psa.query()
            results[str(i)]["r_prime_PSA"] = psa_solution["radius"]

    data = {
        "info": "",
        "results": results
    }

    write_json("experiment_nrw_geo_points", data)


def plot_experiment_nrw_geo_points() -> None:

    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_nrw_geo_points.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data["results"]

    # x-Werte
    x = sorted([int(k) for k in results.keys()])

    # y-Werte für jede Linie
    a_werte = [results[str(k)]["r_prime_DA"] for k in x]
    b_werte = [results[str(k)]["r_prime_RDA"] for k in x]
    c_werte = [results[str(k)]["r_prime_PSA"] for k in x]

    fig, ax = get_subplot()

    # Linien plotten
    ax.plot(x, a_werte, label="DA", color=hsva_to_rgba(360, 1, 0.8, 1))
    ax.plot(x, b_werte, label="RDA", color=hsva_to_rgba(210, 1, 0.8, 1))
    ax.plot(x, c_werte, label="PSA-64", color=hsva_to_rgba(120, 1, 0.8, 1))

    ax.set_xscale("log", base=2)

    # Beschriftung
    ax.set_xlabel("verarbeitete Punkte")
    ax.set_ylabel(r"$r'$")

    # Legende anzeigen
    ax.legend()

    # Grid optional
    ax.grid(True)

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_experiment_nrw_geo_points.jpg")
    fig.savefig(plot_file_path, dpi=200)
    plt.close(fig)
