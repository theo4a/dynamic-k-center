import json
import os
import statistics
import time

from matplotlib import pyplot as plt

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from utils import euclidean_distance, generate_uniform_points, get_subplot, hsva_to_rgba, simulate_streaming, write_json



def experiment_runtime_n() -> None:
    """
    Using different ks when generating the points
    """

    max_n = 200
    k = 10
    d = euclidean_distance
    psa_m = 64

    results = {
        "sec_GO": [],

        "sec_DA": [],

        "sec_RDA": [],

        "sec_PSA": [],
    }

    for i in range(max_n - (k+1)):
        print(i)

        points = generate_uniform_points(
            n=i+(k+1),
            min=-100,
            max=100,
            dim=2,
            seed=i
        )

        # Gonzalez
        go_start = time.perf_counter()
        for i in range(len(points)- (k+1)):
            gonzalez(k, d, points[0:i+(k+2)])
        go_end = time.perf_counter()
        results["sec_GO"].append(go_end - go_start)

        # DA
        da = DoublingKCenter(k=k, d=d)
        da_start = time.perf_counter()
        simulate_streaming(da, points)
        da_end = time.perf_counter()
        results["sec_DA"].append(da_end - da_start)

        # RDA
        rda = RandomizedDoublingKCenter(k=k, d=d)
        rda_start = time.perf_counter()
        simulate_streaming(rda, points)
        rda_end = time.perf_counter()
        results["sec_RDA"].append(rda_end - rda_start)

        # PSA
        psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
        psa_start = time.perf_counter()
        simulate_streaming(psa, points)
        psa_end = time.perf_counter()
        results["sec_PSA"].append(psa_end - psa_start)

    stats = {
        "mean(sec_GO)": statistics.mean(results["sec_GO"]),
        "mean(sec_DA)": statistics.mean(results["sec_DA"]),
        "mean(sec_RDA)": statistics.mean(results["sec_RDA"]),
        "mean(sec_PSA)": statistics.mean(results["sec_PSA"]),
    }

    data = {
        "info": "",
        "results": results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_runtime_n", data)


def plot_experiment_runtime_n() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_runtime_n.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    results = data["results"]

    k = 10

    # Anzahl der Messpunkte
    num_values = len(next(iter(results.values())))

    # Entspricht n = k+1, k+2, ..., 200
    x = list(range(k + 1, k + 1 + num_values))

    fig, ax = get_subplot(figsize=(20, 10))

    labels = {
        "sec_GO": "GO",
        "sec_DA": "DA",
        "sec_RDA": "RDA",
        "sec_PSA": "PSA-64",
    }

    colors = {
        "sec_GO": hsva_to_rgba(60, 1, 0.8, 1),

        "sec_PSA": hsva_to_rgba(120, 1, 0.8, 1),

        "sec_RDA": hsva_to_rgba(210, 1, 0.8, 1),

        "sec_DA": hsva_to_rgba(360, 1, 0.8, 1),
    }

    for key, values in results.items():
        ax.plot(x, values, label=labels.get(key), color=colors.get(key))

    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Laufzeit in Sekunden")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_experiment_runtime_n.jpg")
    plt.savefig(plot_file_path, dpi=200)
    plt.close(fig)

