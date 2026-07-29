import json
import os
import statistics
import time

from matplotlib import pyplot as plt

from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import euclidean_distance, generate_uniform_points, hsva_to_rgba, simulate_streaming, write_json, boxplot



def experiment_runtime_m() -> None:
    """
    Using different ks when generating the points
    """

    n = 100
    d = euclidean_distance
    k = 10
    psa_ms = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    results = {}

    for psa_m in psa_ms:

        results[str(psa_m)] = {
            "sec_PSA": []
        }

        for i in range(100):

            print(i)

            points = generate_uniform_points(
                n=n,
                min=-100,
                max=100,
                dim=2,
                seed=i
            )

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            results[str(psa_m)]["sec_PSA"].append(psa_end - psa_start)

    stats = []

    for psa_m, value in results.items():
        stats.append({
            "psa_m": psa_m,
            "mean(sec_PSA)": statistics.mean(value["sec_PSA"])
        })

    data = {
        "info": "",
        "results": results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_runtime_m", data)


def plot_experiment_runtime_m() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_runtime_m.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        r"$PSA$": "sec_PSA"
    }

    colors = {
        r"$PSA$": hsva_to_rgba(120, 1, 0.8, 1)
    }

    fig, ax = boxplot(
        data=data["results"],
        x_label=r"$m$",
        y_label="",
        algorithms=algorithms,
        colors=colors
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_experiment_runtime_m.jpg")
    plt.savefig(plot_file_path, dpi=200)
    plt.close(fig)


