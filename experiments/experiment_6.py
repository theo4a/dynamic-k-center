import json
import os
import statistics
import time

from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import check_radius, euclidean_distance, simulate_streaming, write_json, boxplot, generate_gauß_clusters


def experiment_6() -> None:
    """
    Using different m's for PSA on points with a cluster standard diviation of 1
    """

    n = 500
    k = 10
    sample_size = 100
    psa_ms = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    d = euclidean_distance

    results: dict[str, dict[str, list[float]]] = {}

    for m in psa_ms:
        results[str(m)] = {
            "psa_sec": [],
            "psa_r": [],
            "psa_c_r": []
        }

        for i in range(sample_size):
            print(i)

            points = generate_gauß_clusters(
                k=k,
                n=n,
                cluster_std=1,
                dim=2,
                center_std=10,
                seed=i
            )

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(m)]["psa_sec"].append(psa_end - psa_start)
            results[str(m)]["psa_r"].append(psa_solution["radius"])
            results[str(m)]["psa_c_r"].append(psa_c_r)

    stats = []

    for m, value in results.items():
        stats.append({
            "m": m,
            "psa_r_mean": statistics.mean(value["psa_r"]),
            "psa_c_r_mean": statistics.mean(value["psa_c_r"]),

        })

    data = {
        "info": "",
        "results" : results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_6", data)


def plot_1_experiment_6() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_6.json")

    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        "PSA r": "psa_r",
        "PSA r'": "psa_c_r",
    }

    plt = boxplot(
        data=data["results"],
        x_label="m",
        y_label="",
        algorithms=algorithms
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_1_experiment_6.jpg")

    plt.savefig(plot_file_path)
    