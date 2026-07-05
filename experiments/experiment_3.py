import json
import os
import statistics
import time

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import chebyshev_distance, check_radius, euclidean_distance, manhattan_distance, simulate_streaming, write_json, boxplot, generate_gauß_clusters


def experiment_3() -> None:
    """
    Using different metrics when executing the algorithms
    """

    metrics = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "chebyshev": chebyshev_distance
    }
    k = 10
    n = 500
    psa_m = 16

    results: dict[str, dict[str, list[float]]] = {}

    for metric_name, d in metrics.items():
        
        results[metric_name] = {
            "gonzalez_sec": [],
            "gonzalez_r": [],

            "da_sec": [],
            "da_r": [],
            "da_c_r": [],

            "rda_sec": [],
            "rda_r": [],
            "rda_c_r": [],

            "psa_sec": [],
            "psa_r": [],
            "psa_c_r": [],
        }

        for i in range(0, 100):
            print(i)
            points = generate_gauß_clusters(
                k=k,
                n=n,
                cluster_std=1,
                dim=2,
                center_std=10,
                seed=i
            )

            # Gonzalez
            gonzalez_start = time.perf_counter()
            gonzalez_solution = gonzalez(k, d, points)
            gonzalez_end = time.perf_counter()
            results[str(metric_name)]["gonzalez_sec"].append(gonzalez_end - gonzalez_start)
            results[str(metric_name)]["gonzalez_r"].append(gonzalez_solution["radius"])

            # DA
            da = DoublingKCenter(k=k, d=d)
            da_start = time.perf_counter()
            da_solution = simulate_streaming(da, points)
            da_end = time.perf_counter()
            da_c_r = check_radius(d, points, da_solution["centers"])
            results[str(metric_name)]["da_sec"].append(da_end - da_start)
            results[str(metric_name)]["da_r"].append(da_solution["radius"])
            results[str(metric_name)]["da_c_r"].append(da_c_r)

            # RDA
            rda = DoublingKCenter(k=k, d=d)
            rda_start = time.perf_counter()
            rda_solution = simulate_streaming(rda, points)
            rda_end = time.perf_counter()
            rda_c_r = check_radius(d, points, rda_solution["centers"])
            results[str(metric_name)]["rda_sec"].append(rda_end - rda_start)
            results[str(metric_name)]["rda_r"].append(rda_solution["radius"])
            results[str(metric_name)]["rda_c_r"].append(rda_c_r)

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(metric_name)]["psa_sec"].append(psa_end - psa_start)
            results[str(metric_name)]["psa_r"].append(psa_solution["radius"])
            results[str(metric_name)]["psa_c_r"].append(psa_c_r)

    stats = []

    for metric, value in results.items():
        gonzalez_r_mean = statistics.mean(value["gonzalez_r"])
        stats.append({
            "metric": metric,
            "gonzalez_r_mean": gonzalez_r_mean,
            "da_r_mean/gonzalez_r_mean": statistics.mean(value["da_r"]) / gonzalez_r_mean,
            "psa_r_mean/gonzalez_r_mean": statistics.mean(value["psa_r"]) / gonzalez_r_mean,

        })

    data = {
        "info": "",
        "results": results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_3", data)


def plot_1_experiment_3() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_3.json")

    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        "Gonzalez r": "gonzalez_r",

        "DA r": "da_r",
        "DA r'": "da_c_r",

        "PSA-16 r": "psa_r",
        "PSA-16 r'": "psa_c_r",
    }

    plt = boxplot(
        data=data["results"],
        x_label="Metric",
        y_label="",
        algorithms=algorithms
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_1_experiment_3.jpg")

    plt.savefig(plot_file_path)