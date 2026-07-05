import json
import math
import os
import statistics
import time

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from utils import check_radius, euclidean_distance, simulate_streaming, write_json, boxplot, generate_gauß_clusters


def experiment_2() -> None:
    """
    Using different ks when generating the points
    """

    n = 512
    sample_size = 100
    ks = [2, 4, 8, 16, 32, 64, 128]
    d = euclidean_distance
    psa_m = 16

    results: dict[str, dict[str, list[float]]] = {}

    for k in ks:

        results[str(k)] = {
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

        for i in range(sample_size):
            print(i)

            points = generate_gauß_clusters(
                k=k,
                n=n,
                cluster_std=1,
                dim=2,
                center_std=2 * math.sqrt(k),
                seed=i
            )

            # Gonzalez
            gonzalez_start = time.perf_counter()
            gonzalez_solution = gonzalez(k, d, points)
            gonzalez_end = time.perf_counter()
            results[str(k)]["gonzalez_sec"].append(gonzalez_end - gonzalez_start)
            results[str(k)]["gonzalez_r"].append(gonzalez_solution["radius"])

            # DA
            da = DoublingKCenter(k=k, d=d)
            da_start = time.perf_counter()
            da_solution = simulate_streaming(da, points)
            da_end = time.perf_counter()
            da_c_r = check_radius(d, points, da_solution["centers"])
            results[str(k)]["da_sec"].append(da_end - da_start)
            results[str(k)]["da_r"].append(da_solution["radius"])
            results[str(k)]["da_c_r"].append(da_c_r)

            # RDA
            rda = RandomizedDoublingKCenter(k=k, d=d)
            rda_start = time.perf_counter()
            rda_solution = simulate_streaming(rda, points)
            rda_end = time.perf_counter()
            rda_c_r = check_radius(d, points, rda_solution["centers"])
            results[str(k)]["rda_sec"].append(rda_end - rda_start)
            results[str(k)]["rda_r"].append(rda_solution["radius"])
            results[str(k)]["rda_c_r"].append(rda_c_r)

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(k)]["psa_sec"].append(psa_end - psa_start)
            results[str(k)]["psa_r"].append(psa_solution["radius"])
            results[str(k)]["psa_c_r"].append(psa_c_r)

    stats = []

    for k, value in results.items():
        gonzalez_r_mean = statistics.mean(value["gonzalez_r"])
        stats.append({
            "k": k,
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
    write_json("experiment_2", data)


def plot_1_experiment_2() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_2.json")

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
        x_label="k",
        y_label="",
        algorithms=algorithms
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_1_experiment_2.jpg")

    plt.savefig(plot_file_path)
