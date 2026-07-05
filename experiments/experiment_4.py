import json
import os
import statistics
import time

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from utils import check_radius, euclidean_distance, simulate_streaming, write_json, boxplot, generate_gauß_clusters


def experiment_4() -> None:
    """
    Using different dimensions for the gernated points
    """

    n = 100
    k = 10
    sample_size = 100
    dims = [2, 4, 8, 16, 32, 64, 128, 256]
    d = euclidean_distance
    psa_m = 16

    results: dict[str, dict[str, list[float]]] = {}

    for dim in dims:
        results[str(dim)] = {
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
                dim=dim,
                center_std=10,
                seed=i
            )

            # Gonzalez
            gonzalez_start = time.perf_counter()
            gonzalez_solution = gonzalez(k, d, points)
            gonzalez_end = time.perf_counter()
            results[str(dim)]["gonzalez_sec"].append(gonzalez_end - gonzalez_start)
            results[str(dim)]["gonzalez_r"].append(gonzalez_solution["radius"])

            # DA
            da = DoublingKCenter(k=k, d=d)
            da_start = time.perf_counter()
            da_solution = simulate_streaming(da, points)
            da_end = time.perf_counter()
            da_c_r = check_radius(d, points, da_solution["centers"])
            results[str(dim)]["da_sec"].append(da_end - da_start)
            results[str(dim)]["da_r"].append(da_solution["radius"])
            results[str(dim)]["da_c_r"].append(da_c_r)

            # RDA
            rda = RandomizedDoublingKCenter(k=k, d=d)
            rda_start = time.perf_counter()
            rda_solution = simulate_streaming(rda, points)
            rda_end = time.perf_counter()
            rda_c_r = check_radius(d, points, rda_solution["centers"])
            results[str(dim)]["rda_sec"].append(rda_end - rda_start)
            results[str(dim)]["rda_r"].append(rda_solution["radius"])
            results[str(dim)]["rda_c_r"].append(rda_c_r)

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(dim)]["psa_sec"].append(psa_end - psa_start)
            results[str(dim)]["psa_r"].append(psa_solution["radius"])
            results[str(dim)]["psa_c_r"].append(psa_c_r)

    stats = []

    for dimensions, value in results.items():
        gonzalez_r_mean = statistics.mean(value["gonzalez_r"])
        stats.append({
            "dimensions": dimensions,
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
    write_json("experiment_4", data)


def plot_1_experiment_4() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_4.json")

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
        x_label="Dimensions",
        y_label="",
        algorithms=algorithms
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_1_experiment_4.jpg")

    plt.savefig(plot_file_path)


def plot_2_experiment_4() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_4.json")

    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        "DA r": "da_r",
        "DA r'": "da_c_r",

        "RDA r": "rda_r",
        "RDA r'": "rda_c_r",
    }

    plt = boxplot(
        data=data["results"],
        x_label="Dimensions",
        y_label="",
        algorithms=algorithms
    )
    
    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_2_experiment_4.jpg")

    plt.savefig(plot_file_path)
