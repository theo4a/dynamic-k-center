import json
import os
import statistics
import time

from matplotlib import pyplot as plt

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from utils import boxplot, check_radius, euclidean_distance, simulate_streaming, write_json, generate_clustered_points


def experiment_1() -> None:
    """
    Using different standard diviations for the clusters when generating the points
    """

    n = 500
    k = 10
    sample_size = 100
    cluster_sds = [0.25, 0.5, 1, 2, 4, 8]
    d = euclidean_distance
    psa_m = 64

    results: dict[str, dict[str, list[float]]] = {}

    for cluster_sd in cluster_sds:

        results[str(cluster_sd)] = {
            "sec_GO": [],
            "R_GO": [],

            "sec_DA": [],
            "r'_DA": [],
            "R_DA": [],

            "sec_RDA": [],
            "r'_RDA": [],
            "R_RDA": [],

            "sec_PSA": [],
            "r'_PSA": [],
            "R_PSA": [],
        }

        for i in range(sample_size):
            print(i)

            points = generate_clustered_points(
                k=k,
                n=n,
                cluster_std=cluster_sd,
                dim=2,
                center_std=10,
                seed=i
            )

            # Gonzalez
            go_start = time.perf_counter()
            go_solution = gonzalez(k, d, points)
            go_end = time.perf_counter()
            results[str(cluster_sd)]["sec_GO"].append(go_end - go_start)
            results[str(cluster_sd)]["R_GO"].append(go_solution["radius"])

            # DA
            da = DoublingKCenter(k=k, d=d)
            da_start = time.perf_counter()
            da_solution = simulate_streaming(da, points)
            da_end = time.perf_counter()
            da_c_r = check_radius(d, points, da_solution["centers"])
            results[str(cluster_sd)]["sec_DA"].append(da_end - da_start)
            results[str(cluster_sd)]["r'_DA"].append(da_solution["radius"])
            results[str(cluster_sd)]["R_DA"].append(da_c_r)

            # RDA
            rda = RandomizedDoublingKCenter(k=k, d=d)
            rda_start = time.perf_counter()
            rda_solution = simulate_streaming(rda, points)
            rda_end = time.perf_counter()
            rda_c_r = check_radius(d, points, rda_solution["centers"])
            results[str(cluster_sd)]["sec_RDA"].append(rda_end - rda_start)
            results[str(cluster_sd)]["r'_RDA"].append(rda_solution["radius"])
            results[str(cluster_sd)]["R_RDA"].append(rda_c_r)

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(cluster_sd)]["sec_PSA"].append(psa_end - psa_start)
            results[str(cluster_sd)]["r'_PSA"].append(psa_solution["radius"])
            results[str(cluster_sd)]["R_PSA"].append(psa_c_r)

    stats = []
    for cluster_sd, value in results.items():
        gonzalez_r_mean = statistics.mean(value["R_GO"])
        stats.append({
            "cluster_sd": cluster_sd,
            "mean(R_GO)": gonzalez_r_mean,
            "mean(r'_DA)/mean(R_GO)": statistics.mean(value["r'_DA"]) / gonzalez_r_mean,
            "mean(r'_PSA)/mean(R_GO)": statistics.mean(value["r'_PSA"]) / gonzalez_r_mean,
        })

    data = {
        "info": f"m: {psa_m}",
        "results": results,
        "statistics": stats
    }

    write_json("experiment_1", data)


def experiment_1_plot_1() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_1.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        r"$R_{GO}$": "R_GO",

        r"$r'_{DA}$": "r'_DA",
        r"$R_{DA}$": "R_DA",

        r"$r'_{PSA-64}$": "r'_PSA",
        r"$R_{PSA-64}$": "R_PSA",
    }

    fig, ax = boxplot(
        data=data["results"],
        x_label="Cluster SD",
        y_label="",
        algorithms=algorithms
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "experiment_1_plot_1.jpg")
    plt.savefig(plot_file_path, dpi=300)
    plt.close(fig)