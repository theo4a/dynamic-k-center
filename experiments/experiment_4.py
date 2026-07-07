import json
import os
import statistics
import time

from matplotlib import pyplot as plt

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
    psa_m = 64

    results: dict[str, dict[str, list[float]]] = {}

    for dim in dims:
        results[str(dim)] = {
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

            points = generate_gauß_clusters(
                k=k,
                n=n,
                cluster_std=1,
                dim=dim,
                center_std=10,
                seed=i
            )

            # Gonzalez
            go_start = time.perf_counter()
            go_solution = gonzalez(k, d, points)
            go_end = time.perf_counter()
            results[str(dim)]["sec_GO"].append(go_end - go_start)
            results[str(dim)]["R_GO"].append(go_solution["radius"])

            # DA
            da = DoublingKCenter(k=k, d=d)
            da_start = time.perf_counter()
            da_solution = simulate_streaming(da, points)
            da_end = time.perf_counter()
            da_c_r = check_radius(d, points, da_solution["centers"])
            results[str(dim)]["sec_DA"].append(da_end - da_start)
            results[str(dim)]["r'_DA"].append(da_solution["radius"])
            results[str(dim)]["R_DA"].append(da_c_r)

            # RDA
            rda = RandomizedDoublingKCenter(k=k, d=d)
            rda_start = time.perf_counter()
            rda_solution = simulate_streaming(rda, points)
            rda_end = time.perf_counter()
            rda_c_r = check_radius(d, points, rda_solution["centers"])
            results[str(dim)]["sec_RDA"].append(rda_end - rda_start)
            results[str(dim)]["r'_RDA"].append(rda_solution["radius"])
            results[str(dim)]["R_RDA"].append(rda_c_r)

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(dim)]["sec_PSA"].append(psa_end - psa_start)
            results[str(dim)]["r'_PSA"].append(psa_solution["radius"])
            results[str(dim)]["R_PSA"].append(psa_c_r)

    stats = []

    for dimensions, value in results.items():
        gonzalez_r_mean = statistics.mean(value["R_GO"])
        stats.append({
            "dimensions": dimensions,
            "mean(GO_R)": gonzalez_r_mean,
            "mean(DA_r')/mean(GO_R)": statistics.mean(value["r'_DA"]) / gonzalez_r_mean,
            "mean(PSA_r')/mean(GO_R)": statistics.mean(value["r'_PSA"]) / gonzalez_r_mean,

        })

    data = {
        "info": "",
        "results": results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_4", data)


def experiment_4_plot_1() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_4.json")
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
        x_label="Dimensions",
        y_label="",
        algorithms=algorithms
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "experiment_4_plot_1.jpg")
    plt.savefig(plot_file_path, dpi=300)
    plt.close(fig)


def experiment_4_plot_2() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_4.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        r"$r'_{DA}$": "r'_DA",
        r"$R_{DA}$'": "R_DA",

        r"$r'_{RDA}$": "r'_RDA",
        r"$R_{RDA}$": "R_RDA",
    }

    fig, ax = boxplot(
        data=data["results"],
        x_label="Dimensions",
        y_label="",
        algorithms=algorithms
    )
    
    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "experiment_4_plot_2.jpg")
    plt.savefig(plot_file_path, dpi=300)
    plt.close(fig)
