import json
import os
import statistics
import time

from matplotlib import pyplot as plt

from algorithms.offline.gonzalez import gonzalez
from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from utils import check_radius, euclidean_distance, generate_uniform_points, hsva_to_rgba, simulate_streaming, write_json, boxplot


def experiment_dimension() -> None:
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
            "r_prime_DA": [],
            "R_DA": [],

            "sec_RDA": [],
            "r_prime_RDA": [],
            "R_RDA": [],

            "sec_PSA": [],
            "r_prime_PSA": [],
            "R_PSA": [],
        }

        for i in range(sample_size):
            print(i)

            # points = generate_clustered_points(
            #     k=k,
            #     n=n,
            #     cluster_std=1,
            #     dim=dim,
            #     center_std=10,
            #     seed=i
            # )

            points = generate_uniform_points(
                n=100,
                min=-100,
                max=100,
                dim=dim,
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
            results[str(dim)]["r_prime_DA"].append(da_solution["radius"])
            results[str(dim)]["R_DA"].append(da_c_r)

            # RDA
            rda = RandomizedDoublingKCenter(k=k, d=d)
            rda_start = time.perf_counter()
            rda_solution = simulate_streaming(rda, points)
            rda_end = time.perf_counter()
            rda_c_r = check_radius(d, points, rda_solution["centers"])
            results[str(dim)]["sec_RDA"].append(rda_end - rda_start)
            results[str(dim)]["r_prime_RDA"].append(rda_solution["radius"])
            results[str(dim)]["R_RDA"].append(rda_c_r)

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(dim)]["sec_PSA"].append(psa_end - psa_start)
            results[str(dim)]["r_prime_PSA"].append(psa_solution["radius"])
            results[str(dim)]["R_PSA"].append(psa_c_r)

    stats = []

    for dimensions, value in results.items():
        mean_r_go = statistics.mean(value["R_GO"])
        stats.append({
            "dimensions": dimensions,
            "mean(sec_GO)": statistics.mean(value["sec_GO"]),
            "mean(R_GO)": mean_r_go,
            "mean(sec_DA)": statistics.mean(value["sec_DA"]),
            "mean(r_prime_DA)/mean(R_GO)": statistics.mean(value["r_prime_DA"]) / mean_r_go,
            "mean(R_DA)/mean(R_GO)": statistics.mean(value["R_DA"]) / mean_r_go,
            "mean(sec_RDA)": statistics.mean(value["sec_RDA"]),
            "mean(r_prime_RDA)/mean(R_GO)": statistics.mean(value["r_prime_RDA"]) / mean_r_go,
            "mean(R_RDA)/mean(R_GO)": statistics.mean(value["R_RDA"]) / mean_r_go,
            "mean(sec_PSA)": statistics.mean(value["sec_PSA"]),
            "mean(PSA_r')/mean(R_GO)": statistics.mean(value["r_prime_PSA"]) / mean_r_go,
            "mean(PSA_R)/mean(R_GO)": statistics.mean(value["R_PSA"]) / mean_r_go,

        })

    data = {
        "info": "",
        "results": results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_dimension", data)


def plot_experiment_dimension() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_dimension.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        r"$R_{GO}$": "R_GO",

        r"$R_{PSA-64}$": "R_PSA",
        r"$r'_{PSA-64}$": "r_prime_PSA",
        
        r"$R_{RDA}$": "R_RDA",
        r"$r'_{RDA}$": "r_prime_RDA",

        r"$R_{DA}$": "R_DA",
        r"$r'_{DA}$": "r_prime_DA",
    }

    colors = {
        r"$R_{GO}$": hsva_to_rgba(60, 1, 0.8, 1),

        r"$R_{PSA-64}$": hsva_to_rgba(120, 1, 0.8, 1),
        r"$r'_{PSA-64}$": hsva_to_rgba(120, 0.25, 0.8, 1),

        r"$R_{RDA}$": hsva_to_rgba(210, 1, 0.8, 1),
        r"$r'_{RDA}$": hsva_to_rgba(210, 0.5, 0.8, 1),

        r"$R_{DA}$": hsva_to_rgba(360, 1, 0.8, 1),
        r"$r'_{DA}$": hsva_to_rgba(360, 0.5, 0.8, 1),
    }

    fig, ax = boxplot(
        data=data["results"],
        x_label=r"$dimensions$",
        y_label="",
        algorithms=algorithms,
        colors=colors
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_experiment_dimension.jpg")
    plt.savefig(plot_file_path, dpi=200)
    plt.close(fig)


