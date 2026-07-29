import json
import os
import statistics
import time

from matplotlib import pyplot as plt

from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import check_radius, euclidean_distance, generate_uniform_points, hsva_to_rgba, simulate_streaming, write_json, boxplot


def experiment_m() -> None:
    """
    Using different m's for PSA on points with a cluster standard diviation of 8
    """

    n = 500
    k = 10
    sample_size = 100
    psa_ms = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    d = euclidean_distance

    results: dict[str, dict[str, list[float]]] = {}

    for m in psa_ms:
        results[str(m)] = {
            "sec_PSA": [],
            "r_prime_PSA": [],
            "R_PSA": [],
        }

        for i in range(sample_size):
            print(i)

            points = generate_uniform_points(
                n=n,
                min=-100,
                max=100,
                dim=2,
                seed=i
            )

            # PSA
            psa = ParallelizedScalingKCenter(k=k, d=d, m=m)
            psa_start = time.perf_counter()
            psa_solution = simulate_streaming(psa, points)
            psa_end = time.perf_counter()
            psa_c_r = check_radius(d, points, psa_solution["centers"])
            results[str(m)]["sec_PSA"].append(psa_end - psa_start)
            results[str(m)]["r_prime_PSA"].append(psa_solution["radius"])
            results[str(m)]["R_PSA"].append(psa_c_r)

    stats = []

    for m, value in results.items():
        stats.append({
            "m": m,
            "mean(r_prime_PSA)": statistics.mean(value["r_prime_PSA"]),
            "mean(R_PSA)": statistics.mean(value["R_PSA"]),
        })

    data = {
        "info": "",
        "results": results,
        "statistics": stats
    }

    # Save results
    write_json("experiment_m", data)


def plot_experiment_m() -> None:
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_m.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    algorithms = {
        r"$R_{PSA}$": "R_PSA",
        r"$r'_{PSA}$": "r_prime_PSA",
    }

    colors = {
        r"$R_{PSA}$": hsva_to_rgba(120, 1, 0.8, 1),
        r"$r'_{PSA}$": hsva_to_rgba(120, 0.25, 0.8, 1),
    }

    fig, ax = boxplot(
        data=data["results"],
        x_label=r"$m$",
        y_label="",
        algorithms=algorithms,
        colors=colors
    )

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "plot_experiment_m.jpg")
    plt.savefig(plot_file_path, dpi=200)
    plt.close(fig)
