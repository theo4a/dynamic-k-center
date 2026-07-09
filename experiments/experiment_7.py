import json
import os

from matplotlib import pyplot as plt
import numpy as np

from algorithms.online.doubling_k_center import DoublingKCenter
from algorithms.online.randomized_doubling_k_center import RandomizedDoublingKCenter
from algorithms.online.parallelized_scaling_k_center import ParallelizedScalingKCenter
from utils import get_subplot, load_dataset, write_json, euclidean_distance



def experiment_7_1() -> None:
    """
    Using geopoints of households in hamburg as an input
    """
    
    k = 16
    d = euclidean_distance
    points = load_dataset()
    psa_m = 64

    da = DoublingKCenter(k=k, d=d)
    rda = RandomizedDoublingKCenter(k=k, d=d)
    psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)

    results = {}

    x = 10000
    bias = 2

    snapshot_indices = (np.linspace(0, 1, x) ** bias * (len(points) - 1)).astype(int)

    for i in range(len(points)):

        print(i)

        da.insert(points[i])
        rda.insert(points[i])
        psa.insert(points[i])

        if i > k and i in snapshot_indices:

            results[str(i)] = {}
            
            da_solution = da.query()
            #results[str(i)]["C_DA"] = da_solution["centers"]
            results[str(i)]["r'_DA"] = da_solution["radius"]

            rda_solution = rda.query()
            #results[str(i)]["C_RDA"] = rda_solution["centers"]
            results[str(i)]["r'_RDA"] = rda_solution["radius"]

            psa_solution = psa.query()
            #results[str(i)]["C_PSA"] = psa_solution["centers"]
            results[str(i)]["r'_PSA"] = psa_solution["radius"]

    data = {
        "info": "",
        "results": results
    }

    write_json("experiment_7", data)

def experiment_7_2() -> None:
    """
    Using geopoints of households in hamburg as an input
    """
    
    k = 16
    d = euclidean_distance
    points = load_dataset()
    psa_m = 64

    da = DoublingKCenter(k=k, d=d)
    rda = RandomizedDoublingKCenter(k=k, d=d)
    psa = ParallelizedScalingKCenter(k=k, d=d, m=psa_m)

    for i in range(len(points)):

        print(i)

        da.insert(points[i])
        rda.insert(points[i])
        psa.insert(points[i])
    
    results = {}

    da_solution = da.query()
    results["C_DA"] = da_solution["centers"]
    results["r'_DA"] = da_solution["radius"]

    rda_solution = rda.query()
    results["C_RDA"] = rda_solution["centers"]
    results["r'_RDA"] = rda_solution["radius"]

    psa_solution = psa.query()
    results["C_PSA"] = psa_solution["centers"]
    results["r'_PSA"] = psa_solution["radius"]

    data = {
        "info": "",
        "results": results
    }

    write_json("experiment_7_2", data)

def experiment_7_2_plot_1() -> None:
    points = load_dataset()

    n_samples = 10000
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(points), size=n_samples, replace=False)
    sample = np.array(points)[indices]

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "data",
        "experiment_7_2.json"
    )

    with open(file_path, "r") as f:
        data = json.load(f)

    results = data["results"]

    c_da = np.array(results["C_DA"])
    r_da = results["r'_DA"]

    c_rda = np.array(results["C_RDA"])
    r_rda = results["r'_RDA"]

    c_psa = np.array(results["C_PSA"])
    r_psa = results["r'_PSA"]


    fig, ax = get_subplot()

    # Punktwolke
    ax.scatter(
        sample[:, 0],
        sample[:, 1],
        s=1,
        alpha=0.5
    )

    # Zentren
    ax.scatter(
        c_da[:, 0],
        c_da[:, 1],
        s=20,
        color="red",
        zorder=3,
        label="DA"
    )

    ax.scatter(
        c_rda[:, 0],
        c_rda[:, 1],
        s=20,
        color="yellow",
        zorder=3,
        label="RDA"
    )

    ax.scatter(
        c_psa[:, 0],
        c_psa[:, 1],
        s=20,
        color="green",
        zorder=3,
        label="PSA"
    )

    # Kreise DA
    for cx, cy in c_da:
        circle = plt.Circle(
            (cx, cy),
            r_da,
            color="red",
            fill=False,
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(circle)

    # Kreise RDA
    for cx, cy in c_rda:
        circle = plt.Circle(
            (cx, cy),
            r_rda,
            color="yellow",
            fill=False,
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(circle)

    # Kreise PSA
    for cx, cy in c_psa:
        circle = plt.Circle(
            (cx, cy),
            r_psa,
            color="green",
            fill=False,
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(circle)

    ax.set_aspect("equal")

    ax.set_xlabel("Easting (m) - UTM32N / EPSG:25832")
    ax.set_ylabel("Northing (m) - UTM32N / EPSG:25832")

    ax.legend()

    plot_file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "plots",
        "experiment_7_plot_1.jpg"
    )

    plt.savefig(plot_file_path, dpi=300)
    plt.close(fig)


def experiment_7_plot_2() -> None:

    points = load_dataset()
    
    n_samples = 10000
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(points), size=n_samples, replace=False)
    sample = np.array(points)[indices]

    
    fig, ax = get_subplot()

    # Plot
    ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel("Easting (m) - UTM32N / EPSG:25832")
    ax.set_ylabel("Northing (m) - UTM32N / EPSG:25832")

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "experiment_7_plot_2.jpg")
    plt.savefig(plot_file_path, dpi=300)
    plt.close(fig)

def experiment_7_plot_3() -> None:

    file_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "experiment_7.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    results = data["results"]

    # x-Werte
    x = sorted([int(k) for k in results.keys()])

    # y-Werte für jede Linie
    a_werte = [results[str(k)]["r'_DA"] for k in x]
    b_werte = [results[str(k)]["r'_RDA"] for k in x]
    c_werte = [results[str(k)]["r'_PSA"] for k in x]

    # Linien plotten
    plt.plot(x, a_werte, label="DA")
    plt.plot(x, b_werte, label="RDA")
    plt.plot(x, c_werte, label="PSA-64")

    plt.xscale("log", base=2)

    # Beschriftung
    plt.xlabel("Inserted Points")
    plt.ylabel(r"$r'$")

    # Legende anzeigen
    plt.legend()

    # Grid optional
    plt.grid(True)

    plot_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "plots", "experiment_7_plot_3.jpg")
    plt.savefig(plot_file_path, dpi=300)
    plt.clf()
