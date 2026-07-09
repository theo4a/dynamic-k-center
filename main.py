import json
import os
import statistics

from experiments.experiment_1 import experiment_1, experiment_1_plot_1
from experiments.experiment_2 import experiment_k_clustered_points, experiment_k_clustered_points_fixed_n, experiment_2_plot_1, experiment_k_uniform_points
from experiments.experiment_3 import experiment_3, experiment_3_plot_1
from experiments.experiment_4 import experiment_4, experiment_4_plot_1, experiment_4_plot_2
from experiments.experiment_5 import experiment_5, experiment_5_plot_1
from experiments.experiment_7 import experiment_7_1, experiment_7_2, experiment_7_2_plot_1, experiment_7_plot_2, experiment_7_plot_3
from utils import nrw_osm_data_to_metric_centroids_csv, write_json


file_path = os.path.join(os.path.dirname(__file__), "results", "data", "experiment_4.json")
with open(file_path, "r") as f:
    data = json.load(f)

results = data["results"]

stats = []

for k, value in results.items():
    gonzalez_r_mean = statistics.mean(value["R_GO"])
    stats.append({
        "k": k,
        "mean(GO_R)": gonzalez_r_mean,
        "mean(DA_r')/mean(GO_R)": statistics.mean(value["r'_DA"]) / gonzalez_r_mean,
        "mean(DA_R)/mean(GO_R)": statistics.mean(value["R_DA"]) / gonzalez_r_mean,
        "mean(PSA_r')/mean(GO_R)": statistics.mean(value["r'_PSA"]) / gonzalez_r_mean,
        "mean(PSA_R)/mean(GO_R)": statistics.mean(value["R_PSA"]) / gonzalez_r_mean,
    })

data["statistics"] = stats

write_json("experiment_4", data)

#experiment_1()
#experiment_1_plot_1()

#experiment_k_clustered_points_fixed_n()
#experiment_k_clustered_points()
#experiment_k_uniform_points()
#experiment_2_plot_1()

#experiment_3()
#experiment_3_plot_1()

#experiment_4()
#experiment_4_plot_1()
#experiment_4_plot_2()

#experiment_5()
#experiment_5_plot_1()

#experiment_6()
#experiment_6_plot_1()

#experiment_7_1()
#experiment_7_2()
#experiment_7_2_plot_1()
#experiment_7_plot_2()
#experiment_7_plot_3()

