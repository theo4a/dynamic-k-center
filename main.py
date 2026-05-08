import random
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from utils import plot, simulate
from streaming_k_center import DoublingKCenter, ParallelizedScalingKCenter
from hochbaum_shmoys import hochbaum_shmoys
from generators import cluster_by_cluster, cluster_by_cluster_fixed_radius, evenly_growing_fixed_radius, evenly_growing_k_cluster, generate_points
from gonzalez import gonzalez

def euclidean_distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



k = 10
d = euclidean_distance
points = evenly_growing_fixed_radius(k, 1000, 10, 0)
#points = generate_points(20)

offline_algos = [
    ("Gonzalez", gonzalez)
]

streaming_algos = [
    ("Doubling", DoublingKCenter(k, d)),
    ("Parallized", ParallelizedScalingKCenter(k, d, 0.1))
]

snapshots = simulate(streaming_algos, offline_algos, points, 9, d, k)
plot(snapshots)