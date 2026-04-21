

import math
import random



def evenly_growing_k_cluster(k: int, n: int, seed: int) -> list[object]:
    rng = random.Random(seed)

    cluster_centers = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(k)]
    points = []

    for _ in range(n):
        cx, cy = rng.choice(cluster_centers)
        x = rng.gauss(cx, 5.0)
        y = rng.gauss(cy, 5.0)
        points.append((x, y))

    return points


def evenly_growing_fixed_radius(k: int, n: int, r: float, seed: int) -> list[object]:
    rng = random.Random(seed)

    centers = []
    while len(centers) < k:
        candidate = (rng.uniform(0, 100), rng.uniform(0, 100))
        if all(math.dist(candidate, c) >= 2 * r for c in centers):
            centers.append(candidate)

    # 👉 wie viele Punkte bekommt jedes Cluster?
    counts = [n // k] * k
    for i in range(n % k):
        counts[i] += 1

    # 👉 für jedes Cluster: wann kommt der Randpunkt?
    boundary_positions = [rng.randrange(counts[i]) for i in range(k)]
    current_counts = [0] * k

    points = []

    for _ in range(n):
        idx = rng.randrange(k)

        # falls dieses Cluster schon voll ist → neues wählen
        while current_counts[idx] >= counts[idx]:
            idx = rng.randrange(k)

        cx, cy = centers[idx]
        angle = rng.uniform(0, 2 * math.pi)

        if current_counts[idx] == boundary_positions[idx]:
            dist = r
        else:
            dist = min(abs(rng.gauss(0, r / 2)), r - 1e-12)

        x = cx + dist * math.cos(angle)
        y = cy + dist * math.sin(angle)
        points.append((x, y))

        current_counts[idx] += 1

    return points

def cluster_by_cluster(k: int, n: int, seed: int) -> list[object]:
    rng = random.Random(seed)

    cluster_centers = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(k)]

    points = []
    points_per_cluster = n // k
    remainder = n % k

    for i, (cx, cy) in enumerate(cluster_centers):
        # verteile Restpunkte auf erste Cluster
        count = points_per_cluster + (1 if i < remainder else 0)

        for _ in range(count):
            x = rng.gauss(cx, 5.0)
            y = rng.gauss(cy, 5.0)
            points.append((x, y))

    return points

def cluster_by_cluster_fixed_radius(k: int, n: int, r: float, seed: int) -> list[object]:
    rng = random.Random(seed)

    # Zentren mit Abstand ≥ 2r
    centers = []
    while len(centers) < k:
        candidate = (rng.uniform(0, 100), rng.uniform(0, 100))
        if all(math.dist(candidate, c) >= 2 * r for c in centers):
            centers.append(candidate)

    points = []
    points_per_cluster = n // k
    remainder = n % k

    for i, (cx, cy) in enumerate(centers):
        count = points_per_cluster + (1 if i < remainder else 0)

        # 👉 zufällige Position für den Randpunkt
        boundary_index = rng.randrange(count)

        for j in range(count):
            angle = rng.uniform(0, 2 * math.pi)

            if j == boundary_index:
                dist = r  # garantierter Randpunkt
            else:
                dist = min(abs(rng.gauss(0, r / 2)), r - 1e-12)

            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            points.append((x, y))

    return points
