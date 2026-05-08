from typing import Callable


def k_center_with_r(r: float, k: int, d: Callable, points: list[object]) -> list[object] | None:
    
    uncovered = set(points)
    centers = []

    while uncovered:
        if len(centers) >= k:
            return

        # Wähle einen beliebigen unbedeckten Punkt als Zentrum
        c = next(iter(uncovered))
        centers.append(c)

        # Entferne alle Punkte im Radius r
        to_remove = set()
        for p in uncovered:
            if d(c, p) <= r:
                to_remove.add(p)

        uncovered -= to_remove

    return centers


def hochbaum_shmoys(k: int, d: Callable, points: list[object]) -> tuple[float, list[object]]:
    
    # Edge case: nur ein Punkt
    if len(points) == 1:
        return 0.0, points
    
    # Alle paarweisen Distanzen als Kandidatenradien
    distances = set()
    n = len(points)

    for i in range(n):
        for j in range(i + 1, n):
            distances.add(d(points[i], points[j]))

    sorted_radii = sorted(distances)

    left, right = 0, len(sorted_radii) - 1
    best_r = sorted_radii[right]
    best_centers = []

    # Binäre Suche
    while left <= right:
        mid = (left + right) // 2
        r = sorted_radii[mid]

        centers = k_center_with_r(r, k, d, points)

        if centers:
            best_r = r
            best_centers = centers
            right = mid - 1
        else:
            left = mid + 1

    return best_r, best_centers