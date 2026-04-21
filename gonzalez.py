"""
Quelle:
T. F. Gonzalez (1985) — "Clustering to minimize the maximum intercluster distance"
Theoretical Computer Science, 38, S. 293-306.
"""

from typing import Callable

def gonzalez(points: list[object], d: Callable[[object, object], float], k: int) -> tuple[float, list[object]]:
    """O(n*k)"""

    if not points or k <= 0:
        return 0.0, []

    # Wähle ein beliebiges Startzentrum (z.B. erstes Element)
    centers = [points[0]]

    # Wähle iterativ das am weitesten entfernte Element als neues Zentrum
    while len(centers) < k:
        next_point = max(
            points,
            key=lambda p: min(d(p, c) for c in centers)
        )
        centers.append(next_point)

    # Berechne den Radius: maximale Distanz eines Punktes zum nächsten Zentrum
    radius = max(
        min(d(p, c) for c in centers)
        for p in points
    )

    return radius, centers