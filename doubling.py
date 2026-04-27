from itertools import combinations
from typing import Callable



class DoublingKCenter:

    def __init__(self, k: int, d: Callable[[object, object], float]):

        self.k = k
        self.d = d

        self._initialized = False
        self._buffer: list[object] = []
        self.centers: list[object] = []
        self.r: float | None = None

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle
    # ------------------------------------------------------------------

    def insert(self, point: object) -> None:

        if not self._initialized:
            self._buffer.append(point)
            if len(self._buffer) == self.k + 1:
                self._initialize()
        else:
            self._update(point)

    def query(self) -> tuple[float, list[object]]:
        """Gibt Radius und Zentren zurück"""
        return 4 * self.r, self.centers

    # ------------------------------------------------------------------
    # Initialisierung  (setzt Phase 1 auf)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        self.centers = list(self._buffer)
        # r₁ = minimaler paarweiser Zentrumsabstand  →  Invariante (c): r₁ ≤ OPT
        if self.r is None:
            self.r = min(self.d(a, b) for a, b in combinations(self.centers, 2))/2

        self._initialized = True
        # k+1 Zentren liegen bereits vor → Merge-Stage direkt anstoßen
        self._merge()

    # ------------------------------------------------------------------
    # Update-Stage: neuen Punkt einordnen
    # ------------------------------------------------------------------

    def _update(self, point: object) -> None:

        # Punkt dem nächsten Zentrum zuweisen, sofern Radiusschranke hält
        closest = min(self.centers, key=lambda c: self.d(c, point))
        if self.d(closest, point) <= 4 * self.r:
            pass  # Punkt liegt im Cluster – Zentrum ändert sich nicht
        else:
            self.centers.append(point)   # neues Cluster

        if len(self.centers) == self.k + 1:
            self._merge()


    def _merge(self) -> None:

        self.r *= 2

        remaining = list(range(len(self.centers)))
        new_centers: list[object] = []

        while remaining:
            rep = remaining.pop(0)
            neighbors = [
                j for j in remaining
                if self.d(self.centers[rep], self.centers[j]) <= self.r * 2
            ]
            for j in neighbors:
                remaining.remove(j)
            new_centers.append(self.centers[rep])

        self.centers = new_centers

        if len(self.centers) == self.k + 1:
            self._merge()

