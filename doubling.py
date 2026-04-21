import math
import random
from itertools import combinations
from typing import Callable



class DoublingClustering:
    """
    Incremental Clustering – Doubling Algorithm (8-Approximation).
    Charikar, Chekuri, Feder, Motwani – STOC 1997.

    Speichert nur Clusterzentren, keine vollständigen Punktmengen.
    Invarianten zu Beginn jeder Phase i:
      (a) radius(Cⱼ) ≤ α·dᵢ
      (b) paarweise Zentrumsabstände ≥ dᵢ
      (c) dᵢ ≤ OPT
    Mit α = β = 2 folgt Approximationsfaktor 2αβ = 8.
    """

    def __init__(self, k: int, dist: Callable[[object, object], float], alpha: float = 2.0, beta: float = 2.0):
        assert alpha / (alpha - 1) <= beta, "Parameterbedingung α/(α-1) ≤ β verletzt"

        self.k, self.dist, self.alpha, self.beta = k, dist, alpha, beta

        self.centers: list[object] = []   # ein Zentrum pro Cluster
        self.d: float = 0.0              # aktueller Phasenschwellwert dᵢ
        self._buffer: list[object] = []   # Puffer vor Initialisierung

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle
    # ------------------------------------------------------------------

    def insert(self, point: object) -> None:


        if len(self.centers) == 0 and len(self._buffer) <= self.k:
            self._buffer.append(point)
            if len(self._buffer) == self.k + 1:
                self._initialize()
        else:
            self._update(point)

    def query(self) -> tuple[float, list[object]]:
        """Gibt Radius und Zentren zurück"""
        return self.alpha * self.d, self.centers

    # ------------------------------------------------------------------
    # Initialisierung  (setzt Phase 1 auf)
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        self.centers = list(self._buffer)
        # d₁ = minimaler paarweiser Zentrumsabstand  →  Invariante (c): d₁ ≤ OPT
        self.d = min(self.dist(a, b) for a, b in combinations(self.centers, 2))
        # k+1 Zentren liegen bereits vor → Merge-Stage direkt anstoßen
        self._merge()

    # ------------------------------------------------------------------
    # Update-Stage: neuen Punkt einordnen
    # ------------------------------------------------------------------

    def _update(self, point: object) -> None:
        radius_bound = self.alpha * self.d

        # Punkt dem nächsten Zentrum zuweisen, sofern Radiusschranke hält
        closest = min(self.centers, key=lambda c: self.dist(c, point))
        if self.dist(closest, point) <= radius_bound:
            pass  # Punkt liegt im Cluster – Zentrum ändert sich nicht
        else:
            self.centers.append(point)   # neues Cluster

        # k+1 Cluster erreicht → Merge-Stage starten
        if len(self.centers) == self.k + 1:
            self._merge()

    # ------------------------------------------------------------------
    # Merge-Stage: Schwellwert verdoppeln, Threshold-Graph mergen
    # ------------------------------------------------------------------

    def _merge(self) -> None:
        self.d *= self.beta   # dᵢ₊₁ = β·dᵢ

        # Threshold-Graph auf Zentren: Kante ⟺ Abstand ≤ dᵢ₊₁
        # Greedy: wähle Knoten als Repräsentant, absorbiere alle Nachbarn
        remaining = list(range(len(self.centers)))
        new_centers: list[object] = []

        while remaining:
            rep = remaining.pop(0)
            # Nachbarn = alle noch nicht verarbeiteten Knoten im Threshold
            neighbors = [
                j for j in remaining
                if self.dist(self.centers[rep], self.centers[j]) <= self.d
            ]
            for j in neighbors:
                remaining.remove(j)
            new_centers.append(self.centers[rep])   # Repräsentant bleibt Zentrum

        self.centers = new_centers

        # Falls immer noch k+1 -> merge erneut aufrufen
        if len(self.centers) == self.k + 1:
            self._merge()

