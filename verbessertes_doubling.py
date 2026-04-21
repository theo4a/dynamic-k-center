import math
import random
from itertools import combinations
from typing import Callable


class ScalingInstance:
    """
    Eine einzelne Instanz des Scaling-Algorithmus mit Parameter α.
    Entspricht dem verallgemeinerten Doubling-Algorithmus aus Section 2,
    wobei r um genau den Faktor α skaliert wird.

    Invarianten zu Beginn jeder Phase i:
      (a) radius(Cⱼ) ≤ η·r,  mit η = 2α²/(α-1)
      (b) paarweise Zentrumsabstände ≥ 2αr
      (c) r ≤ OPT
    """

    def __init__(self, k: int, r0: float, alpha: float, dist: Callable):
        self.k, self.alpha, self.dist = k, alpha, dist
        self.eta = 2 * alpha ** 2 / (alpha - 1)
        self.r = r0
        self.centers: list = []
        self._initialized = False
        self._buffer: list = []

    def insert(self, point) -> None:
        if not self._initialized:
            self._buffer.append(point)
            if len(self._buffer) == self.k + 1:
                self._initialize()
        else:
            self._update(point)

    def radius_bound(self) -> float:
        """Gibt den aktuellen Radius-Bound η·r zurück."""
        return self.eta * self.r

    # ------------------------------------------------------------------
    # Initialisierung
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        self.centers = list(self._buffer)
        self._initialized = True
        # Kein _merge() hier: k+1 Zentren sind der korrekte Startzustand.
        # Die Versetzung r0·α^((i/m)-1) muss erhalten bleiben bis der
        # (k+2)-te Punkt eintrifft und _update() → _merge() auslöst.
        # Ein sofortiger _merge() würde r mit α multiplizieren und damit
        # die Versetzung aller Instanzen auf fast denselben Wert bringen.

    def finalize(self) -> None:
        """Erzwingt Merges bis ≤ k Zentren übrig sind.
        Wird in query() aufgerufen wenn der Input vollständig ist."""
        while len(self.centers) > self.k:
            self._merge()

    # ------------------------------------------------------------------
    # Update-Stage
    # ------------------------------------------------------------------

    def _update(self, point) -> None:
        # Punkt dem nächsten Zentrum zuweisen, sofern η·r eingehalten wird
        closest = min(self.centers, key=lambda c: self.dist(c, point))
        if self.dist(closest, point) > self.eta * self.r:
            self.centers.append(point)   # neues Cluster

        # k+1 Cluster erreicht → Merge-Stage starten
        if len(self.centers) == self.k + 1:
            self._merge()

    # ------------------------------------------------------------------
    # Merge-Stage: r um Faktor α erhöhen, Threshold-Graph mergen
    # ------------------------------------------------------------------

    def _merge(self) -> None:
        self.r *= self.alpha   # rᵢ₊₁ = α·rᵢ

        # Threshold-Graph auf Zentren: Kante ⟺ Abstand ≤ 2αr
        # Greedy: wähle Repräsentant, absorbiere alle Nachbarn
        threshold = 2 * self.alpha * self.r
        remaining = list(range(len(self.centers)))
        new_centers: list = []

        while remaining:
            rep = remaining.pop(0)
            neighbors = [
                j for j in remaining
                if self.dist(self.centers[rep], self.centers[j]) <= threshold
            ]
            for j in neighbors:
                remaining.remove(j)
            new_centers.append(self.centers[rep])

        self.centers = new_centers


class DoublingClustering:
    """
    Parallelisierter Scaling-Algorithmus – (2+ε)-Approximation.
    McCutchen & Khuller (2008), Section 2.
    Basiert auf Charikar, Chekuri, Feder, Motwani (STOC 1997).

    Führt m Instanzen des Scaling-Algorithmus mit versetzten r-Startwerten
    parallel aus. Der Startwert der i-ten Instanz (i=1,...,m) ist:

        r_i = r0 · α^((i/m) - 1)

    sodass die r-Werte aller Instanzen zusammen ein dichtes Gitter bilden.
    Am Ende liefert die Instanz mit dem kleinsten Radius-Bound das Ergebnis.

    Approximationsfaktor: (η/α) · ᵐ√α  →  2+ε
    Parameterwahl:        α = 1/ε,  m = ⌈(1/ε)·ln(1/ε)⌉
    """

    def __init__(
        self,
        k: int,
        dist: Callable,
        eps: float = 0.5,
    ):
        self.k = k
        self.dist = dist

        # Parameterwahl aus Section 2: α = O(1/ε), m = O((1/ε)·ln(1/ε))
        self.alpha = max(2.0, 1.0 / eps)
        # ceil statt round: stellt sicher dass m ≥ 2, damit die Versetzung
        # der Instanzen tatsächlich einen Unterschied macht.
        self.m = max(2, math.ceil((1.0 / eps) * math.log(1.0 / eps)))

        self._buffer: list = []
        self._instances: list[ScalingInstance] = []
        self._r0: float | None = None

        print(
            f"[Init] α={self.alpha:.2f},  m={self.m},  "
            f"erw. Approximationsfaktor ≤ {self._approx_factor():.3f}"
        )

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle
    # ------------------------------------------------------------------

    def insert(self, point) -> None:
        if self._r0 is None:
            self._buffer.append(point)
            if len(self._buffer) == self.k + 1:
                self._setup_instances()
        else:
            for inst in self._instances:
                inst.insert(point)

    def query(self) -> tuple[float, list]:
        """
        Gibt (radius_bound, centers) der besten Instanz zurück.
        Erzwingt zuerst finale Merges bis jede Instanz ≤ k Zentren hat,
        dann wird die Instanz mit dem kleinsten Radius-Bound gewählt.
        """
        for inst in self._instances:
            inst.finalize()
        best = min(self._instances, key=lambda inst: inst.radius_bound())
        return best.radius_bound(), best.centers

    # ------------------------------------------------------------------
    # Instanzen aufsetzen
    # ------------------------------------------------------------------

    def _setup_instances(self) -> None:
        # r0 = halber minimaler paarweiser Abstand der ersten k+1 Punkte
        min_dist = min(
            self.dist(a, b) for a, b in combinations(self._buffer, 2)
        )
        self._r0 = min_dist / 2.0

        # m Instanzen mit versetzten Startwerten
        self._instances = [
            ScalingInstance(
                k=self.k,
                r0=self._r0 * (self.alpha ** ((i / self.m) - 1)),
                alpha=self.alpha,
                dist=self.dist,
            )
            for i in range(1, self.m + 1)
        ]

        # Puffer-Punkte in alle Instanzen einspeisen
        for point in self._buffer:
            for inst in self._instances:
                inst.insert(point)

        print(f"[Setup] r0={self._r0:.4f},  {self.m} Instanzen gestartet")

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _approx_factor(self) -> float:
        # (η/α) · ᵐ√α,  mit η = 2α²/(α-1)
        eta = 2 * self.alpha ** 2 / (self.alpha - 1)
        return (eta / self.alpha) * (self.alpha ** (1.0 / self.m))

