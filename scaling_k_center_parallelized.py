

from itertools import combinations
import math
from typing import Callable


class ScalingKCenter:

    def __init__(self, k: int, r0: float, alpha: float, d: Callable, points: list):
        self.k, self.alpha, self.d = k, alpha, d

        self.eta = 2 * alpha ** 2 / (alpha - 1)
        self.r = r0
        self.centers: list = points

        self._merge()

    # Radius welcher verwendet wird um eingefügten Punkten ein Zentrum zu zuweisen
    @property
    def radius_bound(self) -> float:
        return self.eta * self.r

    # Threshold um Zentren zu mergen
    @property
    def threshold(self) -> float:
        return 2 * self.alpha * self.r


    def insert(self, point) -> None:

        self._update(point)

        self._merge()


    def _update(self, point) -> None:

        closest = min(self.centers, key=lambda c: self.d(c, point))

        if self.d(closest, point) > self.radius_bound:
            self.centers.append(point)


    def _merge(self) -> None:
        
        if len(self.centers) <= self.k:
            return

        self.r *= self.alpha

        remaining = list(range(len(self.centers)))
        new_centers: list = []

        while remaining:
            rep = remaining.pop(0)
            neighbors = [
                j for j in remaining
                if self.d(self.centers[rep], self.centers[j]) <= self.threshold
            ]
            for j in neighbors:
                remaining.remove(j)
            new_centers.append(self.centers[rep])

        self.centers = new_centers
        
        self._merge()

class ScalingKCenterParallelized:

    def __init__(
        self,
        k: int,
        d: Callable,
        eps: float = 0.5,
    ):
        self.k = k
        self.d = d

        # Parameterwahl aus Section 2: α = O(ε^-1), m = O((ε^-1)·ln(ε^-1))
        self.alpha = max(2.0, math.ceil(eps**-1))
        self.eta = 2 * self.alpha ** 2 / (self.alpha - 1)

        # ceil statt round: stellt sicher dass m ≥ 2, damit die Versetzung
        # der Instanzen tatsächlich einen Unterschied macht.
        self.m = max(2, math.ceil((eps ** -1) * math.log(eps ** -1)))

        self._buffer: list = []
        self._instances: list[ScalingKCenter] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle
    # ------------------------------------------------------------------

    def insert(self, point) -> None:
        if not self._initialized:
            self._buffer.append(point)
            if len(self._buffer) >= self.k + 1:
                self._initialize()
        else:
            for inst in self._instances:
                inst.insert(point)

    def query(self) -> tuple[float, list]:
        best = min(self._instances, key=lambda inst: inst.radius_bound)
        return best.radius_bound, best.centers

    # ------------------------------------------------------------------
    # Instanzen aufsetzen
    # ------------------------------------------------------------------

    def _initialize(self) -> None:

        # r0 = halber minimaler paarweiser Abstand der ersten k+1 Punkte
        min_dist = min(
            self.d(a, b) for a, b in combinations(self._buffer, 2)
        )
        r0 = min_dist / 2.0

        # m Instanzen mit versetzten Startwerten
        self._instances = [
            ScalingKCenter(
                k=self.k,
                r0=r0 * (self.alpha ** ((i / self.m) - 1)),
                alpha=self.alpha,
                d=self.d,
                points=self._buffer
            )
            for i in range(1, self.m + 1)
        ]

        self._initialized = True
