from itertools import combinations
import math
from typing import Callable


class DoublingKCenter:

    def __init__(self, k: int, d: Callable[[object, object], float]):

        self.k = k
        self.d = d

        self.alpha = 2
        self.beta = 2

        self.centers: list[object] = []
        self.r: float | None = None
        
        self._initialized = False


    def insert(self, point: object) -> None:

        if not self._initialized:
            self.centers.append(point)
            if len(self.centers) == self.k + 1:
                self._initialize()
        else:
            self._update_stage(point)
            while len(self.centers) >= self.k + 1:
                # Nächste Phase startet ab hier
                self._merge_stage()

    def query(self) -> tuple[float, list[object]]:
        return self.alpha * 2 * self.r, self.centers


    def _initialize(self) -> None:

        # initialisiere r mit dem halben minimalen Zentrenabstand
        closet = min(self.d(a, b) for a, b in combinations(self.centers, 2))
        self.r = closet / 2

        # Phase mit der merge stage starten, da k+1 Zentren vorhanden sind
        self._merge_stage()

        self._initialized = True


    def _merge_stage(self) -> None:

        self.r *= self.beta

        merged = []

        while self.centers:
            current = self.centers.pop(0)
            merged.append(current)

            # Entferne alle Nachbarn aus der Zentrumsmenge
            self.centers = [
                c for c in self.centers
                if self.d(current, c) > self.r * 2
            ]

        self.centers = merged


    def _update_stage(self, point: object) -> None:

        closest = min(self.centers, key=lambda c: self.d(c, point))

        # Neues Zentrum erstellen, falls das nächste Zentrum zu weit entfernt ist
        if self.d(closest, point) > self.alpha * 2 * self.r:
            self.centers.append(point)


class ScalingKCenter(DoublingKCenter):

    def __init__(self, k: int, d: Callable[[object, object], float], beta: float, r0: float):

        self.k = k
        self.d = d

        self.alpha = beta/(beta-1)
        self.beta = beta

        self.centers: list[object] = []
        self.r: float = r0
        
        self._initialized = False

    def _initialize(self) -> None:

        # Phase mit der merge stage starten, da k+1 Zentren vorhanden sind
        self._merge_stage()

        self._initialized = True

class ParallelizedScalingKCenter:

    def __init__(
        self,
        k: int,
        d: Callable,
        eps: float,
    ):
        self.k = k
        self.d = d

        self.beta = eps**-1
        self.m = math.ceil(eps ** -1 * math.log(eps ** -1))
        
        self._instances: list[ScalingKCenter] = []

        self._buffer: list = []
        self._initialized: bool = False


    def insert(self, point) -> None:

        # Füge Punkt in den Buffer ein, falls nocht nicht initialisiert wurde
        if not self._initialized:
            self._buffer.append(point)
            if len(self._buffer) >= self.k + 1:
                self._initialize()

        # Füge den Punkt in jede Instanz ein
        else:
            for inst in self._instances:
                inst.insert(point)


    def query(self) -> tuple[float, list]:

        # Gibt das Ergebnis der Instanz mit dem geringsten Radius zurück
        return min(
            (inst.query() for inst in self._instances),
            key=lambda x: x[0]
        )


    def _initialize(self) -> None:

        # r0 = halber minimaler paarweiser Abstand der ersten k+1 Punkte
        min_dist = min(
            self.d(a, b) for a, b in combinations(self._buffer, 2)
        )
        r0 = min_dist / 2.0

        # m Instanzen mit versetztem r0
        self._instances = [
            ScalingKCenter(
                k=self.k,
                d=self.d,
                beta=self.beta,
                r0=r0 * (self.beta ** ((i / self.m) - 1)),
            )
            for i in range(1, self.m + 1)
        ]

        # Zentren aus dem Buffer in die Instanzen hinzufügen
        for inst in self._instances:
            for p in self._buffer:
                inst.insert(p)
        self._buffer = []

        self._initialized = True