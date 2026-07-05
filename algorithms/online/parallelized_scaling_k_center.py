from itertools import combinations
from typing import Callable

from algorithms.online.scaling_k_center import ScalingKCenter
from algorithms.online.streaming_k_center import StreamingKCenter


class ParallelizedScalingKCenter(StreamingKCenter):

    def __init__(
        self,
        k: int,
        d: Callable[[object, object], float],
        m: int,
    ):
        self.k = k
        self.d = d

        self.beta: float = m+1
        self.m = m
        
        self._instances: list[ScalingKCenter] = []

        self._buffer: list = []
        self._initialized: bool = False


    def insert(self, point: object) -> None:

        # Füge Punkt in den Buffer ein, falls nocht nicht initialisiert wurde
        if not self._initialized:
            self._buffer.append(point)
            if len(self._buffer) >= self.k + 1:
                self._initialize()

        # Füge den Punkt in jede Instanz ein
        else:
            for inst in self._instances:
                inst.insert(point)


    def query(self) -> dict:

        # Gibt das Ergebnis der Instanz mit dem geringsten Radius zurück
        return min(
            (inst.query() for inst in self._instances),
            key=lambda x: x["radius"]
        )


    def _initialize(self) -> None:

        # r0 = halber minimaler paarweiser Abstand der ersten k+1 Punkte
        min_dist = min(
            self.d(a, b) for a, b in combinations(self._buffer, 2)
        )
        r1 = min_dist / 2.0

        # m Instanzen mit versetztem r1
        self._instances = [
            ScalingKCenter(
                k=self.k,
                d=self.d,
                beta=self.beta,
                r1=r1 * (self.beta ** ((i / self.m) - 1)),
            )
            for i in range(1, self.m + 1)
        ]

        # Zentren aus dem Buffer in die Instanzen hinzufügen
        for inst in self._instances:
            for p in self._buffer:
                inst.insert(p)
        self._buffer = []

        self._initialized = True
