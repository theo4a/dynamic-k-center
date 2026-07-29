import heapq
from itertools import combinations
from typing import Callable

from algorithms.online.doubling_k_center import DoublingKCenter


class ScalingKCenter(DoublingKCenter):

    def __init__(self, k: int, d: Callable[[object, object], float], beta: float, r1: float):

        self.k: int = k
        self.d: Callable[[object, object], float] = d

        self.alpha: float = beta / (beta - 1)
        self.beta: float = beta

        self.r: float = r1

        self.distances: list[tuple[float, int, int]] = []
        self.centers: dict[int, object] = {}

        self._initialized = False


    def _initialize(self) -> None:

        points = list(self.centers.values())

        for a, b in combinations(points, 2):
            heapq.heappush(self.distances, (self.d(a, b), id(a), id(b)))

        self._merging_stage()
        self._initialized = True