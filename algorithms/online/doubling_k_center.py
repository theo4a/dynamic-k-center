import heapq
from itertools import combinations
from typing import Callable

from algorithms.online.streaming_k_center import StreamingKCenter


class DoublingKCenter(StreamingKCenter):

    def __init__(self, k: int, d: Callable[[object, object], float]):

        self.k: Callable[[object, object], float] = k
        self.d: int = d

        self.alpha: float = 2
        self.beta: float = 2

        self.r: float = 0

        self.distances: list[tuple[float, int, int]] = []
        self.centers: dict[int, object] = {}

        self._initialized = False


    def insert(self, point: object) -> None:

        if not self._initialized:
            self.centers[id(point)] = point
            if len(self.centers) == self.k + 1:
                self._initialize()
        else:
            self._update_stage(point)
            while len(self.centers) >= self.k + 1:
                self._merge_stage()


    def query(self) -> dict:
        return {
            "radius": self.alpha * 2 * self.r,
            "centers": list(self.centers.values())
        }


    def _initialize(self) -> None:

        points = list(self.centers.values())
        min_dist = min(self.d(a, b) for a, b in combinations(points, 2))
        self.r = min_dist / 2

        self.centers = {id(c): c for c in points}

        for a, b in combinations(points, 2):
            heapq.heappush(self.distances, (self.d(a, b), id(a), id(b)))

        self._merge_stage()
        self._initialized = True


    def _merge_stage(self) -> None:

        self.r *= self.beta

        merged_centers: set = set()

        while self.distances and self.distances[0][0] <= 2 * self.r:
            dist, id_a, id_b = heapq.heappop(self.distances)

            if id_a not in self.centers or id_b not in self.centers:
                continue

            if id_a not in merged_centers and id_b not in merged_centers:
                self.centers.pop(id_b)
                merged_centers.add(id_a)

            elif id_a in merged_centers and id_b not in merged_centers:
                self.centers.pop(id_b)

            elif id_a not in merged_centers and id_b in merged_centers:
                self.centers.pop(id_a)


    def _update_stage(self, point: object) -> None:

        if not self.centers:
            self.centers[id(point)] = point
            return

        closest = min(self.centers.values(), key=lambda c: self.d(c, point))

        if self.d(closest, point) > self.alpha * 2 * self.r:
            for c in self.centers.values():
                heapq.heappush(self.distances, (self.d(c, point), id(c), id(point)))
            self.centers[id(point)] = point