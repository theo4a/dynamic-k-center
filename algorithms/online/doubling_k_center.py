import heapq
from itertools import combinations
from typing import Callable

from algorithms.online.streaming_k_center import StreamingKCenter


class DoublingKCenter(StreamingKCenter):

    def __init__(self, k: int, d: Callable[[object, object], float]):

        self.k: int = k
        self.d: Callable[[object, object], float] = d

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
            self._maybe_compact()


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
        threshold = 2 * self.r

        # 1. Alle Kanten <= threshold aus dem Heap entfernen
        edges: list[tuple[float, int, int]] = []
        while self.distances and self.distances[0][0] <= threshold:
            edges.append(heapq.heappop(self.distances))

        # 2. Adjazenzliste nur aus Kanten zwischen (noch) gültigen Zentren
        adjacency: dict[int, list[int]] = {}
        for dist, id_a, id_b in edges:
            if id_a not in self.centers or id_b not in self.centers:
                continue
            adjacency.setdefault(id_a, []).append(id_b)
            adjacency.setdefault(id_b, []).append(id_a)

        # 3. Einmalig über die Zentren iterieren (Snapshot der Keys!)
        for center_id in list(self.centers.keys()):
            for neighbor_id in adjacency.get(center_id, []):
                if center_id in self.centers and neighbor_id in self.centers:
                    self.centers.pop(neighbor_id)


    def _update_stage(self, point: object) -> None:

        if not self.centers:
            self.centers[id(point)] = point
            return

        closest = min(self.centers.values(), key=lambda c: self.d(c, point))

        if self.d(closest, point) > self.alpha * 2 * self.r:
            for c in self.centers.values():
                heapq.heappush(self.distances, (self.d(c, point), id(c), id(point)))
            self.centers[id(point)] = point

    
    def _compact_distances(self) -> None:
        valid_ids = set(self.centers.keys())
        self.distances = [
            entry for entry in self.distances
            if entry[1] in valid_ids and entry[2] in valid_ids
        ]
        heapq.heapify(self.distances)


    def _maybe_compact(self) -> None:
        max_valid = ((self.k + 1) * self.k) // 2  # C(k+1, 2)
        if len(self.distances) > 4 * max_valid:
            self._compact_distances()