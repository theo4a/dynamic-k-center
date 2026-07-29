from abc import ABC, abstractmethod

class StreamingKCenter(ABC):

    @abstractmethod
    def insert(self, point: object) -> None:
        pass

    @abstractmethod
    def query(self) -> dict:
        pass

