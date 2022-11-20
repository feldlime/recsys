"""
Rec sys models interface
"""

from typing import Callable, List


class RecModel:
    """Rec model base"""
    model: Callable[[int, int], List[int]]
    weights: List
    _k: int
    _trained: bool = True

    def __init__(self, rec_model: Callable[[int, int], List[int]]) -> None:
        self.model = rec_model  # type: ignore
        self._k = 10

    @property
    def k(self) -> int:
        return self._k

    def train(self) -> None:
        self._trained = True
        self.weights = []

    def predict(self, inlet: int) -> List[int]:
        if not self._trained:
            raise Exception("Model was not trained.")
        return self.model(inlet, self._k)  # type: ignore
