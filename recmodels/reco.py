"""
Rec sys models interface
"""

from typing import Callable, List


class RecModel:
    """Rec model base"""

    def __init__(self, rec_model: Callable[[int, int], List[int]]) -> None:
        self.model = rec_model
        self._trained = True
        self.k = 10

    def train(self) -> None:
        self._trained = True

    def predict(self, inlet: int) -> List[int]:
        if not self._trained:
            raise Exception("Model was not trained.")
        return self.model(inlet, self.k)
