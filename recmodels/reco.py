"""
Rec sys models interface
"""

from typing import Callable, List

class RecModel:
    """Rec model base"""
    model: Callable[[int, int], List]
    weights: List
    __trained: bool = True


    def __init__(self, rec_model: Callable[[int, int], List]) -> None:
        self.model = rec_model # type: ignore
        self._k = 10

    @property
    def k(self) -> int:
        return self._k

    def train(self) -> None:
        self.__trained = True
        self.weights = []

    def predict(self, inlet: int) -> List:
        if self.__trained:
            prediction = self.model(inlet, self.k)
            return prediction
        else:
            raise Exception("Model was not trained.")
