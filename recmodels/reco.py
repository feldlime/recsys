"""
Rec sys models interface
"""

from typing import Callable, List

class RecModel:
    """Rec model base"""
    model: Callable
    weights: List
    __trained: bool = True
    

    def __init__(self, rec_model: Callable) -> None:
        self.model = rec_model
        self._k = 10

    @property
    def k(self):
        return self._k

    def train(self) -> None:
        self.__trained = True
        self.weights = []

    def predict(self, inlet: int) -> None:
        if self.__trained:
            return self.model(inlet, self.k)
        else:
            raise Exception("Model was not trained.")

def set_model(func):
    def wrapper():
        return RecModel(func)
    return wrapper