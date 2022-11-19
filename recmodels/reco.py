"""
Rec sys models interface
"""

from typing import Callable, List

class RecModel:
    """Rec model base"""
    model: Callable
    weights: List
    _k: int = 10
    __trained: bool = True

    def __init__(self, rec_model: Callable) -> None:
        self.model = rec_model

    @property
    def k(self) -> int:
        return self._k

    def train(self) -> None:
        self.__trained = True

    def predict(self, inlet: List) -> None:
        if self.__trained:
            return self.model(inlet, self._k)
        else:
            raise Exception()

def set_model(func):
    def wrapper():
        return RecModel(func)
    return wrapper