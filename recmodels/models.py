'''
Put your rec model here.
'''
from .reco import RecModel

__all__ = ('test',)

def simple_range(i: int, k: int=10) -> list:
        return list(range(k))

test = RecModel(simple_range)