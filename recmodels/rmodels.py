"""
Put your rec model here.
"""

from typing import List

from .reco import RecModel


def simple_range(user_id: int, k_recs: int = 10) -> List[int]:
    return list(range(k_recs))


test = RecModel(simple_range)  # don't remove this test model

# You can add models prepared to working in service into this dict
to_prod = {'test': test}
