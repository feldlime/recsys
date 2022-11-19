from .reco import set_model

@set_model
def simple_range(i, k) -> list:
    return list(range(k))

