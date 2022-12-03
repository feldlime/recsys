"""
Put your rec model here.
"""

from typing import List

from recmodels.reco import RecModel

loaded_model = None


def load_model(model_path: str, dataset_path: str) -> RecModel:
    global loaded_model
    loaded_model = RecModel(model_path, dataset_path)
    return loaded_model


def get_predictions(user_id: int, k_recs: int = 10) -> List[int]:
    loaded_model.k = k_recs
    return loaded_model.predict(user_id)


if __name__ == "__main__":
    model_path: str = r"./data/models/userknn_tined.joblib"
    dataset_path: str = r"./data/datasets/interactions_processed.csv"
    print('Loading model and dataset...')
    load_model(model_path, dataset_path)
    print("Predicting...")
    print(get_predictions(699317))
