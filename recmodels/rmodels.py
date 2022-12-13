import os
import sys
from typing import List

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset

from recmodels.reco import RecModel

sys.path.append(os.path.join(os.path.dirname("./recmodels"), "reco"))
sys.path.append(os.path.join(os.path.dirname("./data/models"), "models"))

DATASET_PATH = r"./data/datasets/interactions_processed.csv"


class TestModel(RecModel):
    """Using for testing. Returns first k_recs items"""

    def predict(self, user_id: int, k_recs: int = None) -> List[int]:
        if k_recs is None:
            k_recs = self.k
        return list(range(k_recs))


class LightFMModel(RecModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset: pd.DataFrame = self.dataset
        self.dataset.rename(
            columns={
                "user_id": Columns.User,
                "item_id": Columns.Item,
                "last_watch_dt": Columns.Datetime,
                "watched_pct": Columns.Weight,
            },
            inplace=True,
        )
        self.dataset["datetime"] = pd.to_datetime(self.dataset["datetime"])
        self.dataset = Dataset.construct(interactions_df=self.dataset)

    def predict(self, user_id: int, k_recs: int = None) -> List[int]:
        try:
            res = getattr(self.model, self.predict_method)(
                users=[user_id], k=k_recs, filter_viewed=True, dataset=self.dataset
            )["item_id"].tolist()
        except KeyError:
            res = self.get_popular_recs(k_recs)
        return res


def load_model(model_name: str) -> RecModel:
    if model_name == "test":
        return TestModel()

    if model_name == "hw2":
        return RecModel(
            model=r"./data/models/userknn_1W.joblib",
            dataset=DATASET_PATH,
        )

    if model_name == "hw3":
        return LightFMModel(
            model=r"./data/models/LightFM_warp_16-2.joblib",
            dataset=DATASET_PATH,
            method="recommend",
        )

    raise ValueError("Model name is not valid")


if __name__ == "__main__":
    print("Testing models:")
    available_models = ["test", "hw2", "hw3"]
    users = [1, 428, 2222, 768, 63443, 25442]
    k = 15
    for model in available_models:
        print(f"{model}:")
        current_model = load_model(model)
        for user in users:
            print(f"\t{user}:\t", current_model.predict(user, k))
