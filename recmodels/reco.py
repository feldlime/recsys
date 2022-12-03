"""
Rec sys models interface
"""

from typing import Any, Iterable, List, Union

import joblib
import pandas as pd
from rectools import Columns


class RecModel:
    """Rec model base"""

    def __init__(
                self,
                model: Union[str, Any] = None,
                dataset: Union[pd.DataFrame, str] = None,
                k_recs: int = 10
                ) -> None:
        if isinstance(model, str):
            self.load(model)
        elif model is not None:
            self.model = model
            self._trained = True
        else:
            self._trained = False

        if dataset is not None:
            self.set_dataset(dataset)
        else:
            self.dataset: pd.DataFrame = None

        self.k: int = k_recs

    def train(self) -> None:
        self._trained = True

    def set_dataset(self, dataset: Union[pd.DataFrame, str]) -> None:
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else:
            try:
                self.dataset = pd.read_csv(dataset)
            except FileNotFoundError:
                raise Exception("Model load error")
            self.dataset.rename(columns={
                                        'user_id': Columns.User,
                                        'item_id': Columns.Item,
                                        'last_watch_dt': Columns.Datetime,
                                        'total_dur': Columns.Weight
                                        },
                                inplace=True)

    def predict(self, inlet: int) -> List[int]:
        if not self._trained:
            raise Exception("Model was not trained.")
        if self.dataset is None:
            raise Exception("Dataset was not loaded.")
        if 'predict' not in dir(self.model):
            raise Exception("Model has no predict method.")
        # try:
        #     user_features = self.dataset[self.dataset[Columns.User] == inlet]
        # except KeyError:
        #     raise Exception(f"Dataset has no user_id {inlet}.")
        return self.model.predict_one(
                                inlet,
                                self.dataset,
                                N_recs=self.k
                                ).tolist()

    def predict_all(self) -> pd.DataFrame:
        if not self._trained:
            raise Exception("Model was not trained.")
        if self.dataset is None:
            raise Exception("Dataset was not loaded.")
        return self.model.predict(
                                self.dataset[Columns.UserItem],
                                self.dataset,
                                N_recs=self.k
                                )

    def load(self, model_path: str) -> None:
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError("Dataset not found")
        self._trained = True
