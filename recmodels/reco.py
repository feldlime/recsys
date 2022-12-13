"""
Rec sys models interface
"""

from typing import Any, List, Union

import joblib
import pandas as pd


class RecModel:
    """
    Rec sys models interface
    """

    def __init__(
        self,
        model: Union[str, Any] = None,
        dataset: Union[pd.DataFrame, str] = None,
        method: str = "predict",
        k_recs: int = 10,
    ) -> None:
        """
        :param model: path to model or model itself
        :param dataset: path to dataset or dataset itself
        :param k_recs: number of recommendations
        """
        self.popular_recs: List[int] = None
        if isinstance(model, str):
            self.set_model(model)
        elif model is not None:
            self.model = model
            self.model_loaded = True
        else:
            self.model_loaded = False

        if dataset is not None:
            self.set_dataset(dataset)

        self.set_predict_method(method)

        self.k: int = k_recs

    def set_model(self, model_path: str) -> None:
        """
        Set model for prediction
        :param model_path: path to model
        """
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found")
        self.model_loaded = True

    def set_dataset(self, dataset: Union[pd.DataFrame, str]) -> None:
        """
        Set dataset for model
        :param dataset: path to dataset or dataset itself
        """
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else:
            try:
                self.dataset = pd.read_csv(dataset)
            except FileNotFoundError:
                raise Exception("Dataset file not found")
        self._check_dataset()
        self.calculate_popular_recs()

    def set_predict_method(self, method: str) -> None:
        """
        Set method for prediction
        :param method: method name
        """
        if self.model_loaded:
            self._check_method(method)
        self.predict_method = method

    def calculate_popular_recs(self) -> None:
        self._check_dataset()
        self.popular_recs = (
            self.dataset.groupby("item_id")
            .count()
            .sort_values(by="user_id", ascending=False)
            .index.to_list()
        )

    def get_popular_recs(self, k: int = 10) -> List[int]:
        """
        Get popular recommendations
        :param k: number of recommendations
        """
        if self.popular_recs is None:
            try:
                self.calculate_popular_recs()
            except Exception as e:
                raise Exception("Can't calculate popular recommendations") from e
        return list(self.popular_recs)[:k]

    def predict(
        self,
        inlet: int,
        k_recs: int,
    ) -> List[int]:
        """
        Predict recommendations
        :param inlet: user_id
        :param predict_method: method for prediction
        """
        if not self.model_loaded:
            raise Exception("Model is not loaded.")
        return getattr(self.model, self.predict_method)(inlet, k_recs)

    def _check_dataset(self) -> None:
        """
        Check if dataset is loaded and has user_id and item_id columns
        """
        if not isinstance(self.dataset, pd.DataFrame):
            raise Exception("Dataset is not a pandas DataFrame.")
        if not set(["user_id", "item_id"]).issubset(self.dataset.columns):
            raise Exception("Dataset has no user_id or item_id columns.")

    def _check_method(self, method: str) -> None:
        """
        Check if predict method exists
        :param predict_method: method for prediction
        """
        if method not in dir(self.model):
            raise Exception(f"Model has no this method: {method}.")
        if not callable(getattr(self.model, method)):
            raise Exception(f"Method {method} is not callable.")
