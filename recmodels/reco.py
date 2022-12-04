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
        k_recs: int = 10,
    ) -> None:
        """
        :param model: path to model or model itself
        :param dataset: path to dataset or dataset itself
        :param k_recs: number of recommendations
        """
        if isinstance(model, str):
            self.set_model(model)
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

    def set_model(self, model_path: str) -> None:
        """
        Set model for prediction
        :param model_path: path to model
        """
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError("Model file not found")
        self._trained = True

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

    def predict(
        self, inlet: Union[int, pd.DataFrame], predict_method: str = "predict"
    ) -> List[int]:
        """
        Predict recommendations
        :param inlet: user_id or dataframe with user_id
        :param predict_method: method for prediction
        """
        self._check_model()
        self._check_dataset()
        self._check_method(predict_method)
        return getattr(self.model, predict_method)(inlet, N_recs=self.k)

    def _check_model(self) -> None:
        """
        Check if model is loaded
        """
        if not self._trained:
            raise Exception("Model was not trained.")

    def _check_dataset(self) -> None:
        """
        Check if dataset is loaded and has user_id and item_id columns
        """
        if self.dataset is None:
            raise Exception("Dataset was not loaded.")
        if not set(["user_id", "item_id"]).issubset(self.dataset.columns):
            raise Exception("Dataset has no user_id or item_id columns.")

    def _check_method(self, method: str) -> None:
        """
        Check if predict method exists
        :param predict_method: method for prediction
        """
        if method not in dir(self.model):
            raise Exception(f"Model has no this method: {method}.")
