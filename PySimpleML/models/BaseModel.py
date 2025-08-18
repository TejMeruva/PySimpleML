from abc import ABC, abstractmethod
import pandas as pd

class MLModel(ABC):
    def train(self, X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs):
        self._train(X, y, *args, **kwargs)

    def predict(self, X:pd.DataFrame):
        return self._predict(X)

    @abstractmethod
    def _train(self, X:pd.DataFrame, y:pd.DataFrame, *args, **kwargs):
        pass

    @abstractmethod
    def _predict(self, X:pd.DataFrame, *args, **kwargs):
        pass
