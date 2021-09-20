from abc import ABC, abstractmethod

from numpy import ndarray


class FeatureConstructor(ABC):

    @abstractmethod
    def calculate_q(self, weights: ndarray, state: ndarray) -> ndarray:
        pass

    @abstractmethod
    def get_features(self, state: ndarray, action: int) -> ndarray:
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        pass
