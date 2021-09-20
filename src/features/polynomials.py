import itertools
from typing import List

import numpy as np

from src.features.feature_constructor import FeatureConstructor


class Polynomials(FeatureConstructor):
    __n_actions: int
    __order: int
    __n_polynomials: int
    __exponents: List
    __n_features: int

    def __init__(self, n_actions: int, order: int, n_dimensions: int) -> None:
        self.__n_actions = n_actions
        self.__order = order
        self.__n_polynomials = (order + 1) ** n_dimensions
        self.__exponents = list(itertools.product(np.arange(order + 1),
                                                  repeat=n_dimensions))
        self.__n_features = self.__n_polynomials * n_actions

    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        for action in range(self.__n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        for i in range(self.__n_polynomials):
            prod_terms = np.power(state, self.__exponents[i])
            features[action * self.__n_polynomials + i] = np.prod(prod_terms)

        return features

    @property
    def n_features(self) -> int:
        return self.__n_features

    def __str__(self) -> str:
        return f"Polynomials: order = {self.__order}"
