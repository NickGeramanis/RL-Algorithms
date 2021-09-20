import itertools
import math
from typing import List

import numpy as np

from src.features.feature_constructor import FeatureConstructor


class FourierBasis(FeatureConstructor):
    __n_actions: int
    __order: int
    __state_space_low: np.ndarray
    __state_space_high: np.ndarray
    __n_functions: int
    __integer_vectors: List
    __n_features: int

    def __init__(self, n_actions: int, order: int, state_space_low: np.ndarray,
                 state_space_high: np.ndarray) -> None:
        self.__n_actions = n_actions
        self.__order = order
        self.__state_space_low = state_space_low
        self.__state_space_high = state_space_high
        n_dimensions = len(state_space_high)
        self.__n_functions = (order + 1) ** n_dimensions
        self.__integer_vectors = list(itertools.product(np.arange(order + 1),
                                                        repeat=n_dimensions))
        self.__n_features = self.__n_functions * n_actions

    def calculate_q(self, weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        for action in range(self.__n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        norm_state = self.__normalize(state)
        for i in range(self.__n_functions):
            cos_term = math.pi * np.dot(norm_state, self.__integer_vectors[i])
            features[action * self.__n_functions + i] = math.cos(cos_term)

        return features

    def __normalize(self, value: np.ndarray) -> np.ndarray:
        numerator = value - self.__state_space_low
        denominator = self.__state_space_high - self.__state_space_low
        return numerator / denominator

    @property
    def n_features(self) -> int:
        return self.__n_features

    def __str__(self) -> str:
        return f'Fourier Basis: order = {self.__order}'
