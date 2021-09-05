import itertools
from typing import List

import numpy as np
from gym.spaces import Box

from features.feature_constructor import FeatureConstructor


class RBF(FeatureConstructor):
    def __init__(self, n_actions: int, observation_space: Box,
                 centers_per_dimension: List[List[float]],
                 rbf_standard_deviation: float) -> None:
        self.__centers_per_dimension = centers_per_dimension
        self.__rbf_standard_deviation = rbf_standard_deviation
        self.__n_actions = n_actions
        self.__observation_space = observation_space

        self.__rbf_centers = np.array(
            list(itertools.product(*self.__centers_per_dimension)))

        self.__rbf_variance = 2 * self.__rbf_standard_deviation ** 2
        self.__n_functions = self.__rbf_centers.shape[0] + 1

    def calculate_q(self, weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        for action in range(self.__n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))

        state = self.__normalize(state)
        features[action * self.__n_functions] = 1
        for function_i in range(self.__rbf_centers.shape[0]):
            feature_i = action * self.__n_functions + function_i + 1
            features[feature_i] = np.exp(
                - np.linalg.norm(state - self.__rbf_centers[function_i]) ** 2
                / self.__rbf_variance)

        return features

    def __normalize(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.__observation_space.low)
                / (self.__observation_space.high
                   - self.__observation_space.low))

    @property
    def n_features(self):
        return self.__n_functions * self.__n_actions

    @property
    def info(self):
        return (f'Radial Basis Function:'
                f'centers per dimension = {self.__centers_per_dimension},'
                f'standard deviation = {self.__rbf_standard_deviation}')
