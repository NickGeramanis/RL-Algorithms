from typing import Tuple

import numpy as np


class Discretizer:
    __n_bins: Tuple[int, ...]
    __bins: np.ndarray

    def __init__(self,
                 n_bins: Tuple[int, ...],
                 state_space_low: np.ndarray,
                 state_space_high: np.ndarray) -> None:
        self.__n_bins = n_bins
        n_dimensions = len(n_bins)

        self.__bins = np.empty((n_dimensions,), dtype=np.ndarray)
        for i in range(n_dimensions):
            self.__bins[i] = np.linspace(state_space_low[i],
                                         state_space_high[i],
                                         num=n_bins[i] + 1)

    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        discrete_state = ()
        for i, bin_ in enumerate(self.__bins):
            discrete_state += (np.digitize(state[i], bin_) - 1,)

        return discrete_state

    @property
    def n_bins(self):
        return self.__n_bins

    def __str__(self) -> str:
        return f"Discretizer: bins = {self.__bins}"
