import numpy as np
import pytest

from src.features.discretizer import Discretizer


class TestDiscretizer:
    discretizer: Discretizer

    @pytest.fixture(autouse=True)
    def init_discretizer(self):
        n_bins = (10, 5)
        state_space_low = np.array([0, 0])
        state_space_high = np.array([10, 5])
        self.discretizer = Discretizer(n_bins, state_space_low,
                                       state_space_high)

    def test_discretize(self):
        state = np.array([3.2, 4.2])

        discrete_state = self.discretizer.discretize(state)

        assert discrete_state == (3, 4)
