import math

import numpy as np
import pytest

from src.features.fourier_basis import FourierBasis


class TestFourierBasis:
    fourier_basis: FourierBasis

    @pytest.fixture(autouse=True)
    def init_fourier_basis(self):
        order = 1
        n_actions = 2
        state_space_low = np.array([0, 0])
        state_space_high = np.array([10, 5])
        self.fourier_basis = FourierBasis(n_actions, order, state_space_low,
                                          state_space_high)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.fourier_basis.get_features(state, 0)
        expected_features = [math.cos(0),
                             math.cos(0.4 * math.pi),
                             math.cos(0.3 * math.pi),
                             math.cos(0.3 * math.pi + 0.4 * math.pi),
                             0, 0, 0, 0]

        assert (np.isclose(expected_features, features)).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(8)])

        q = self.fourier_basis.calculate_q(weights, state)
        expected_q = [0 * math.cos(0)
                      + 1 * math.cos(0.4 * math.pi)
                      + 2 * math.cos(0.3 * math.pi)
                      + 3 * math.cos(0.3 * math.pi + 0.4 * math.pi),
                      4 * math.cos(0)
                      + 5 * math.cos(0.4 * math.pi)
                      + 6 * math.cos(0.3 * math.pi)
                      + 7 * math.cos(0.3 * math.pi + 0.4 * math.pi)]

        assert (np.isclose(expected_q, q)).all()
