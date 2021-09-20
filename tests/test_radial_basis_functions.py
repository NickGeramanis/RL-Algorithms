import numpy as np
import pytest

from src.features.radial_basis_functions import RadialBasisFunctions


class TestRadialBasisFunctions:
    rbf: RadialBasisFunctions

    @pytest.fixture(autouse=True)
    def init_rbf(self):
        n_actions = 2
        state_space_low = np.array([0, 0])
        state_space_high = np.array([10, 5])
        centers_per_dimension = [
            [0.33, 0.67],
            [0.33, 0.67]
        ]
        standard_deviation = 0.1
        self.rbf = RadialBasisFunctions(n_actions, state_space_low,
                                        state_space_high,
                                        centers_per_dimension,
                                        standard_deviation)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.rbf.get_features(state, 0)
        expected_features = [1,
                             0.7482635675785652,
                             0.024972002042276155,
                             0.0008333973656066949,
                             2.7813195266616742e-05,
                             0, 0, 0, 0, 0]

        assert (np.isclose(expected_features, features)).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(10)])

        q = self.rbf.calculate_q(weights, state)
        expected_q = [0 * 1
                      + 1 * 0.7482635675785652
                      + 2 * 0.024972002042276155
                      + 3 * 0.0008333973656066949
                      + 4 * 2.7813195266616742e-05,
                      5 * 1
                      + 6 * 0.7482635675785652
                      + 7 * 0.024972002042276155
                      + 8 * 0.0008333973656066949
                      + 9 * 2.7813195266616742e-05]

        assert (np.isclose(expected_q, q)).all()
