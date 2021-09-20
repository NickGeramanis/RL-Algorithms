import numpy as np
import pytest

from src.features.tile_coding import TileCoding


class TestTileCoding:
    tile_coding: TileCoding

    @pytest.fixture(autouse=True)
    def init_tile_coding(self):
        n_tiles_per_dimension = [2, 2]
        displacement_vector = np.array([1, 1])
        n_tilings = 2
        n_actions = 2
        state_space_low = np.array([0, 0])
        state_space_high = np.array([10, 5])
        self.tile_coding = TileCoding(n_actions, n_tilings,
                                      n_tiles_per_dimension, state_space_low,
                                      state_space_high, displacement_vector)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.tile_coding.get_features(state, 0)
        expected_features = [1, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 1, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0]

        assert (features == expected_features).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(36)])

        q = self.tile_coding.calculate_q(weights, state)
        expected_q = [0 + 13, 18 + 31]

        assert (q == expected_q).all()
