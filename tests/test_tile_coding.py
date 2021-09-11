import numpy as np
from gym import spaces

from src.features.tile_coding import TileCoding


class TestTileCoding:

    def test_get_features(self):
        low = np.array([0, 0])
        high = np.array([10, 5])
        observation_space = spaces.Box(low=low, high=high, shape=(2,),
                                       dtype=np.float32)
        tiles_per_dimension = [2, 2]
        displacement_vector = [1, 1]
        n_tilings = 2
        n_actions = 2
        tile_coding = TileCoding(n_actions, n_tilings, tiles_per_dimension,
                                 observation_space, displacement_vector)

        state = np.array([3, 2])
        features = tile_coding.get_features(state, 0)
        print(features)
        assert np.all(features == [1, 0, 0,
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
                                   0, 0, 0])

    def test_calculate_q(self):
        low = np.array([0, 0])
        high = np.array([10, 5])
        observation_space = spaces.Box(low=low, high=high, shape=(2,),
                                       dtype=np.float32)
        tiles_per_dimension = [2, 2]
        displacement_vector = [1, 1]
        n_tilings = 2
        n_actions = 2
        tile_coding = TileCoding(n_actions, n_tilings, tiles_per_dimension,
                                 observation_space, displacement_vector)

        state = np.array([3, 2])
        weights = np.array([i for i in range(36)])
        q = tile_coding.calculate_q(weights, state)
        assert np.all(q == [0 + 13, 18 + 31])
