import gym
import numpy as np

from features.tile_coding import TileCoding


class TestTileCoding:

    def test_get_features(self):
        env_name = 'MountainCar-v0'
        env = gym.make(env_name)
        tiles_per_dimension = [3, 3]
        displacement_vector = [1, 1]
        n_tilings = 2
        n_actions = 2
        tile_coding = TileCoding(
            n_actions, n_tilings, tiles_per_dimension,
            env.observation_space, displacement_vector)

        state = env.reset()
        action = 0
        features = tile_coding.get_features(state, action)
        expected_features = np.array(
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        assert np.array_equal(expected_features, features)
