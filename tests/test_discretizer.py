import gym

from features.discretizer import Discretizer


class TestDiscretizer:

    def test_get_state(self):
        env_name = 'MountainCar-v0'
        env = gym.make(env_name)
        n_bins = (4, 19)
        discretizer = Discretizer(n_bins, env.observation_space)

        observation = env.reset()
        state = discretizer.get_state(observation)
        assert state == (1, 9)
