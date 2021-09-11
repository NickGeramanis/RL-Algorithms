import numpy as np
from gym import spaces

from src.features.discretizer import Discretizer


class TestDiscretizer:

    def test_get_state(self):
        low = np.array([0, 0])
        high = np.array([10, 5])
        observation_space = spaces.Box(low=low, high=high, shape=(2,),
                                       dtype=np.float32)
        n_bins = (10, 5)
        discretizer = Discretizer(n_bins, observation_space)

        observation = np.array([3.2, 4.2])
        state = discretizer.get_state(observation)
        assert state == (3, 4)
