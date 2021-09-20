from src.rl_algorithms.rl_algorithm import RLAlgorithm


class ActorCriticEligibilityTraces(RLAlgorithm):

    def __init__(self) -> None:
        RLAlgorithm.__init__(self, 'info.log')

    def train(self, n_episodes: int) -> None:
        raise NotImplementedError

    def run(self, n_episodes: int, render: bool = False) -> None:
        raise NotImplementedError
