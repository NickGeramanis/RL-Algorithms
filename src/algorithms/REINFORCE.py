from src.algorithms.rl_algorithm import RLAlgorithm


class REINFORCE(RLAlgorithm):

    def __init__(self) -> None:
        RLAlgorithm.__init__(self)

    def train(self, training_episodes: int) -> None:
        pass

    def run(self, episodes: int, render: bool) -> None:
        pass
