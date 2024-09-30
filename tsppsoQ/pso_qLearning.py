import abc
from dataclasses import dataclass
from random import Random
from tsp import Problem, Solution, pso_q_learning_minimize


class PSO(abc.ABC):
    @abc.abstractmethod
    def minimize(self, problem: Problem) -> Solution:
        pass


@dataclass
class TSPPSO(PSO):
    n: int
    poolsize: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    max_no_improv: int = 3
    rng: Random = None

    def minimize(self, problem: Problem) -> Solution:
        return pso_q_learning_minimize(
            n=self.n,
            poolsize=self.poolsize,
            distances=problem.distances,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            max_no_improv=self.max_no_improv,
            rng=self.rng
        )
