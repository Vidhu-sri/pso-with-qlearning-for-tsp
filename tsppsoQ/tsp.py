from dataclasses import dataclass
from functools import total_ordering
from random import Random
from typing import Union, Tuple, List, Iterable, Sequence, Dict

from datatypes import FloatMatrix, Position
from utilities import distances_matrix, lshift, neighborhood_inversion

class Problem(object):
    def __init__(
            self,
            x: Sequence[float] = None,
            y: Sequence[float] = None,
            xy: Iterable[Union[Position, Tuple[float, float]]] = None
    ):
        self.__positions: List[Position]
        self.__distances: FloatMatrix

        if x is not None or y is not None:
            assert len(x) == len(y), "x and y sequences must be of the same length."
            self.__positions = [Position(xcoord, ycoord) for xcoord, ycoord in zip(x, y)]
        elif xy is not None:
            self.__positions = [Position(xcoord, ycoord) for xcoord, ycoord in xy]
        else:
            raise ValueError('Either xy or x and y arguments must not be None.')

        # Pre-compute distances matrix between points.
        self.__distances = distances_matrix(self.__positions)

    @property
    def positions(self) -> List[Position]:
        return self.__positions

    @property
    def distances(self) -> FloatMatrix:
        return self.__distances


@dataclass
@total_ordering
class Solution(object):
    sequence: List[int]
    cost: float
    best_sequence: List[int]
    best_cost: float

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost


def pso_q_learning_minimize(
        n: int,
        poolsize: int,
        distances: FloatMatrix,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        max_no_improv: int = 3,
        rng: Random = None
):
    rng = Random() if rng is None else rng

    # Initialize pool of solutions.
    solutions = list()
    base_indices = list(range(len(distances)))
    for i in range(poolsize):
        solution_indices = base_indices.copy()
        rng.shuffle(solution_indices)
        solution_cost = evaluate_cost(solution_indices, distances)
        solution = create_solution(solution_indices, solution_cost)
        solutions.append(solution)

    global_solution_index = solutions.index(min(solutions))
    global_solution = copy_solution(solutions[global_solution_index])

    counter = 0

    convergence = []

    # Initialize Q-table: Each particle has its own Q-table
    # States: Difference in cost between current cost and personal best and global best
    # Actions: 0 (independent), 1 (personal best), 2 (global best)
    q_tables = [{} for _ in range(poolsize)]

    while n > 0:
        print('Iteration:', counter, 'Global Best Cost:', global_solution.cost)

        convergence.append(global_solution.cost)

        counter += 1

        for i, solution in enumerate(solutions):
       
            # state can be categorized based on cost difference
            

            cost_diff_personal = (solution.cost - solution.best_cost) / solution.best_cost
            cost_diff_global = (solution.cost - global_solution.cost) / global_solution.cost


            state = (categorize_relative_cost_diff(cost_diff_personal), categorize_relative_cost_diff(cost_diff_global))

            #  ε-greedy policy
            q_table = q_tables[i]
            if rng.random() < epsilon:
                action = rng.choice([0, 1, 2])  # Explore
            else:
                action = select_best_action(q_table, state)

            # Save current cost for reward calculation
            prev_cost = solution.cost

           
            if action == 0:  # move independently
                move_solution_independently(solution, distances, max_no_improv, rng)
            elif action == 1:  # move toward personal best
                move_solution_to_personal_best(solution, distances)
            else:  # move toward swarm best
                move_solution_to_swarm_best(solution, global_solution, distances)

            #reward
            reward = -(solution.cost - prev_cost)

            if solution.cost < solution.best_cost:
                solution.best_sequence = solution.sequence.copy()
                solution.best_cost = solution.cost

      
            next_state = (categorize_relative_cost_diff(solution.cost - solution.best_cost),
                          categorize_relative_cost_diff(solution.cost - global_solution.cost))

            update_q_table(q_table, state, action, reward, next_state,alpha,gamma)

        # Update global best solution
        global_solution_index = solutions.index(min(solutions))
        copy_solution_to(solutions[global_solution_index], global_solution)

        n -= 1

    return global_solution, convergence


def categorize_relative_cost_diff(cost_diff):
    if cost_diff > 0.05:
        return 'worse'
    elif cost_diff < -0.05:
        return 'better'
    else:
        return 'similar'



def select_best_action(q_table: Dict, state):

    state_actions = q_table.get(state, {})
    if not state_actions:
        return 0  
    max_q = max(state_actions.values())
    max_actions = [action for action, q_value in state_actions.items() if q_value == max_q]
    return max_actions[0]  # return the first action with max Q-value


def update_q_table(q_table: Dict, state, action, reward, next_state, alpha, gamma):
    # Update Q-value using the Q-learning update rule
    state_actions = q_table.setdefault(state, {})
    q_value = state_actions.get(action, 0)

    # Get max Q-value for the next state
    next_state_actions = q_table.get(next_state, {})
    if next_state_actions:
        max_next_q_value = max(next_state_actions.values())
    else:
        max_next_q_value = 0

    new_q_value = alpha*q_value + (1-alpha) * (reward + gamma * max_next_q_value - q_value)
    state_actions[action] = new_q_value



def move_solution_independently(solution: Solution, distances: FloatMatrix, max_no_improv: int, rng: Random):
    sequence, delta_cost = neighborhood_inversion_search(solution.sequence, distances, max_no_improv, rng)
    solution.sequence = sequence
    solution.cost += delta_cost


def move_solution_to_personal_best(solution: Solution, distances: FloatMatrix):
    sequence, cost = path_relinking_search(
        solution.sequence, solution.best_sequence, solution.best_cost, distances)
    solution.sequence = sequence
    solution.cost = cost


def move_solution_to_swarm_best(solution: Solution, swarm_solution: Solution, distances: FloatMatrix):
    sequence, cost = path_relinking_search(
        solution.sequence, swarm_solution.sequence, swarm_solution.cost, distances)
    solution.sequence = sequence
    solution.cost = cost


def evaluate_cost(seq: List[int], distances: FloatMatrix) -> float:
    cost = 0
    n = len(seq)
    for i in range(1, n):
        cost += distances[seq[i - 1]][seq[i]]
    return cost + distances[seq[-1]][seq[0]]


def path_relinking_search(
        origin: List[int],
        target: List[int],
        target_cost: float,
        distances: FloatMatrix
) -> Tuple[List[int], float]:

    best_seq = target.copy()
    best_cost = target_cost

    target_value = target[0]
    target_index = origin.index(target_value)
    seq = lshift(origin, target_index)

    n = len(target)
    for i in range(1, n - 1):
        target_value = target[i]
        right_seq = seq[i:]
        target_index = right_seq.index(target_value)  # target element index that is used as shifting distance.
        seq[i:] = lshift(right_seq, target_index)

        cost = evaluate_cost(seq, distances)
        if cost < best_cost:
            best_seq = seq.copy()
            best_cost = cost

    return best_seq, best_cost


def neighborhood_inversion_search(
        seq: List[int],
        distances: FloatMatrix,
        max_no_improv: int,
        rng: Random = None
) -> Tuple[List[int], float]:
    rng = Random() if rng is None else rng

    best_delta_cost = 0
    best_i = 0
    best_j = 0

    n = len(seq)  # sequence size.
    m = 2  # neighborhood size.
    no_improv_count = 0  # number of iterations with no improvement for current neighborhood size.

    while n - m > 1:
        # Generate neighborhood range [i, j].
        i = rng.randint(0, n - 1)
        j = i + m - 1
        j = j if j < n else j - n

        ia = seq[i - 1]
        ib = seq[i]
        ja = seq[j]
        jb = seq[j + 1 if j + 1 < n else 0]

        cost0 = distances[ia][ib] + distances[ja][jb]
        cost1 = distances[ia][ja] + distances[ib][jb]
        delta_cost = cost1 - cost0

        if delta_cost < best_delta_cost:
            best_delta_cost = delta_cost
            best_i = i
            best_j = j
            m += 1
            no_improv_count = 0
        else:
            no_improv_count += 1
            if no_improv_count >= max_no_improv:
                m += 1
                no_improv_count = 0

    result_seq = neighborhood_inversion(seq, best_i, best_j)
    return result_seq, best_delta_cost


def create_solution(sequence: List[int], cost: float) -> Solution:
    return Solution(sequence.copy(), cost, sequence.copy(), cost)


def copy_solution(solution: Solution) -> Solution:
    return Solution(solution.sequence.copy(), solution.cost, solution.best_sequence.copy(), solution.best_cost)


def copy_solution_to(src: Solution, dst: Solution):
    dst.sequence = src.sequence.copy()
    dst.cost = src.cost
    dst.best_sequence = src.best_sequence.copy()
    dst.best_cost = src.best_cost

import matplotlib.pyplot as plt
from typing import List

def plot_solution_path(problem: Problem, solution: Solution):
    # Extract the positions from the problem using the sequence in the solution
    positions = problem.positions
    sequence = solution.sequence

    # Create lists of x and y coordinates in the order of the sequence
    x_coords = [positions[i].x for i in sequence] + [positions[sequence[0]].x]  # to complete the loop
    y_coords = [positions[i].y for i in sequence] + [positions[sequence[0]].y]  # to complete the loop

    # Plotting the path
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')

    # Annotate the points
    for i, pos in enumerate(positions):
        plt.text(pos.x, pos.y, f'{i}', fontsize=12, ha='right')

    # Plot start point
    plt.plot(x_coords[0], y_coords[0], 'ro')  # Red dot for the start point

    # Labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Solution Path')

    plt.grid(True)
    plt.show()

def plot_convergence(convergence_history: List[float]):
    # Plotting the convergence history
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(convergence_history)), convergence_history, marker='o', linestyle='-', color='r')

    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Cost')
    plt.title('Convergence Graph')

    plt.grid(True)
    plt.show()
