from pso_qLearning import TSPPSO
from tsp import Problem,Solution,plot_solution_path,pso_q_learning_minimize
from datatypes import Position
import os
from datasets import TSPDataset


os.chdir('./tsppsoQ')



def berlin52():
    from datasets import TSPDataset
    dataset = TSPDataset()
    dataset.read('../datasets/berlin52.tsp')
    dataset.unique(eps=1e-4, inplace=True)
    print('dataset size:', len(dataset.positions))

    import random
    seed = random.randint(0, 999999)
    rng = random.Random(seed)
    print('seed used:', seed)
    
    # 7544.365901904087 - 328015

    problem = Problem(xy=dataset.positions)
    optimizer = TSPPSO(100, 100, p1=0.95, p2=0.03, p3=0.02, max_no_improv=3, rng=rng)
    solution = optimizer.minimize(problem)

    print()
    print('seq:\n{}\n\ncost:\n{}'.format(solution.sequence, solution.cost))
    plot_solution_path(problem,solution)


def custom_data():

    dataset_folder = '../datasets/stdsets'
    results_folder = '../results/withQlearning_on_standardDatasets'
    # Dataset = r"C:\Users\Lenovo\Desktop\practice+projex\swarmRL\pso\tsp-pso\datasets\cities_100_f.tsp"
  
    for file in os.listdir(dataset_folder):

     
    
        file_name = file.split('.')[0]
        
        file_path = os.path.join(dataset_folder, file)

    
        dataset = TSPDataset()
        dataset.read(file_path)
        dataset.unique(eps=1e-4, inplace=True)
        print('dataset size:', len(dataset.positions))

        import random
        seed = random.randint(0, 999999)
        rng = random.Random(seed)
        print('seed used:', seed)   


        problem = Problem(xy=dataset.positions)
      
        
    

        

        optimizer = TSPPSO(
                n=100,
                poolsize=100,
                alpha=0.1,
                gamma=0.9,
                epsilon=0.1,
                max_no_improv=3,
                rng=rng
            )

            # Run the optimization
        solution, convergence_history = optimizer.minimize(problem)
        print()
        print('seq:\n{}\n\ncost:\n{}'.format(solution.sequence, solution.cost))
        plot_solution_path(problem,solution)
     
        #plot_convergence(convergence_data)
        

        result_file_path = os.path.join(results_folder, f'{file_name}.txt')
        with open(result_file_path, 'a') as result_file:  
            result_file.write(f'{solution.cost}, {solution.sequence}\n')


    '''code to plot poolsize vs cost'''
    # dataset = TSPDataset()
    # dataset.read(Dataset)
    # dataset.unique(eps=1e-4, inplace=True)
    # print('dataset size:', len(dataset.positions))

    # import random
    # seed = random.randint(0, 999999)
    # rng = random.Random(seed)
    # print('seed used:', seed)


    # problem = Problem(xy=dataset.positions)

    # sizes = [25,50,75,100]
    # to_plot = []
    # for poolsize in sizes:
    #     optimizer = TSPPSO(100, poolsize, p1=0.95, p2=0.03, p3=0.02, max_no_improv=3, rng=rng)

    #     solution,convergence_data = optimizer.minimize(problem)
    #     to_plot.append(solution)
    # plot_poolsize_vs_cost(to_plot)
    
               


                


def zi929():
    from datasets import TSPDataset
    dataset = TSPDataset()
    dataset.read('../datasets/zi929.tsp')
    dataset.unique(eps=1e-4, inplace=True)
    print('dataset size:', len(dataset.positions))

    import random
    rng = random.Random()

    problem = Problem(xy=dataset.positions)
    optimizer = TSPPSO(1000, 20, max_no_improv1=1000, max_no_improv2=5, rng=rng)
    solution = optimizer.minimize(problem)

    print()
    print('seq:\n{}\n\ncost:\n{}'.format(solution.sequence, solution.cost))


def lu634():
    from datasets import TSPDataset
    dataset = TSPDataset()
    dataset.read('../datasets/lu634.tsp')
    dataset.unique(eps=1e-4, inplace=True)
    print('dataset size:', len(dataset.positions))

    import random
    rng = random.Random(97855)

    problem = Problem(xy=dataset.positions)
    optimizer = TSPPSO(1000, 40, max_no_improv1=20, max_no_improv2=5, rng=rng)
    solution = optimizer.minimize(problem)

    print()
    print('seq:\n{}\n\ncost:\n{}'.format(solution.sequence, solution.cost))


def canada():
    from datasets import TSPDataset
    dataset = TSPDataset()
    dataset.read('../datasets/canada.tsp')
    dataset.unique(inplace=True)
    print('dataset size:', len(dataset.positions))

    import random
    rng = random.Random(97855)

    problem = Problem(xy=dataset.positions)
    optimizer = TSPPSO(200, 20, max_no_improv1=10, rng=rng)
    solution = optimizer.minimize(problem)

    print()
    print('seq:\n{}\n\ncost:\n{}'.format(solution.sequence, solution.cost))


def test():
    def brute_force_tsp(problem: Problem):
        import sys
        from itertools import permutations
        from tsp import evaluate_cost
        min_seq = None
        min_cost = sys.float_info.max
        base_seq = list(range(len(problem.distances)))
        for seq in permutations(base_seq):
            seq = list(seq)
            cost = evaluate_cost(seq, problem.distances)
            if cost <= min_cost:
                min_cost = cost
                min_seq = seq
        return min_seq, min_cost

    positions = [
        (2, 9),  # A
        (8, 9),  # B
        (3, 7),  # C
        (1, 6),  # D
        (7, 6),  # E
        (4, 4),  # F
        (7, 1),  # G
    ]

    problem = Problem(xy=positions)

    seq, cost = brute_force_tsp(problem)
    print("seq:", list(seq), "cost:", cost)

    import random
    seed = random.randint(0, 100000)
    print("seed:", seed)

    optimizer = TSPPSO(100, 30, max_no_improv1=10, rng=random.Random(seed))
    solution = optimizer.minimize(problem)
    


if __name__ == "__main__":
    # test()
    # canada()
    # lu634()
    # zi929()
    custom_data()
