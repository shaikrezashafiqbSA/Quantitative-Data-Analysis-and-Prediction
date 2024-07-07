import pygad
from backtesters.get_trading_decision import run_backtest
from optimization.parameter_construction import organize_params, build_params

# Define the fitness function
def fitness_func(solution, solution_idx):
    # Convert the solution array to a dictionary of parameters
    data_params_payload = dict(zip(continuous_values_dict.keys(), solution))
    
    # Run the backtest and get the cost
    cost = run_backtest(data_params_payload)
    
    # Since PyGAD tries to maximize the fitness, we return the negative cost
    return -cost

# Define the number of generations, the number of individuals in the population, 
# and the number of parameters to optimize
num_generations = 100
num_individuals = 50
num_parameters = len(continuous_values_dict)

# Create an instance of the GA class
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=num_individuals,
    num_genes=num_parameters,
    init_range_low=-1.0,
    init_range_high=1.0,
    parent_selection_type="sss",
    keep_parents=1,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

# Run the genetic algorithm
ga_instance.run()
