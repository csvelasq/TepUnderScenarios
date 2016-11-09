import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
import TepScenariosModel
import time
from datetime import timedelta
import logging


class TepScenariosNsga2SolverParams(object):
    def __init__(self, number_generations=1, number_individuals=100,
                 crossover_probability=0.9,
                 mutate_probability=0.05):
        self.number_generations = number_generations
        self.number_individuals = number_individuals
        self.crossover_probability = crossover_probability
        self.mutate_probability = mutate_probability


def eval_tep_scenarios(individual, tep_model):
    # type: (list, TepScenariosModel.TepScenariosModel) -> List[float]
    plan = TepScenariosModel.StaticTePlan.from_integer_gene(tep_model, individual)
    return plan.total_costs.values()


def my_init_statictep_individual(container, gene_structure):
    return container([random.randint(0, gene_structure[i]) for i in range(len(gene_structure))])


def my_mut_uniform_int(individual, indpb, gene_structure):
    for i in xrange(len(individual)):
        if random.random() < indpb:
            if gene_structure[i] == 1:
                individual[i] = type(individual[i])(not individual[i])
            else:
                new_gene = random.randint(0, gene_structure[i])
                while new_gene == individual[i]:
                    new_gene = random.randint(0, gene_structure[i])
                individual[i] = new_gene
    return individual,


def get_nondominated(population):
    # k=1 because if k=0 the function returns []; if first_front_only=True, then k is not used
    return tools.sortNondominated(population, k=1, first_front_only=True)[0]


class TepScenariosNsga2Solver(TepScenariosModel.ScenariosTepParetoFrontBuilder):
    def __init__(self, tep_model, ga_params=TepScenariosNsga2SolverParams(),
                 initial_individuals=None):
        # type: (TepScenariosModel.TepScenariosModel) -> object
        TepScenariosModel.ScenariosTepParetoFrontBuilder.__init__(self, tep_model)
        self.ga_params = ga_params
        if initial_individuals is not None:
            assert isinstance(initial_individuals, list)
            for ind in initial_individuals:
                assert isinstance(ind, TepScenariosModel.StaticTePlan)
            self.initial_individuals = initial_individuals
        # Number of scenarios = number of objective functions to minimize
        self.num_scenarios = len(self.tep_model.tep_system.scenarios)
        # A tuple of integers between 0 and 'n' (number of equivalent candidate lines in a given corridor)
        #       For example, if there are 5 corridors, three with one candidate line,
        #       one with two equivalent candidate lines,
        #       and one with three equivalent candidate lines, then: gene_structure=(1,1,2,1,3)
        self.gene_structure = [len(lgroup) for lgroup in self.tep_model.tep_system.candidate_lines_groups]
        # Create problem: fitness and individuals
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.num_scenarios)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        # Structure initializers
        #       Define 'individual' to be an individual consisting of genes
        #       with structure provided by self.gene_structure
        self.toolbox.register("individual", my_init_statictep_individual, creator.Individual, self.gene_structure)
        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.population = None
        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", eval_tep_scenarios)
        # register the crossover operator
        self.toolbox.register("mate", tools.cxUniform, indpb=0.20)
        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", my_mut_uniform_int, indpb=0.05, gene_structure=self.gene_structure)
        # operator for selecting >individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selNSGA2)
        # ----------
        self.logbook = None

    def build_pareto_front(self, seed=None):
        random.seed(seed)
        # Statistics
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
        NUM_DECIMALS = 2
        stats_fit.register("avg", lambda x: np.around(np.mean(x, axis=0), decimals=NUM_DECIMALS))
        stats_fit.register("std", lambda x: np.around(np.std(x, axis=0), decimals=NUM_DECIMALS))
        stats_fit.register("min", lambda x: np.around(np.min(x, axis=0), decimals=NUM_DECIMALS))
        stats_fit.register("max", lambda x: np.around(np.max(x, axis=0), decimals=NUM_DECIMALS))
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "avg", "std", "min", "max", "pareto_individuals", "elapsed"

        # Initialize population
        if self.population is None:
            logging.debug("Initializing random population")
            pop = self.toolbox.population(n=self.ga_params.number_individuals)
        else:
            for ind in self.population:
                assert isinstance(ind, self.creator.Individual)
            if len(self.population) == self.ga_params.number_individuals:
                pop = self.population
            else:
                logging.debug("Population in this instance holds {} individuals but {} individuals are needed.".
                              format(len(self.population), self.ga_params.number_individuals))
        # Add initial individuals provided to this instance
        if self.initial_individuals is not None:
            idx = 0
            for initial_ind in self.initial_individuals:
                initial_ind_gene = creator.Individual(initial_ind.to_integer_gene())
                logging.debug(("Removing individual at position {} of the initial population "
                               "and appending initial individual (with lines '{}' and representation '{}')").
                              format(idx, initial_ind.to_str_repr(), initial_ind_gene)
                              )
                del pop[idx]
                pop.append(initial_ind_gene)
            logging.debug(("A population with {} individuals results "
                           "after replacing some random individuals by provided initial individuals").
                          format(len(pop))
                          )
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = self.toolbox.evaluate(ind, self.tep_model)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = self.toolbox.select(pop, len(pop))

        record = stats_fit.compile(pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

        # Begin the generational process
        for gen in xrange(1, self.ga_params.number_generations + 1):
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(ind1, ind2)
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind, self.tep_model)

            # Select the next generation population
            pop = self.toolbox.select(pop + offspring, self.ga_params.number_individuals)

            # Report progress
            self.population = pop
            self.pareto_pop = get_nondominated(pop)
            self.elapsed_seconds = time.clock() - self.start_time
            record = stats_fit.compile(pop)
            record['pareto_individuals'] = len(self.pareto_pop)
            record['elapsed'] = timedelta(seconds=self.elapsed_seconds)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            logging.info(self.logbook.stream)
            logging.debug("Evaluated generation {} with {} efficient individuals (elapsed={}).".
                          format(gen, len(self.pareto_pop), timedelta(seconds=self.elapsed_seconds)))
        # Select unique individuals in the pareto front, sorted by total cost in first scenario
        self.efficient_alternatives = [TepScenariosModel.StaticTePlan.from_integer_gene(self.tep_model, ind)
                                       for ind in self.pareto_pop]
        self.efficient_alternatives.sort(key=lambda a: a.total_costs[self.tep_model.tep_system.scenarios[0]])
        # Set optimal plan for each scenario
        for alternative in self.efficient_alternatives:
            for scenario in self.tep_model.tep_system.scenarios:
                if self.optimal_plans[scenario] is None \
                        or alternative.total_costs[scenario] < self.optimal_plans[scenario].total_costs[scenario]:
                    self.optimal_plans[scenario] = alternative
        return self.efficient_alternatives

    def i_have_valid_population(self):
        # type: () -> bool
        if self.population is None:
            return False
        else:
            for ind in self.population:
                if not isinstance(ind, self.creator.Individual):
                    return False
            if len(self.population) == self.ga_params.number_individuals:
                return True
        return False
