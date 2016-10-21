import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
import TepScenariosModel
import time
from datetime import timedelta
import logging


def eval_tep_scenarios(individual, tep_model):
    # type: (list, TepScenariosModel.TepScenariosModel) -> List[float]
    plan = TepScenariosModel.StaticTePlan.from_binary_gene(tep_model, individual)
    return plan.total_costs.values()


def population_unique_fitnesses(pop):
    fitnesses_set_tuple = []
    for p in pop:
        if p.fitness not in fitnesses_set_tuple:
            fitnesses_set_tuple.append(p.fitness)
    return set(fitnesses_set_tuple)


def pareto_population_unique_fitnesses(pop):
    return population_unique_fitnesses(tools.sortNondominated(pop, k=2, first_front_only=True)[0])


class TepScenariosNsga2SolverParams(object):
    def __init__(self, number_generations=25, number_individuals=100,
                 crossover_probability=0.9,
                 mutate_probability=0.05):
        self.number_generations = number_generations
        self.number_individuals = number_individuals
        self.crossover_probability = crossover_probability
        self.mutate_probability = mutate_probability


class TepScenariosNsga2Solver(TepScenariosModel.ScenariosTepParetoFrontBuilder):
    def __init__(self, tep_model, ga_params=TepScenariosNsga2SolverParams()):
        # type: (TepScenariosModel.TepScenariosModel) -> object
        TepScenariosModel.ScenariosTepParetoFrontBuilder.__init__(self, tep_model)
        self.ga_params = ga_params
        # Number of scenarios = number of objective functions to minimize
        self.num_scenarios = len(self.tep_model.tep_system.scenarios)
        # Candidates lines
        self.num_candidates_lines = len(self.tep_model.tep_system.candidate_lines)
        # Create problem: fitness and individuals
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.num_scenarios)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        # Attribute generator
        #                      define 'attr_bool' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 'self.num_candidates_lines' 'attr_bool' elements ('genes')
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, self.num_candidates_lines)
        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", eval_tep_scenarios)
        # register the crossover operator
        self.toolbox.register("mate", tools.cxUniform, indpb=0.20)
        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
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

        pop = self.toolbox.population(n=self.ga_params.number_individuals)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        # fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind, self.tep_model)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit
        for ind in invalid_ind:
            ind.fitness.values = self.toolbox.evaluate(ind, self.tep_model)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = self.toolbox.select(pop, len(pop))

        record = stats_fit.compile(pop)
        pareto_pop_fitnesses = None
        # record['pareto_individuals'] = len(pareto_pop_fitnesses)
        # record['elapsed'] = 0
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(self.logbook.stream)

        # Begin the generational process
        for gen in range(1, self.ga_params.number_generations):
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
            # fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind, self.tep_model)
            # for ind, fit in zip(invalid_ind, fitnesses):
            #     ind.fitness.values = fit
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind, self.tep_model)

            # Select the next generation population
            pop = self.toolbox.select(pop + offspring, self.ga_params.number_individuals)

            # Check for convergence
            # TODO use existing front from select
            pareto_pop_prev_fitnesses = pareto_pop_fitnesses
            pareto_pop_fitnesses = pareto_population_unique_fitnesses(pop)
            if pareto_pop_prev_fitnesses == pareto_pop_fitnesses:
                logging.debug("Generations {} and {} have the exact same fitnesses in the pareto front.".
                              format(gen, gen - 1))

            # Report progress
            self.elapsed_seconds = time.clock() - self.start_time
            record = stats_fit.compile(pop)
            record['pareto_individuals'] = len(pareto_pop_fitnesses)
            record['elapsed'] = timedelta(seconds=self.elapsed_seconds)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            logging.info(self.logbook.stream)
            logging.debug("Evaluated generation {} with {} efficient individuals (elapsed={}).".
                          format(gen, len(pareto_pop_fitnesses), timedelta(seconds=self.elapsed_seconds)))
        self.efficient_alternatives = [TepScenariosModel.StaticTePlan.from_binary_gene(self.tep_model, ind)
                                       for ind in tools.sortNondominated(pop, k=2, first_front_only=True)[0]]
        # Set optimal plan for each scenario
        for alternative in self.efficient_alternatives:
            for scenario in self.tep_model.tep_system.scenarios:
                if self.optimal_plans[scenario] is None \
                        or alternative.total_costs[scenario] < self.optimal_plans[scenario].total_costs[scenario]:
                    self.optimal_plans[scenario] = alternative
        return self.efficient_alternatives
