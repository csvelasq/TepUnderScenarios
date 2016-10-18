import logging
import tepmodel as tep
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull


class TriangularProbabilityDistribution(object):
    """Formulas for calculating the pdf and the cdf of a triangular probability distribution"""

    def __init__(self, a=0, b=1, c=0.5):
        self.a = a
        self.b = b
        self.c = c

    def eval_pdf(self, x):
        # from https://en.wikipedia.org/wiki/Triangular_distribution
        if x < self.a:
            return 0
        elif x < self.c:
            return 2 * (x - self.a) / ((self.b - self.a) * (self.c - self.a))
        elif x <= self.b:
            return 2 * (self.b - x) / ((self.b - self.a) * (self.b - self.c))
        return 0

    def eval_cdf(self, x):
        # from https://en.wikipedia.org/wiki/Triangular_distribution
        if x < self.a:
            return 0
        elif x < self.c:
            return (x - self.a) ** 2 / ((self.b - self.a) * (self.c - self.a))
        elif x <= self.b:
            return 1 - (self.b - x) ** 2 / ((self.b - self.a) * (self.b - self.c))
        return 1


class MySimplexForRobustness(object):
    """Represents a facet of the convex pareto front for TEP under scenarios with no probabilities"""

    def __init__(self, scenario_probabilities,
                 vertices_plans, touches_border, neighbour_simplices):
        self.scenario_probabilities = scenario_probabilities
        self.vertices = vertices_plans
        self.touches_border = touches_border  # True if this facet touches a border (i.e. one vertex is in the border, or is an optimal solution for some scenario)
        self.neighbour_simplices = neighbour_simplices


class Robustness2ndOrderMeasure(object):
    """Calculates second order robustness measure for the set of efficient transmission expansion alternatives"""

    def __init__(self, efficient_alternatives):
        # type: (List[TepScenariosModel.StaticTePlan]) -> object
        self.efficient_alternatives = efficient_alternatives
        # Convex hull of pareto front is obtained, in order to efficiently calculate the robustness measure
        self.efficient_points = np.array(list(alt.total_costs.values() for alt in self.efficient_alternatives))
        self.pareto_chull = ConvexHull(self.efficient_points)
        # simplices of the convex hull
        self.simplices = dict()
        simplices_nonpareto_idx = []
        # Create simplices
        for idx, simplex in enumerate(self.pareto_chull.simplices):
            # calculate scenario probabilities based on the normal to each convex simplex
            nd_normal = self.pareto_chull.equations[idx][0:-1]  # delete last component which is the offset
            if np.all(nd_normal < 0):
                nd_normal = np.absolute(nd_normal)
                nd_prob = nd_normal / sum(nd_normal)
                s = MySimplexForRobustness(nd_prob,
                                           list(self.efficient_alternatives[s] for s in simplex),
                                           False, [])
                self.simplices[idx] = s
            else:
                simplices_nonpareto_idx.append(idx)
        # Relate my simplices among them with neighbour data from convex hull
        for idx, neighbors_idxs in enumerate(self.pareto_chull.neighbors):
            if idx in self.simplices.keys():
                for neighbor_idx in [s for s in neighbors_idxs]:
                    if neighbor_idx in simplices_nonpareto_idx:
                        self.simplices[idx].touches_border = True
                    else:
                        self.simplices[idx].neighbour_simplices.append(self.simplices[neighbor_idx])
        # Relate expansion plans with simplices and calculate robustness measure
        self.plans_with_robustness = []
        for plan in self.efficient_alternatives:
            plan_simplices = list(s for s in self.simplices.itervalues() if plan in s.vertices)
            is_in_border = len(plan_simplices) < len(plan.tep_model.tep_system.scenarios)
            plan_with_robustness = StaticTePlanForRobustnessCalculation(plan, plan_simplices, is_in_border)
            self.plans_with_robustness.append(plan_with_robustness)


class StaticTePlanForRobustnessCalculation(object):
    """Calculates the robustness measure for one particular transmission expansion plan,
    based on convex hull information on the pareto front of transmission expansion alternatives"""

    def __init__(self, plan, simplices, is_in_border):
        # type: (tepmodel.StaticTePlan, object, object) -> object
        self.plan = plan
        self.plan_id = self.plan.get_plan_id()
        self.simplices = simplices
        self.is_in_border = is_in_border
        self.robustness_measure = float('nan')
        if not self.is_in_border:
            if len(self.simplices) == 2:
                probabilities_first_scenario = sorted(list(s.scenario_probabilities[0] for s in self.simplices))
                logging.info("Range of probabilities for plan {}: {}".format(self.plan_id,
                                                                             probabilities_first_scenario)
                             )
                # Calculates robustness as a probability of optimality given by the integral of second order pdf,
                # which can be expressed in closed form in this case (2-d and triangular pdf)
                my_prob_distr = TriangularProbabilityDistribution()
                self.robustness_measure = my_prob_distr.eval_cdf(probabilities_first_scenario[1]) \
                                          - my_prob_distr.eval_cdf(probabilities_first_scenario[0])
                logging.info("Robustness measure for plan {}: {}".format(self.plan_id, self.robustness_measure))
            else:
                logging.warning(("Robustness measure cannot (yet) be calculated "
                                 "for more than two scenarios (or two simplices per plan), "
                                 "but plan {} has {} simplices").format(self.plan_id,
                                                                        len(self.simplices)
                                                                        )
                                )
        else:
            # TODO identify the border touched by bordered expansion alternatives and calculate robustness measure for these alternatives also
            logging.warning(("Robustness measure cannot (yet) be calculated "
                             "for plans in the borders of the trade-off curve, "
                             "such as plan {}").format(self.plan_id)
                            )
