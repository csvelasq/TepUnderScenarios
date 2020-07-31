from __future__ import division
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import logging


class OptParameters(object):
    """Encapsulator of parameters for pyomo optimization models"""

    def __init__(self, do_log=False, log_filepath="",
                 dump_model=False, dump_model_filepath=None,
                 solver='glpk'):
        """

        :param log_filepath: path of log file of the solver
        :param dump_model_filepath: filepath to which the model will be written if dump_model==True, extension must be *.lp
        :param solver: string specifying the solver
        :param do_log: True if gurobi log is activated, false otherwise
        :param dump_model: True if gurobi model will be written to an output textfile, false otherwise
        """
        self.do_log = do_log
        self.log_filepath = log_filepath
        self.dump_model = dump_model
        self.dump_model_filepath = dump_model_filepath
        self.solver = solver


class OptModel(object):
    """Base class for pyomo optimization models. Objective must be self.model.objective"""

    def __init__(self, model=pyo.ConcreteModel('UnnamedOptModel'),
                 params=OptParameters()):
        self.model = model
        self.params = params
        self.model_is_built = False
        self.optimizer = pyo.SolverFactory(self.params.solver)  # , solver_io='python')
        self.results = None

    def build_model(self):
        self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.model_is_built = True
        if self.params.dump_model:
            self.model.write(filename=self.params.dump_model_filepath)

    def solve(self):
        """Solves this optimization model

        :return: The objective function value if the model was solved to optimality; None otherwise.
        """
        if not self.model_is_built:
            self.build_model()
        self.results = self.optimizer.solve(self.model)
        if not self.is_solved_to_optimality():
            logging.info("Optimization model not solved to optimality, termination condition:", self.results.solver.termination_condition)
            return None
        return self.get_grb_objective_value()

    def is_solved_to_optimality(self):
        return (self.results.solver.status == SolverStatus.ok) and (self.results.solver.termination_condition == TerminationCondition.optimal)

    def get_grb_objective_value(self):
        return pyo.value(self.model.objective)

    def get_grb_vars_solution(self, grb_vars):
        return grb_vars.extract_values()

    def get_grb_constraints_shadow_prices(self, grb_constraints):
        return self.model.dual[grb_constraints]

    def create_grb_abs_constraint_pair(self, lhs_in_abs, rhs, name):
        """Transforms the constraint abs(lhs_in_abs)<=rhs in two constraints: -rhs<=lhs_in_abs<=+rhs"""
        self.model.add_component(name + "_minusAbs", pyo.Constraint(expr=-rhs <= lhs_in_abs))
        self.model.add_component(name + "_plusAbs", pyo.Constraint(expr=+rhs >= lhs_in_abs))
        return self.model.component(name + "_minusAbs"), self.model.component(name + "_plusAbs")
