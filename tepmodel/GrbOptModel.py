import gurobipy as grb
import logging


class GrbOptParameters(object):
    """Encapsulator of parameters for gurobi optimization models"""

    def __init__(self, do_log=False, log_filepath="",
                 dump_model=False, dump_model_filepath=None):
        """

        :param do_log: True if gurobi log is activated, false otherwise
        :param dump_model: True if gurobi model will be written to an output textfile, false otherwise
        """
        self.do_log = do_log
        self.log_filepath = log_filepath
        self.dump_model = dump_model
        self.dump_model_filepath = dump_model_filepath


class GrbOptModel(object):
    """Base class for gurobi optimization models"""

    def __init__(self, model=grb.Model(''),
                 params=GrbOptParameters()):
        self.model = model
        self.params = params  # type: GrbOptParameters
        self.model.params.OutputFlag = self.params.do_log
        self.model.params.LogToConsole = self.params.do_log
        self.model.params.LogFile = self.params.log_filepath
        self.model_is_built = False

    def build_model(self):
        self.model_is_built = True
        if self.params.dump_model:
            self.model.write(self.params.dump_model_filepath)

    def solve(self):
        """Solves this optimization model

        :return: The objective function value
        """
        if not self.model_is_built:
            self.build_model()
        if self.params.dump_model:
            self.model.write(self.params.dump_model_filepath)
        self.model.optimize()
        if not self.is_solved_to_optimality():
            logging.info("Optimization model not solved to optimality.")
        return self.get_grb_objective_value()

    def is_solved_to_optimality(self):
        return self.model.Status == grb.GRB.OPTIMAL

    def get_grb_objective_value(self):
        return self.model.ObjVal

    def get_grb_vars_solution(self, grb_vars):
        return self.model.getAttr("X", grb_vars)

    def get_grb_constraints_shadow_prices(self, grb_constraints):
        return self.model.getAttr("Pi", grb_constraints)

    def create_grb_abs_constraint_pair(self, lhs_in_abs, rhs, name=None):
        """Transforms the constraint abs(lhs_in_abs)<=rhs in two constraints: -rhs<=lhs_in_abs<=+rhs"""
        if name is None:
            constr1 = self.model.addConstr(-rhs, grb.GRB.LESS_EQUAL, lhs_in_abs)
            constr2 = self.model.addConstr(lhs_in_abs, grb.GRB.LESS_EQUAL, rhs)
        else:
            constr1 = self.model.addConstr(-rhs, grb.GRB.LESS_EQUAL, lhs_in_abs, name=name + "_minusAbs")
            constr2 = self.model.addConstr(lhs_in_abs, grb.GRB.LESS_EQUAL, rhs, name=name + "_plusAbs")
        return constr1, constr2
