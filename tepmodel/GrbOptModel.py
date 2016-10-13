import gurobipy as grb


class GrbOptModel(object):
    """Base class for gurobi optimization models"""

    def __init__(self, model=grb.Model('')):
        self.model = model
        self.model.params.OutputFlag = 0

    def solve(self):
        self.model.optimize()
        return self.model.objVal

    def get_grb_vars_solution(self, grb_vars):
        return self.model.getAttr("X", grb_vars)
