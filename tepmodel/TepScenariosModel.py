import gurobipy as grb
import GrbOptModel as mygrb
import OpfModel
import Utils


class StaticTePlan(object):
    def __init__(self, tep_system):
        # type: (powersys.PowerSystemPlanning.PowerSystemTransmissionPlanning) -> None
        self.tep_system = tep_system
        self.candidate_lines_built = []

    def set_plan_by_id(self, plan_id):
        Utils.subset_from_id(self.tep_system.candidate_lines, plan_id)

    def get_plan_id(self):
        return Utils.subset_to_id(self.tep_system.candidate_lines, self.candidate_lines_built)


class TepScenariosModel(mygrb.GrbOptModel):
    def __init__(self, tep_system,
                 tep_scenarios_model_parameters,
                 candidate_lines,
                 model=grb.Model('')):
        # type: (powersys.PowerSystemState.PowerSystemState, TepScenariosModelParameters, list, gurobipy.Model) -> None
        mygrb.GrbOptModel.__init__(self, model)
        self.tep_system = tep_system
        self.tep_scenarios_model_parameters = tep_scenarios_model_parameters
        self.candidate_lines = candidate_lines


class TepScenariosModelParameters(object):
    def __init__(self, opf_model_params,
                 investment_costs_multiplier=1, operation_costs_multiplier=1):
        # type: (double, double, OpfModel.OpfModelParameters) -> None
        self.investment_costs_multiplier = investment_costs_multiplier
        self.operation_costs_multiplier = operation_costs_multiplier
        self.opf_model_params = opf_model_params
