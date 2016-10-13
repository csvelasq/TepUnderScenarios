import gurobipy as grb
import pandas as pd
import GrbOptModel as mygrb


class OpfModel(mygrb.GrbOptModel):
    def __init__(self, state, opf_model_params, model=grb.Model('')):
        # type: (powersys.PowerSystemState.PowerSystemState, OpfModelParameters, gurobipy.Model) -> None
        mygrb.GrbOptModel.__init__(self, model)
        self.state = state
        self.model = model
        self.opf_model_params = opf_model_params
        # Model variables
        self.pgen = {}
        self.load_shed = {}
        self.bus_angle = {}
        self.power_flow = {}
        # Model constraints
        self.nodal_power_balance = {}
        self.dc_power_flow = {}
        # Create nodal variables: pgen, load_shed and bus_angle
        for node_state in self.state.node_states:
            # variables are created even if a node has no load or generator
            self.pgen[node_state] = self.model.addVar(lb=0,
                                                      ub=node_state.available_generating_capacity,
                                                      obj=node_state.generation_marginal_cost * self.state.duration)
            self.load_shed[node_state] = self.model.addVar(lb=0,
                                                           ub=node_state.load_state,
                                                           obj=self.opf_model_params.load_shedding_cost * self.state.duration)
            if node_state == self.opf_model_params.slack_bus:
                self.bus_angle[node_state] = self.model.addVar(lb=0,
                                                               ub=0,
                                                               obj=0)
            else:
                self.bus_angle[node_state] = self.model.addVar(lb=self.opf_model_params.bus_angle_min,
                                                               ub=self.opf_model_params.bus_angle_max,
                                                               obj=0)
        # Create line variables: power_flow
        for line_state in self.state.transmission_lines_states:
            # unavailable transmission lines are deactivated but included in the model
            cap = line_state.isavailable * line_state.transmission_line.thermal_capacity
            self.power_flow[line_state] = self.model.addVar(lb=-cap,
                                                            ub=+cap,
                                                            obj=0)
        # Update model to integrate new variables
        self.model.modelSense = grb.GRB.MINIMIZE
        self.model.update()
        # Create nodal_power_balance constraints
        for node_state in self.state.node_states:
            rhs_expr = grb.LinExpr(self.pgen[node_state])
            for line_state in node_state.find_incoming_lines_states():
                rhs_expr += self.power_flow[line_state]
            for line_state in node_state.find_outgoing_lines_states():
                rhs_expr -= self.power_flow[line_state]
            self.nodal_power_balance[node_state] = \
                self.model.addConstr(rhs_expr,
                                     grb.GRB.EQUAL,
                                     node_state.load_state - self.load_shed[node_state]
                                     )
        # Create dc_power_flow constraints
        for line_state in self.state.transmission_lines_states:
            bus_angle_from = self.bus_angle[line_state.node_from_state]
            bus_angle_to = self.bus_angle[line_state.node_to_state]
            # unavailable transmission lines are deactivated but included in the model
            susceptance = line_state.isavailable * line_state.transmission_line.susceptance
            self.dc_power_flow[line_state] = \
                self.model.addConstr(self.power_flow[line_state],
                                     grb.GRB.EQUAL,
                                     susceptance * bus_angle_from - susceptance * bus_angle_to)
            # TODO Create bus angle difference constraint, based on self.bus_angle_max_difference

    def get_nodal_solution(self):
        # TODO encapsulate opf model results in another class
        pgen_sol = self.get_grb_vars_solution(self.pgen)
        load_shed_sol = self.get_grb_vars_solution(self.load_shed)
        bus_angle_sol = self.get_grb_vars_solution(self.bus_angle)
        df_nodal_solution = pd.DataFrame(columns=['Node', 'Power Generated [MW]', 'Load Shed [MW]', 'Bus Angle [rad]'])
        n = 0
        for node_state in self.state.node_states:
            row = [node_state.node.name,
                   pgen_sol[node_state],
                   load_shed_sol[node_state],
                   bus_angle_sol[node_state]]
            df_nodal_solution.loc[n] = row
            n += 1
        return df_nodal_solution


class OpfModelParameters(object):
    def __init__(self, state, load_shedding_cost=2000,
                 slack_bus=None,
                 bus_angle_min=-grb.GRB.INFINITY, bus_angle_max=grb.GRB.INFINITY,
                 bus_angle_max_difference=None):
        self.state = state
        self.load_shedding_cost = load_shedding_cost
        self.slack_bus = slack_bus
        if slack_bus is None:
            self.slack_bus = state.find_node_state(self.state.system.nodes[0])
        self.bus_angle_min = bus_angle_min
        self.bus_angle_max = bus_angle_max
        self.bus_angle_max_difference = bus_angle_max_difference


class StaticStatesOpfModel(mygrb.GrbOptModel):
    def __init__(self, states, opf_model_params,
                 model=grb.Model('')):
        # type: (powersys.PowerSystemState.PowerSystemState, OpfModelParameters, gurobipy.Model) -> None
        mygrb.GrbOptModel.__init__(self, model)
        self.states = states
        self.opf_model_params = opf_model_params
        # Build OPF model for each state
        self.opf_models = []
        for state in states:
            opf_model = OpfModel(state, self.opf_model_params, model)
            self.opf_models.append(opf_model)
            # TODO build and encapsulate opf static states model results in another class
