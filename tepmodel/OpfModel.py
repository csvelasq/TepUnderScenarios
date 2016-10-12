import gurobipy as grb
import Utils


class OpfModel:
    def __init__(self, state, load_shedding_cost, model=grb.Model('')):
        self.state = state
        self.model = model
        self.load_shedding_cost = load_shedding_cost
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
                                                           obj=self.load_shedding_cost * self.state.duration)
            self.bus_angle[node_state] = self.model.addVar(obj=0)
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

    def solve(self):
        self.model.optimize()  # the opf is always a feasible LP due to load shedding
        return self.model.objVal

    def get_nodal_solution(self):
        dict1 = self.get_load_shed_solution()
        dict2 = self.get_power_generated_solution()
        dict3 = self.get_bus_angle_solution()
        dicts = dict1, dict2, dict3
        nodal_solution = {k: [d.get(k) for d in dicts] for k in {k for d in dicts for k in d}}
        # http://stackoverflow.com/questions/2365921/merging-python-dictionaries
        return nodal_solution

    def get_power_generated_solution(self):
        return self.model.getAttr("X", self.pgen)

    def get_load_shed_solution(self):
        return self.model.getAttr("X", self.load_shed)

    def get_bus_angle_solution(self):
        return self.model.getAttr("X", self.bus_angle)

    def get_power_flows_solution(self):
        return self.model.getAttr("X", self.power_flow)
