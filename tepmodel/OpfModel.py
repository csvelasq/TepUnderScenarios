import gurobipy as grb
import pandas as pd
import GrbOptModel as mygrb
import Utils


class OpfModel(mygrb.GrbOptModel):
    def __init__(self, state, opf_model_params, model=grb.Model('')):
        # type: (powersys.PowerSystemState.PowerSystemState, OpfModelParameters, gurobipy.Model) -> None
        mygrb.GrbOptModel.__init__(self, model)
        self.state = state
        self.model = model
        self.opf_model_params = opf_model_params
        if self.opf_model_params.slack_bus is None:
            self.slack_bus = self.state.system.nodes[0]
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
            pgen_obj = node_state.generation_marginal_cost * self.state.duration
            self.pgen[node_state] = self.model.addVar(lb=0,
                                                      ub=node_state.available_generating_capacity,
                                                      obj=pgen_obj * self.opf_model_params.obj_func_multiplier,
                                                      name='Pg{0}_{1}'.format(node_state.node.name, self.state))
            lshed_obj = self.opf_model_params.load_shedding_cost * self.state.duration
            self.load_shed[node_state] = self.model.addVar(lb=0,
                                                           ub=node_state.load_state,
                                                           obj=lshed_obj * self.opf_model_params.obj_func_multiplier,
                                                           name='Ls{0}_{1}'.format(node_state.node.name, self.state))
            if node_state.node == self.opf_model_params.slack_bus:
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

    def solve(self):
        # TODO review and improve this reporting
        obj_val = super(OpfModel, self).solve()
        if self.opf_model_params.save_detailed_solution:
            self.model.write("opf.lp")
            return OpfModelResults(self, print_solution=True)
        else:
            return obj_val


class OpfModelResults(object):
    def __init__(self, opf_model, print_solution=False):
        # type: (OpfModel) -> None
        self.model_state_name = opf_model.state.name
        # opf_model raw solutions
        pgen_sol = opf_model.get_grb_vars_solution(opf_model.pgen)
        load_shed_sol = opf_model.get_grb_vars_solution(opf_model.load_shed)
        bus_angle_sol = opf_model.get_grb_vars_solution(opf_model.bus_angle)
        power_flow_sol = opf_model.get_grb_vars_solution(opf_model.power_flow)
        # summary of solution
        self.total_power_output = sum(pgen_sol.values())
        self.total_energy_output = self.total_power_output * opf_model.state.duration
        self.total_load_shed = sum(load_shed_sol.values())
        self.total_energy_shed = self.total_load_shed * opf_model.state.duration
        self.hourly_gen_costs = sum(pgen_sol[node_state] * node_state.generation_marginal_cost
                                    for node_state in opf_model.state.node_states)
        self.total_gen_costs = self.hourly_gen_costs * opf_model.state.duration
        self.hourly_ls_costs = self.total_load_shed * opf_model.opf_model_params.load_shedding_cost
        self.total_ls_costs = self.hourly_gen_costs * opf_model.state.duration
        self.hourly_op_costs = self.hourly_gen_costs + self.hourly_ls_costs
        self.total_op_costs = self.total_gen_costs + self.total_ls_costs
        # nodal hourly solution (MW and US$/h)
        self.df_nodal_soln = pd.DataFrame(columns=['Node',
                                                   'Hourly Operation Cost [US$/h]',
                                                   'Hourly Generation Cost [US$/h]',
                                                   'Hourly Load Shedding Cost [US$/h]',
                                                   'Power Generated [MW]',
                                                   'Load Shed [MW]',
                                                   'Consumption [MW]',
                                                   'Bus Angle [rad]'])
        n = 0
        obj_mult = opf_model.opf_model_params.obj_func_multiplier
        for node_state in opf_model.state.node_states:
            # costs are converted back from MMUS$ to US$
            pcost = pgen_sol[node_state] * node_state.generation_marginal_cost / obj_mult
            lscost = load_shed_sol[node_state] * opf_model.opf_model_params.load_shedding_cost / obj_mult
            row = [node_state.node.name,
                   pcost + lscost,
                   pcost,
                   lscost,
                   pgen_sol[node_state],
                   load_shed_sol[node_state],
                   node_state.load_state - load_shed_sol[node_state],
                   bus_angle_sol[node_state]]
            self.df_nodal_soln.loc[n] = row
            n += 1
        # transmission lines power flows
        self.df_lines_soln = pd.DataFrame(columns=['Transmission Line',
                                                   'Power Flow [MW]',
                                                   'Angle from [rad]',
                                                   'Angle to [rad]',
                                                   'Susceptance [pu]'])
        n = 0
        for line_state in opf_model.state.transmission_lines_states:
            row = [line_state.transmission_line.name,
                   power_flow_sol[line_state],
                   bus_angle_sol[line_state.node_from_state],
                   bus_angle_sol[line_state.node_to_state],
                   line_state.transmission_line.susceptance]
            self.df_lines_soln.loc[n] = row
            n += 1
        # print solution if required
        if print_solution:
            self.print_to_console()

    def print_to_console(self):
        print '*' * 50
        print '** Opf Model state {0} solution:'.format(self.model_state_name)
        Utils.print_scalar_attributes_to_console(self)
        print '\n\n'
        print '*' * 25
        print '** Detailed nodal solution:'
        print self.df_nodal_soln
        print '\n\n'
        print '** Detailed lines solution:'
        print self.df_lines_soln
        print '*' * 50


class OpfModelParameters(object):
    def __init__(self, load_shedding_cost=2000,
                 slack_bus=None,
                 bus_angle_min=-grb.GRB.INFINITY, bus_angle_max=grb.GRB.INFINITY,
                 bus_angle_max_difference=None,
                 obj_func_multiplier=1e-6,  # US$ -> MMUS$
                 save_detailed_solution=False):
        self.load_shedding_cost = load_shedding_cost
        self.slack_bus = slack_bus
        self.bus_angle_min = bus_angle_min
        self.bus_angle_max = bus_angle_max
        self.bus_angle_max_difference = bus_angle_max_difference
        self.obj_func_multiplier = obj_func_multiplier
        self.save_detailed_solution = save_detailed_solution


class ScenarioOpfModel(mygrb.GrbOptModel):
    def __init__(self, scenario, opf_model_params,
                 model=grb.Model('')):
        # type: (powersys.PowerSystemScenario.PowerSystemScenario, OpfModelParameters, gurobipy.Model) -> None
        mygrb.GrbOptModel.__init__(self, model)
        self.scenario = scenario
        self.opf_model_params = opf_model_params
        # Build OPF model for each state
        self.opf_models = dict()
        self.opf_models_results = dict()
        for state in self.scenario.states:
            opf_model = OpfModel(state, self.opf_model_params, self.model)
            self.opf_models[state] = opf_model
            # TODO build and encapsulate opf static states model results in another class


class ScenariosOpfModel(object):
    def __init__(self, scenarios, opf_model_params):
        # type: (list[powersys.PowerSystemScenario.PowerSystemScenario], OpfModelParameters) -> None
        self.scenarios = scenarios
        self.opf_model_params = opf_model_params
        # Build independent simulation models for each scenario
        self.scenarios_models = dict()
        for scenario in self.scenarios:
            scenario_model = ScenarioOpfModel(scenario, self.opf_model_params, model=grb.Model(''))
            scenario_model.solve()
            self.scenarios_models[scenario] = scenario_model
            # TODO build and encapsulate opf static states model results in another class

    def solve(self):
        operation_costs_scenarios = dict()
        for scenario in self.scenarios:
            operation_costs_scenario = self.scenarios_models[scenario].solve()
            operation_costs_scenarios[scenario] = operation_costs_scenario
        return operation_costs_scenarios
