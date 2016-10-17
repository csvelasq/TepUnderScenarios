import gurobipy as grb
import pandas as pd
import collections
import xlsxwriter
import GrbOptModel as mygrb
import Utils


class OpfModelParameters(object):
    def __init__(self, load_shedding_cost=2000,
                 base_mva=100,
                 slack_bus=None,
                 bus_angle_min=-grb.GRB.INFINITY, bus_angle_max=grb.GRB.INFINITY,
                 bus_angle_max_difference=None,
                 obj_func_multiplier=1e-6,  # US$ -> MMUS$
                 grb_opt_params=mygrb.GrbOptParameters()):
        self.load_shedding_cost = load_shedding_cost
        self.base_mva = base_mva
        self.slack_bus = slack_bus
        self.bus_angle_min = bus_angle_min
        self.bus_angle_max = bus_angle_max
        self.bus_angle_max_difference = bus_angle_max_difference
        self.obj_func_multiplier = obj_func_multiplier
        self.grb_opt_params = grb_opt_params


class OpfModel(mygrb.GrbOptModel):
    """A Linear Optimal Power Flow model (DC power flow), always feasible LP,
    for one particular state of the power system (a state which lasts for a given amount of hours)"""

    def __init__(self, state,
                 opf_model_params=OpfModelParameters(),
                 model=grb.Model('')):
        # type: (powersys.PowerSystemState.PowerSystemState, OpfModelParameters, gurobipy.Model) -> None
        mygrb.GrbOptModel.__init__(self, model, opf_model_params.grb_opt_params)
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
        # build gurobi model
        self.create_grb_vars()
        self.model.modelSense = grb.GRB.MINIMIZE
        self.model.update()  # Update model to integrate new variables
        self.create_grb_constraints()

    def create_grb_vars(self):
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
            if node_state.node == self.slack_bus:
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

    def create_grb_constraints(self):
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
            susceptance = line_state.isavailable * line_state.transmission_line.susceptance * self.opf_model_params.base_mva
            self.dc_power_flow[line_state] = \
                self.model.addConstr(self.power_flow[line_state],
                                     grb.GRB.EQUAL,
                                     susceptance * bus_angle_from - susceptance * bus_angle_to)
            # TODO Create bus angle difference constraint, based on self.bus_angle_max_difference

    def get_lmps(self):
        """Calculates nodal spot prices for the solution in this opf"""
        assert self.is_solved_to_optimality()
        shadow_prices = self.get_grb_constraints_shadow_prices(self.nodal_power_balance)
        for key, val in shadow_prices.iteritems():
            shadow_prices[key] = val / (self.opf_model_params.obj_func_multiplier * self.state.duration)
        return shadow_prices


class OpfModelResults(object):
    """Detailed results for a solved OPF model"""

    def __init__(self, opf_model, print_solution=False):
        # type: (OpfModel) -> None
        assert opf_model.is_solved_to_optimality()
        self.model_state_name = opf_model.state.name
        self.state = opf_model.state
        # opf_model raw solutions
        self.obj_val = opf_model.get_grb_objective_value()
        pgen_sol = opf_model.get_grb_vars_solution(opf_model.pgen)
        load_shed_sol = opf_model.get_grb_vars_solution(opf_model.load_shed)
        bus_angle_sol = opf_model.get_grb_vars_solution(opf_model.bus_angle)
        power_flow_sol = opf_model.get_grb_vars_solution(opf_model.power_flow)
        spot_prices_sol = opf_model.get_lmps()
        spot_prices_sol_list = Utils.get_values_from_dict(spot_prices_sol)
        # summary of solution [MW, GWh, US$/h, MUS$, US$/MWh]
        self.summary_sol = collections.OrderedDict()
        self.summary_sol['Objective Function Value'] = opf_model.get_grb_objective_value()
        self.summary_sol['State Duration [hours]'] = self.state.duration
        hourly_gen_costs = sum(pgen_sol[node_state] * node_state.generation_marginal_cost
                               for node_state in opf_model.state.node_states)
        hourly_ls_costs = sum(load_shed_sol[node_state] * opf_model.opf_model_params.load_shedding_cost
                              for node_state in opf_model.state.node_states)
        self.summary_sol['Total Operation Costs [MMUS$]'] = (
                                                                hourly_gen_costs + hourly_ls_costs) * self.state.duration / 1e6
        self.summary_sol['Total Generation Costs [MMUS$]'] = hourly_gen_costs * self.state.duration / 1e6
        self.summary_sol['Total Load Shedding Costs [MMUS$]'] = hourly_ls_costs * self.state.duration / 1e6
        self.summary_sol['Relative Load Shedding Costs [%]'] = self.summary_sol['Total Load Shedding Costs [MMUS$]'] / \
                                                               self.summary_sol['Total Operation Costs [MMUS$]']
        self.summary_sol['Congestion Rents [MMUS$]'] = 0
        self.summary_sol['Congestion Rents / Operation Costs [%]'] = 0
        self.summary_sol['Hourly Operation Costs [US$/h]'] = hourly_gen_costs + hourly_ls_costs
        self.summary_sol['Hourly Generation Costs [US$/h]'] = hourly_gen_costs
        self.summary_sol['Hourly Load Shedding Costs [US$/h]'] = hourly_ls_costs
        self.summary_sol['Congestion Rents [US$/h]'] = 0  # value is assigned later
        self.summary_sol['Minimum Spot Price [US$/MWh]'] = min(spot_prices_sol_list)
        self.summary_sol['Average Spot Price [US$/MWh]'] = float(sum(spot_prices_sol_list)) / len(spot_prices_sol_list)
        self.summary_sol['Maximum Spot Price [US$/MWh]'] = max(spot_prices_sol_list)
        self.summary_sol['Total Power Output [MW]'] = sum(pgen_sol.values())
        self.summary_sol['Total Energy Output [GWh]'] = self.summary_sol[
                                                            'Total Power Output [MW]'] * opf_model.state.duration / 1e3
        # TODO add total generation utilization percent, here and in derived results as well
        demand = self.state.get_total_demand_power()
        self.summary_sol['Total Demand [MW]'] = demand
        self.summary_sol['Total Demand [GWh]'] = self.state.get_total_demand_energy()
        ls = sum(load_shed_sol.values())
        self.summary_sol['Total Load Shed [MW]'] = ls
        self.summary_sol['Total Energy Shed [GWh]'] = self.summary_sol[
                                                          'Total Load Shed [MW]'] * opf_model.state.duration / 1e3
        self.summary_sol['Relative Load Shedding [%]'] = Utils.get_utilization(ls, demand)
        self.is_load_lost = ls > 0
        # nodal hourly solution (MW and US$/h)
        self.df_nodal_soln = pd.DataFrame(columns=['Node',
                                                   'Hourly Operation Cost [US$/h]',
                                                   'Hourly Generation Cost [US$/h]',
                                                   'Hourly Load Shedding Cost [US$/h]',
                                                   'Relative Load Shedding Cost [%]',
                                                   'Spot Price [US$/MWh]',
                                                   'Marginal Generation Cost [US$/MWh]',
                                                   'Power Generated [MW]',
                                                   'Available Generating Capacity [MW]',
                                                   'Generation Capacity Factor [%]',
                                                   'Load [MW]',
                                                   'Load Shed [MW]',
                                                   'Consumption [MW]',
                                                   'Relative Load Shed [%]',
                                                   'Bus Angle [rad]'])
        n = 0
        for node_state in opf_model.state.node_states:
            # costs are converted back from MMUS$ to US$
            pcost = pgen_sol[node_state] * node_state.generation_marginal_cost
            lscost = load_shed_sol[node_state] * opf_model.opf_model_params.load_shedding_cost
            op_cost = pcost + lscost
            row = [node_state.node.name,
                   op_cost,
                   pcost,
                   lscost,
                   Utils.get_utilization(lscost, op_cost),
                   spot_prices_sol[node_state],
                   node_state.generation_marginal_cost,
                   pgen_sol[node_state],
                   node_state.available_generating_capacity,
                   Utils.get_utilization(pgen_sol[node_state], node_state.available_generating_capacity),
                   node_state.load_state,
                   load_shed_sol[node_state],
                   node_state.load_state - load_shed_sol[node_state],
                   Utils.get_utilization(load_shed_sol[node_state], node_state.load_state),
                   bus_angle_sol[node_state]]
            self.df_nodal_soln.loc[n] = row
            n += 1
        # transmission lines power flows in solution [MW]
        self.df_lines_soln = pd.DataFrame(columns=['Transmission Line',
                                                   'Is active?',
                                                   'Power Flow [MW]',
                                                   'Thermal Capacity [MW]',
                                                   'Utilization [%]',
                                                   'Spot price from [US$/MWh]',
                                                   'Spot price to [US$/MWh]',
                                                   'Spot price from-to [US$/MWh]',
                                                   'Congestion Rents [US$/h]',
                                                   'Congestion Rents [MMUS$]',
                                                   'Angle from [rad]',
                                                   'Angle to [rad]',
                                                   'Angle from-to [rad]',
                                                   'Susceptance [pu]'])
        n = 0
        total_congestion_rents = 0
        for line_state in opf_model.state.transmission_lines_states:
            spot_diff = spot_prices_sol[line_state.node_from_state] - spot_prices_sol[line_state.node_to_state]
            congestion_rent = abs(spot_diff * power_flow_sol[line_state])
            total_congestion_rents += congestion_rent
            row = [str(line_state.transmission_line),
                   line_state.isavailable,
                   power_flow_sol[line_state],
                   line_state.transmission_line.thermal_capacity,
                   Utils.get_utilization(abs(power_flow_sol[line_state]),
                                         line_state.transmission_line.thermal_capacity),
                   spot_prices_sol[line_state.node_from_state],
                   spot_prices_sol[line_state.node_to_state],
                   spot_diff,
                   congestion_rent,
                   congestion_rent * self.state.duration / 1e6,
                   bus_angle_sol[line_state.node_from_state],
                   bus_angle_sol[line_state.node_to_state],
                   bus_angle_sol[line_state.node_from_state] - bus_angle_sol[line_state.node_to_state],
                   line_state.transmission_line.susceptance]
            self.df_lines_soln.loc[n] = row
            n += 1
        self.summary_sol['Congestion Rents [US$/h]'] = total_congestion_rents
        self.summary_sol['Congestion Rents [MMUS$]'] = total_congestion_rents * self.state.duration / 1e6
        self.summary_sol['Congestion Rents / Operation Costs [%]'] = self.summary_sol[
                                                                         'Congestion Rents [MMUS$]'] / \
                                                                     self.summary_sol[
                                                                         'Total Operation Costs [MMUS$]']
        self.df_summary = pd.DataFrame(self.summary_sol, index=[self.state.name])
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

    def to_excel(self, filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        self.to_excel_sheets(writer)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_excel_sheets(self, writer, sheetname_prefix="", recursive=False):
        if not recursive:
            sheetname_summary = sheetname_prefix + "{0}_summary".format(self.state.name)
            Utils.df_to_excel_sheet_autoformat(self.df_summary, writer, sheetname_summary)
        sheetname_nodal = sheetname_prefix + "{0}_nodes".format(self.state.name)
        Utils.df_to_excel_sheet_autoformat(self.df_nodal_soln, writer, sheetname_nodal)
        sheetname_lines = sheetname_prefix + "{0}_lines".format(self.state.name)
        Utils.df_to_excel_sheet_autoformat(self.df_lines_soln, writer, sheetname_lines)


class ScenarioOpfModel(mygrb.GrbOptModel):
    """An OPF model for a particular scenario of a power system.
    A scenario is understood as a static collection of power system states
    (for example, hourly states for a full year under some set of assumptions)"""

    def __init__(self, scenario, opf_model_params=OpfModelParameters(),
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


class ScenarioOpfModelResults(object):
    """Detailed results for a solved Scenario OPF model"""

    def __init__(self, scenario_model):
        # type: (ScenarioOpfModel) -> None
        assert scenario_model.is_solved_to_optimality()
        self.scenario = scenario_model.scenario
        self.model_scenario_name = scenario_model.scenario.name
        # build detailed solutions for each opf_model
        self.opf_models_results = dict()
        for state in self.scenario.states:
            self.opf_models_results[state] = OpfModelResults(scenario_model.opf_models[state])
        # build summary of solution for this scenario [MW, GWh, US$/h, MUS$, US$/MWh]
        opf_results = self.opf_models_results.values()
        self.obj_val = scenario_model.get_grb_objective_value()
        self.summary_sol = collections.OrderedDict()
        self.summary_sol['Objective Function Value'] = self.obj_val
        self.summary_sol['Total Operation Costs [MMUS$]'] = sum(r.summary_sol['Total Operation Costs [MMUS$]']
                                                                for r in opf_results)
        self.summary_sol['Total Generation Costs [MMUS$]'] = sum(r.summary_sol['Total Generation Costs [MMUS$]']
                                                                 for r in opf_results)
        self.summary_sol['Total Load Shedding Costs [MMUS$]'] = sum(r.summary_sol['Total Load Shedding Costs [MMUS$]']
                                                                    for r in opf_results)
        self.summary_sol['Relative Load Shedding Costs [%]'] = self.summary_sol['Total Load Shedding Costs [MMUS$]'] / \
                                                               self.summary_sol['Total Operation Costs [MMUS$]']
        self.summary_sol['Congestion Rents [MMUS$]'] = sum(r.summary_sol['Congestion Rents [MMUS$]']
                                                           for r in opf_results)
        self.summary_sol['Congestion Rents / Operation Costs [%]'] = self.summary_sol['Congestion Rents [MMUS$]'] / \
                                                                     self.summary_sol['Total Operation Costs [MMUS$]']
        self.summary_sol['Minimum Spot Price [US$/MWh]'] = min(r.summary_sol['Minimum Spot Price [US$/MWh]']
                                                               for r in opf_results)
        self.summary_sol['Average Spot Price [US$/MWh]'] = sum(
            r.summary_sol['Average Spot Price [US$/MWh]'] * r.state.duration
            for r in opf_results) / sum(r.state.duration for r in opf_results)
        self.summary_sol['Peak Spot Price [US$/MWh]'] = max(r.summary_sol['Maximum Spot Price [US$/MWh]']
                                                            for r in opf_results)
        self.summary_sol['Total Energy Output [GWh]'] = sum(r.summary_sol['Total Energy Output [GWh]']
                                                            for r in opf_results)
        total_demand = sum(r.summary_sol['Total Demand [GWh]']
                           for r in opf_results)
        self.summary_sol['Total Energy Demand [GWh]'] = total_demand
        self.summary_sol['Total Energy Shed [GWh]'] = sum(r.summary_sol['Total Energy Shed [GWh]']
                                                          for r in opf_results)
        self.summary_sol['Relative Energy Shed [%]'] = self.summary_sol['Total Energy Shed [GWh]'] / total_demand
        self.summary_sol['Loss of Load [hours]'] = sum(r.is_load_lost * r.state.duration
                                                       for r in opf_results)
        self.df_summary = pd.DataFrame(self.summary_sol, index=[self.model_scenario_name])
        # join summary of solution for each state in this scenario
        list_states_summaries = []
        for state in self.scenario.states:
            sname = "{0}_{1}".format(self.scenario.name, state.name)
            list_states_summaries.append(self.opf_models_results[state].df_summary)
        self.df_states_summaries = pd.concat(list_states_summaries)
        # TODO detailed nodal and line solutions for this scenario (collection of states)

    def to_excel(self, filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        self.to_excel_sheets(writer)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_excel_sheets(self, writer, recursive=False):
        # summary of this scenario. if recursive, this summary is written along other scenarios
        if not recursive:
            sheetname_summary = "{0}".format(self.scenario.name)
            Utils.df_to_excel_sheet_autoformat(self.df_summary, writer, sheetname_summary)
        # summary of operation under all states of this scenario
        sheetname_states = "AllStates_{0}".format(self.scenario.name)
        Utils.df_to_excel_sheet_autoformat(self.df_states_summaries, writer, sheetname_states)
        if recursive:
            for state in self.scenario.states:
                self.opf_models_results[state].to_excel_sheets(writer, sheetname_prefix=self.scenario.name + '_',
                                                               recursive=recursive)


class ScenariosOpfModel(object):
    """Simulation of optimal power system operation under multiple independent scenarios.
    Optimal operation is simulated under each scenario independent of the operation in other scenarios."""

    def __init__(self, scenarios,
                 opf_model_params=OpfModelParameters()):
        # type: (list[powersys.PowerSystemScenario.PowerSystemScenario], OpfModelParameters) -> None
        self.scenarios = scenarios
        self.opf_model_params = opf_model_params
        # Build independent simulation models for each scenario
        self.model_each_scenario = dict()
        for scenario in self.scenarios:
            self.model_each_scenario[scenario] = ScenarioOpfModel(scenario, self.opf_model_params, model=grb.Model(''))
        self.operation_costs_scenarios = dict()

    def solve(self):
        """Simulates the optimal operation for each scenario

        :return: A dictionary with the minimum operation cost under each scenario
        """
        for scenario in self.scenarios:
            self.operation_costs_scenarios[scenario] = self.model_each_scenario[scenario].solve()
        return self.operation_costs_scenarios


class ScenariosOpfModelResults(object):
    """Results for the simulation of optimal power system operation under multiple independent scenarios"""

    def __init__(self, scenarios_model):
        # type: (ScenariosOpfModel) -> None
        assert len(scenarios_model.operation_costs_scenarios) == len(scenarios_model.scenarios)
        self.scenarios = scenarios_model.scenarios
        # build detailed solutions for each scenario model and build a summary table of scenarios
        self.scenarios_models_results = collections.OrderedDict()
        list_dfs_all_states_summaries = []
        list_dfs_all_scenarios_summaries = []
        for scenario in self.scenarios:
            self.scenarios_models_results[scenario] = ScenarioOpfModelResults(
                scenarios_model.model_each_scenario[scenario])
            list_dfs_all_scenarios_summaries.append(self.scenarios_models_results[scenario].df_summary)
            for state in scenario.states:
                list_dfs_all_states_summaries.append(
                    pd.DataFrame(self.scenarios_models_results[scenario].opf_models_results[state].summary_sol,
                                 index=[scenario.name + '_' + state.name]))
        self.df_all_states_summaries = pd.concat(list_dfs_all_states_summaries)
        self.scenarios_models_summaries = pd.concat(list_dfs_all_scenarios_summaries)

    def to_excel(self, filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        self.to_excel_sheets(writer)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_excel_sheets(self, writer, recursive=False):
        # summary of operation under all scenarios, in a single sheet
        sheetname_summary = 'Scenarios'
        Utils.df_to_excel_sheet_autoformat(self.scenarios_models_summaries, writer, sheetname_summary)
        # summary of operation under all states, in a single sheet per scenario
        sheetname_summary = 'AllStates'
        Utils.df_to_excel_sheet_autoformat(self.df_all_states_summaries, writer, sheetname_summary)
        # TODO fix the behaviour under recursive=True and use more pd.concat instead of for loops
        if recursive:
            for scenario in self.scenarios:
                self.scenarios_models_results[scenario].to_excel_sheets(writer, recursive)
