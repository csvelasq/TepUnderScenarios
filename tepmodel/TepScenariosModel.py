import OpfModel as opf
import powersys.PowerSystemPlanning as pwsp
import Utils
import plotly
from plotly.graph_objs import Scatter, Layout
import pandas as pd
import xlsxwriter
import collections
import logging
import time
from datetime import timedelta


class TepScenariosModelParameters(object):
    def __init__(self, opf_model_params=opf.OpfModelParameters(),
                 investment_costs_multiplier=1, operation_costs_multiplier=1):
        # type: (double, double, opf.OpfModelParameters) -> None
        self.investment_costs_multiplier = investment_costs_multiplier
        self.operation_costs_multiplier = operation_costs_multiplier
        self.opf_model_params = opf_model_params


class TepScenariosModel(object):
    def __init__(self, tep_system,
                 tep_scenarios_model_parameters=TepScenariosModelParameters()):
        # type: (powersys.PowerSystemPlanning.PowerSystemTransmissionPlanning, TepScenariosModelParameters, gurobipy.Model) -> None
        self.tep_system = tep_system
        self.tep_scenarios_model_parameters = tep_scenarios_model_parameters

    @staticmethod
    def import_from_excel(excel_filepath):
        imported_tep_system = pwsp.PowerSystemTransmissionPlanning.import_from_excel(excel_filepath)
        params = Utils.excel_worksheet_to_dict(excel_filepath, 'TepParameters')
        base_mva = 100 if 'base_mva' not in params.keys() else params['base_mva']
        opf_model_params = opf.OpfModelParameters(load_shedding_cost=params['load_shedding_cost'],
                                                  base_mva=base_mva)
        tep_scenarios_model_parameters = \
            TepScenariosModelParameters(opf_model_params,
                                        investment_costs_multiplier=params['investment_costs_multiplier'],
                                        operation_costs_multiplier=params['operation_costs_multiplier'])
        imported_tep_model = TepScenariosModel(imported_tep_system, tep_scenarios_model_parameters)
        return imported_tep_model

    def evaluate_plan(self, plan, tep_scenarios_model_parameters=None):
        # type: (StaticTePlan) -> [dict[powersys.PowerSystemScenario.PowerSystemScenario, float], dict[powersys.PowerSystemScenario.PowerSystemScenario, float], TepScenariosModelParameters]
        params = tep_scenarios_model_parameters
        if params is None:
            params = self.tep_scenarios_model_parameters
        total_costs = collections.OrderedDict()
        investment_cost = plan.get_total_investment_cost()
        # modify all scenarios so as to build lines in the provided plan
        # AFFECTS ALL REFERENCES TO THE PowerSystemTransmissionPlanning INSTANCE!
        self.tep_system.commit_build_lines(plan.candidate_lines_built)
        # simulate operation under each scenario
        scenario_simulation_model = opf.ScenariosOpfModel(self.tep_system.scenarios,
                                                          params.opf_model_params)
        operation_costs = scenario_simulation_model.solve()
        # total costs = operation + investment costs (with multipliers)
        investment_cost *= params.investment_costs_multiplier
        op_multiplier = params.operation_costs_multiplier
        for scenario in self.tep_system.scenarios:
            total_costs[scenario] = op_multiplier * operation_costs[scenario] + investment_cost
        return [scenario_simulation_model, total_costs, operation_costs]

    def get_numberof_possible_expansion_plans(self):
        return 1 << len(self.tep_system.candidate_lines)


class StaticTePlan(object):
    """A static transmission expansion plan (which lines to build, all simultaneously)"""

    def __init__(self, tep_model, candidate_lines_built=[]):
        # type: (TepScenariosModel, List[CandidateTransmissionLine]) -> None
        self.tep_model = tep_model
        self.candidate_lines_built = candidate_lines_built
        [self.scenario_simulation_model, self.total_costs, self.operation_costs] = self.tep_model.evaluate_plan(self)

    @staticmethod
    def from_id(tep_model, plan_id):
        # type: (TepScenariosModel, int) -> StaticTePlan
        plan = StaticTePlan(tep_model,
                            candidate_lines_built=Utils.subset_from_id(tep_model.tep_system.candidate_lines,
                                                                       plan_id))
        return plan

    def get_plan_id(self):
        return Utils.subset_to_id(self.tep_model.tep_system.candidate_lines, self.candidate_lines_built)

    def get_total_investment_cost(self):
        return sum(line.investment_cost for line in self.candidate_lines_built)

    # from https://github.com/DEAP/deap/blob/master/deap/base.py
    def dominates(self, other_plan):
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.total_costs.values(), other_plan.total_costs.values()):
            if self_wvalue < other_wvalue:
                not_equal = True
            elif self_wvalue > other_wvalue:
                return False
        return not_equal

    def is_nondominated(self, other_plans):
        for other_plan in other_plans:
            if other_plan != self and not self.dominates(other_plan):
                return False
        return True

    def get_plan_summary(self):
        # type: () -> OrderedDict
        # Basic info on this plan
        plan_summary = collections.OrderedDict()
        plan_summary['Plan ID'] = self.get_plan_id()
        # self.plan_summary['Kind'] = -- set elsewhere
        plan_summary['Number of Built Lines'] = len(self.candidate_lines_built)
        plan_summary['Built Lines'] = str(map(str, self.candidate_lines_built))
        plan_summary['Investment Cost [MMUS$]'] = self.get_total_investment_cost()
        plan_summary['Scaled Investment Cost [MMUS$]'] = plan_summary[
                                                             'Investment Cost [MMUS$]'] * self.tep_model.tep_scenarios_model_parameters.investment_costs_multiplier
        # Operation costs under each scenario
        for scenario in self.tep_model.tep_system.scenarios:
            plan_summary['Operation Costs {0} [MMUS$]'.format(scenario.name)] = self.operation_costs[scenario]
            plan_summary['Scaled Operation Costs {0} [MMUS$]'.format(scenario.name)] = self.operation_costs[
                                                                                           scenario] * self.tep_model.tep_scenarios_model_parameters.operation_costs_multiplier
            plan_summary['Total Costs {0} [MMUS$]'.format(scenario.name)] = self.total_costs[scenario]
        return plan_summary

    def regret(self, optimal_plan, scenario):
        """Calculates the regret of this expansion plan in a given scenario, provided the optimal plan for that scenario

        :param optimal_plan: Optimal plan under the provided scenario
        :param scenario: The scenario for which regret is calculated
        :return:The regret (lost opportunity cost)
        """
        # type: (StaticTePlan, powersys.PowerSystemScenario) -> float
        return self.total_costs[scenario] - optimal_plan.total_costs[scenario]

    def max_regret(self, optimal_plans):
        """Calculates the maximum regret for this plan among all scenarios

        :param optimal_plans: The set of optimal plans for each scenario
        :return: The maximum regret of this transmission expansion plan
        """
        # type: (Dict[powersys.PowerSystemScenario, StaticTePlan]) -> float
        return max(self.regret(optimal_plans[scenario], scenario)
                   for scenario in self.tep_model.tep_system.scenarios)

    def __str__(self):
        return "Plan {}".format(self.get_plan_id())


class StaticTePlanDetails(object):
    """Details for a given static transmission expansion plan, mainly performance assessment under scenarios"""

    def __init__(self, plan):
        self.plan = plan  # type: StaticTePlan
        # plan summary
        self.plan_summary = self.plan.get_plan_summary()
        # Details on the operation under each scenario
        self.scenarios_results = opf.ScenariosOpfModelResults(self.plan.scenario_simulation_model)
        for scenario in self.scenarios_results.scenarios:
            scenario_result = self.scenarios_results.scenarios_models_results[scenario]
            self.plan_summary['Relative Load Shedding Costs {0} [%]'.format(scenario.name)] = \
                scenario_result.summary_sol['Relative Load Shedding Costs [%]']
            self.plan_summary['Relative Energy Shed {} [%]'.format(scenario.name)] = \
                scenario_result.summary_sol['Relative Energy Shed [%]']
            self.plan_summary['Congestion Rents / Operation Costs {} [%]'.format(scenario.name)] = \
                scenario_result.summary_sol['Congestion Rents / Operation Costs [%]']
            self.plan_summary['Loss of Load {} [hours]'.format(scenario.name)] = \
                scenario_result.summary_sol['Loss of Load [hours]']
        self.plan_df_summary = pd.DataFrame(self.plan_summary, index=['Plan{0}'.format(self.plan_summary['Plan ID'])])

    def to_excel(self, filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        self.to_excel_sheets(writer)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_excel_sheets(self, writer):
        sheetname_summary = 'Plan {0}'.format(self.plan_summary['Plan ID'])
        Utils.df_to_excel_sheet_autoformat(self.plan_df_summary, writer, sheetname_summary)
        self.scenarios_results.to_excel_sheets(writer, recursive=True)


class ScenariosTepParetoFrontByBruteForceParams(object):
    def __init__(self, save_all_alternatives=False,
                 plans_processed_for_reporting=1000):
        self.save_all_alternatives = save_all_alternatives
        self.plans_processed_for_reporting = plans_processed_for_reporting


class ScenariosTepParetoFrontByBruteForce(object):
    def __init__(self, tep_model, pareto_brute_force_params=ScenariosTepParetoFrontByBruteForceParams()):
        # type: (TepScenariosModel, ScenariosTepParetoFrontByBruteForceParams) -> None
        self.tep_model = tep_model
        self.pareto_brute_force_params = pareto_brute_force_params
        if self.pareto_brute_force_params.save_all_alternatives:
            self.all_alternatives = []  # type: list[StaticTePlan]
        self.efficient_alternatives = []  # type: list[StaticTePlan]
        # enumerate and evaluate all possible alternatives
        self.number_of_possible_plans = self.tep_model.get_numberof_possible_expansion_plans()
        self.best_plan_scenario = collections.OrderedDict()
        for scenario in self.tep_model.tep_system.scenarios:
            self.best_plan_scenario[scenario] = None
        self.start_clock = time.clock()
        self.eval_all_alternatives()
        self.stop_time = time.clock()
        self.elapsed_seconds = self.stop_time - self.start_clock

    def eval_all_alternatives(self):
        for plan_id in range(self.number_of_possible_plans):
            # build and solve plan
            alternative = StaticTePlan.from_id(self.tep_model, plan_id)
            # update set of efficient plans with the newly created plan
            self.update_set_of_efficient_plans(alternative)
            # record plan anyway if all alternatives are being saved
            if self.pareto_brute_force_params.save_all_alternatives:
                self.all_alternatives.append(alternative)
            # report progress
            if (plan_id % self.pareto_brute_force_params.plans_processed_for_reporting) == 0:
                elapsed_time = time.clock() - self.start_clock
                logging.info(("Processed {0} / {1} alternatives "
                              "(elapsed: {2})...").format(plan_id, self.number_of_possible_plans,
                                                          str(timedelta(seconds=elapsed_time))
                                                          )
                             )

    def update_set_of_efficient_plans(self, alternative):
        dominated_to_remove = []
        alternative_is_efficient = True
        for idx, other_alt in enumerate(self.efficient_alternatives):
            if other_alt.dominates(alternative):
                alternative_is_efficient = False
                break
            elif alternative.dominates(other_alt):
                dominated_to_remove.append(other_alt)
        self.efficient_alternatives = [a for a in self.efficient_alternatives
                                       if a not in dominated_to_remove]
        if alternative_is_efficient:
            self.efficient_alternatives.append(alternative)
            # update best plan for each scenario
            for scenario in self.tep_model.tep_system.scenarios:
                if self.best_plan_scenario[scenario] is None \
                        or alternative.total_costs[scenario] < self.best_plan_scenario[scenario].total_costs[scenario]:
                    self.best_plan_scenario[scenario] = alternative

    def get_dominated_alternatives(self):
        return (alt for alt in self.all_alternatives if alt not in self.efficient_alternatives)

    def plot_alternatives(self, filename):
        def alternatives_to_plot_list(s1, s2, alternatives):
            x = list(alternative.total_costs[s1]
                     for alternative in alternatives)
            y = list(alternative.total_costs[s2]
                     for alternative in alternatives)
            tags = list(alternative.get_plan_id()
                        for alternative in alternatives)
            return [x, y, tags]

        assert len(self.tep_model.tep_system.scenarios) == 2
        first_scenario = self.tep_model.tep_system.scenarios[0]
        second_scenario = self.tep_model.tep_system.scenarios[1]
        # plot efficient alternatives and frontier
        efficient_alts = sorted(self.efficient_alternatives, key=lambda a: a.total_costs[first_scenario])
        [x_efficient, y_efficient, tags_efficient] = alternatives_to_plot_list(first_scenario, second_scenario,
                                                                               efficient_alts)
        trace_efficient = Scatter(
            x=x_efficient,
            y=y_efficient,
            name='Efficient',
            mode='lines+markers',
            marker=dict(
                size=10
            ),
            text=tags_efficient
        )
        data_plotly = [trace_efficient]
        # plot dominated alternatives, if all alternatives were recorded
        if self.pareto_brute_force_params.save_all_alternatives:
            dominated_alternatives = list(self.get_dominated_alternatives())
            [x_dominated, y_dominated, tags_dominated] = alternatives_to_plot_list(first_scenario, second_scenario,
                                                                                   dominated_alternatives)
            trace_dominated = Scatter(
                x=x_dominated,
                y=y_dominated,
                mode='markers',
                name='Dominated',
                marker=dict(
                    size=5,
                    color='rgb(200,200,200)'
                ),
                text=tags_dominated
            )
            data_plotly.append(trace_dominated)
        # render plot
        show_grid_lines = True
        plot_layout = Layout(
            title="Pareto Front for TEP under Scenarios",
            hovermode='closest',
            xaxis=dict(
                title='Total Cost Scenario 1',
                showgrid=show_grid_lines
            ),
            yaxis=dict(
                title='Total Cost Scenario 2',
                showgrid=show_grid_lines
            )
        )
        plotly.offline.plot({
            "data": data_plotly,
            "layout": plot_layout
        },
            filename=filename)


class ScenariosTepParetoFrontByBruteForceSummary(object):
    """Summarizes results of a TEP pareto front"""

    def __init__(self, pareto_brute):
        # type: (ScenariosTepParetoFrontByBruteForce) -> None
        self.summary_pareto_front = collections.OrderedDict()
        self.summary_pareto_front['Elapsed time'] = str(timedelta(seconds=pareto_brute.elapsed_seconds))
        self.summary_pareto_front['Number of Efficient Alternatives'] = len(pareto_brute.efficient_alternatives)
        self.optimal_plans = collections.OrderedDict()
        for scenario in pareto_brute.tep_model.tep_system.scenarios:
            self.optimal_plans[scenario] = pareto_brute.best_plan_scenario[scenario]
            self.summary_pareto_front['Optimal Plan {}'.format(scenario.name)] = str(self.optimal_plans[scenario])
            self.summary_pareto_front['Optimal Cost {}'.format(scenario.name)] = \
                self.optimal_plans[scenario].total_costs[scenario]
        # Process efficient alternatives
        self.minimax_cost = float('inf')
        self.minimax_cost_plan = None
        self.minimax_regret = float('inf')
        self.minimax_regret_plan = None
        self.df_efficient_alternatives = None
        self.summarize_efficient_alternatives(pareto_brute.efficient_alternatives)
        self.summary_pareto_front['Minimax Cost Plan'] = str(self.minimax_cost_plan)
        self.summary_pareto_front['Minimax Cost'] = self.minimax_cost
        self.summary_pareto_front['Minimax Regret Plan'] = str(self.minimax_regret_plan)
        self.summary_pareto_front['Minimax Regret'] = self.minimax_regret
        self.df_summary_pareto_front = pd.DataFrame(self.summary_pareto_front, index=['Value']).transpose()

    def summarize_efficient_alternatives(self, efficient_alternatives):
        # type: (List[StaticTePlan], List[StaticTePlan], float, bool) -> object
        # TODO save details for each plan in the pareto solver, do not construct details again (it's idiotic)
        list_summaries_alternatives = []
        for alternative in efficient_alternatives:
            summary = StaticTePlanDetails(alternative).plan_summary
            summary['Kind'] = 'Efficient'  # if alternative in efficient_alternatives else 'Dominated'
            summary = pd.DataFrame(summary, index=['Plan{0}'.format(summary['Plan ID'])])
            list_summaries_alternatives.append(summary)
            # updates minimax cost
            max_cost_this_alternative = max(alternative.total_costs.values())
            if max_cost_this_alternative < self.minimax_cost:
                self.minimax_cost = max_cost_this_alternative
                self.minimax_cost_plan = alternative
            # updates minimax regret
            max_regret_this_alternative = alternative.max_regret(self.optimal_plans)
            if max_regret_this_alternative < self.minimax_regret:
                self.minimax_regret = max_regret_this_alternative
                self.minimax_regret_plan = alternative
        self.df_efficient_alternatives = pd.concat(list_summaries_alternatives)

    def to_excel(self, excel_filename, sheetname=['ParetoAlternatives', 'ParetoSummary']):
        # TODO save detailed solution excel for all efficient alternatives
        writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
        self.to_excel_sheet(writer, sheetname=sheetname)
        writer.save()

    def to_excel_sheet(self, writer, sheetname=['ParetoAlternatives', 'ParetoSummary']):
        Utils.df_to_excel_sheet_autoformat(self.df_summary_pareto_front, writer, sheetname[1])
        Utils.df_to_excel_sheet_autoformat(self.df_efficient_alternatives, writer, sheetname[0])
