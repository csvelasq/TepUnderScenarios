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
        self.start_clock = time.clock()
        for plan_id in range(self.number_of_possible_plans):
            # build and solve plan
            alternative = StaticTePlan.from_id(self.tep_model, plan_id)
            # update efficient set of plans with the newly created plan
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
            # record plan anyway if all alternatives are being saved
            if self.pareto_brute_force_params.save_all_alternatives:
                self.all_alternatives.append(alternative)
            # report progress
            if (plan_id % self.pareto_brute_force_params.plans_processed_for_reporting) == 0:
                self.elapsed_time = time.clock() - self.start_clock
                logging.info(("Processed {0} / {1} alternatives "
                              "(elapsed: {2})...").format(plan_id, self.number_of_possible_plans,
                                                          str(timedelta(seconds=self.elapsed_time))
                                                          )
                             )

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
        self.save_all_alternatives = pareto_brute.pareto_brute_force_params.save_all_alternatives
        if self.save_all_alternatives:
            # Process all alternatives if the user selected such option
            self.df_all_alternatives = ScenariosTepParetoFrontByBruteForceSummary.summarize_alternatives(
                pareto_brute.all_alternatives, pareto_brute.efficient_alternatives,
                alternatives_to_report_progress=100)
            self.df_efficient_alternatives = self.df_all_alternatives.loc[
                self.df_all_alternatives['Kind'] == 'Efficient']
        else:
            # Process only efficient alternatives
            self.df_efficient_alternatives = ScenariosTepParetoFrontByBruteForceSummary.summarize_alternatives(
                pareto_brute.efficient_alternatives, pareto_brute.efficient_alternatives)

    @staticmethod
    def summarize_alternatives(alternatives, efficient_alternatives, alternatives_to_report_progress=None):
        list_summaries_alternatives = []
        number_of_alternatives_to_process = len(alternatives)
        number_of_processed_alternatives = 0
        start_clock = time.clock()
        for alternative in alternatives:
            summary = StaticTePlanDetails(alternative).plan_summary
            summary['Kind'] = 'Efficient' if alternative in efficient_alternatives else 'Dominated'
            summary = pd.DataFrame(summary, index=['Plan{0}'.format(summary['Plan ID'])])
            list_summaries_alternatives.append(summary)
            number_of_processed_alternatives += 1
            if alternatives_to_report_progress is not None:
                if number_of_processed_alternatives % alternatives_to_report_progress == 0:
                    elapsed_time = time.clock() - start_clock
                    logging.info(("Processed {0} / {1} alternatives "
                                  "(elapsed: {2})...").format(number_of_processed_alternatives,
                                                              number_of_alternatives_to_process,
                                                              str(timedelta(seconds=elapsed_time))
                                                              )
                                 )
        return pd.concat(list_summaries_alternatives)

    def to_excel(self, excel_filename, sheetname='ParetoAlternatives'):
        # TODO save detailed solution excel for all efficient alternatives
        writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
        self.to_excel_sheet(writer, sheetname=sheetname)
        writer.save()

    def to_excel_sheet(self, writer, sheetname='ParetoAlternatives'):
        if self.save_all_alternatives:
            Utils.df_to_excel_sheet_autoformat(self.df_all_alternatives, writer, sheetname)
        else:
            Utils.df_to_excel_sheet_autoformat(self.df_efficient_alternatives, writer, sheetname)
