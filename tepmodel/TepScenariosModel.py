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
import random


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
    def from_binary_gene(tep_model, candidate_lines_binary_gene):
        # type: (TepScenariosModel, List) -> StaticTePlan
        return StaticTePlan(tep_model,
                            [tep_model.tep_system.candidate_lines[i]
                             for i in range(len(candidate_lines_binary_gene))
                             if candidate_lines_binary_gene[i]])

    @staticmethod
    def from_id(tep_model, plan_id):
        # type: (TepScenariosModel, int) -> StaticTePlan
        plan = StaticTePlan(tep_model,
                            candidate_lines_built=Utils.subset_from_id(tep_model.tep_system.candidate_lines,
                                                                       plan_id))
        return plan

    @staticmethod
    def is_first_unique_by_id(plan_id, equivalent_lines_idx):
        for equivalent_lines_idx_group in equivalent_lines_idx:
            for i in range(1, len(equivalent_lines_idx_group)):
                idx = equivalent_lines_idx_group[i]
                idx_prev = equivalent_lines_idx_group[i - 1]
                if (plan_id >> idx) & 1 and not (plan_id >> idx_prev) & 1:
                    # the line in idx is active, but the previous (and equivalent) line is not active
                    # hence another equivalent plan has lower id
                    # and thus it was already processed by an exhaustive enumerator
                    return False
        return True

    @staticmethod
    def is_first_unique(candidate_lines_built, equivalent_lines):
        for equivalent_lines_group in equivalent_lines:
            for i in range(1, len(equivalent_lines_group)):
                if equivalent_lines_group[i] in candidate_lines_built \
                        and equivalent_lines_group[i - 1] not in candidate_lines_built:
                    return False
        return True

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
    def __init__(self, save_extra_alternatives=False,
                 save_alternative_handle=None, save_alternative_handle_params=None,
                 plans_processed_for_reporting=None):
        # type: (bool, object, object, int) -> None
        """Parameters for brute-force construction of pareto front

        :param save_extra_alternatives: True if all alternatives are to be saved, following some filtering criteria
        :param save_alternative_handle: Function handle to filter which alternatives to save
        :param save_alternative_handle_params: Parameters for the filtering function
        :param plans_processed_for_reporting: Number of plans that must be processed before reporting progress
        """
        self.save_extra_alternatives = save_extra_alternatives
        self.save_alternative_handle = save_alternative_handle
        self.save_alternative_handle_params = save_alternative_handle_params
        self.plans_processed_for_reporting = plans_processed_for_reporting

    def save_alternative(self, pareto_brute, alternative):
        """Determines whether the given alternative must be recorded while building the pareto front

        :param pareto_brute: The Brute-force pareto builder instance
        :param alternative: The alternative transmission expansion plan
        :return: True if the alternative should be recorded, false otherwise
        """
        return self.save_alternative_handle(pareto_brute, alternative, self.save_alternative_handle_params)

    @staticmethod
    def save_all_alternatives(pareto_brute, alternative, options=None):
        # type: (ScenariosTepParetoFrontByBruteForce, StaticTePlan, object) -> bool
        return True

    @staticmethod
    def save_random_alternatives(pareto_brute, alternative, options=0.2):
        # type: (ScenariosTepParetoFrontByBruteForce, StaticTePlan, float) -> bool
        return random.random() > options

    @staticmethod
    def save_maxrandom_alternatives(pareto_brute, alternative, options=[0.1, 200]):
        # type: (ScenariosTepParetoFrontByBruteForce, StaticTePlan, float) -> bool
        # r = random.random()
        # b = len(pareto_brute.extra_alternatives) < options[1] and r < options[0]
        # if b:
        #     logging.info("Recording extra plan {} ({} plans currently recorded), r={}".
        #                  format(alternative.get_plan_id(), len(pareto_brute.extra_alternatives), r)
        #                  )
        return len(pareto_brute.extra_alternatives) < options[1] and random.random() < options[0]

    @staticmethod
    def save_alternatives_in_range(pareto_brute, alternative, options=None):
        # type: (ScenariosTepParetoFrontByBruteForce, StaticTePlan, Dict[PowerSystemScenario, List[float]]) -> bool
        assert options is not None
        for scenario in pareto_brute.tep_model.tep_system.scenarios:
            obj_range = options[scenario]
            obj_plan = alternative.total_costs[scenario]
            if min(obj_range) > obj_plan or max(obj_range) < obj_plan:
                return False
        return True


class ScenariosTepParetoFrontBuilder(object):
    def __init__(self, tep_model):
        # type: (TepScenariosModel) -> None
        self.tep_model = tep_model
        # Plans in pareto front, set by function 'build_pareto_front'
        self.efficient_alternatives = []  # type: list[StaticTePlan]
        # Optimal plan for each scenario, set by function 'build_pareto_front'
        self.optimal_plans = collections.OrderedDict()
        for scenario in self.tep_model.tep_system.scenarios:
            self.optimal_plans[scenario] = None
        # Summary: these are filled in the function 'summarize',
        # which must be called upon completion of 'build_pareto_front' by implementing classes
        self.start_time = None
        self.elapsed_seconds = None
        self.summary_pareto_front = collections.OrderedDict()
        self.df_summary_pareto_front = None
        self.minimax_cost = float('inf')
        self.minimax_cost_plan = None
        self.minimax_regret = float('inf')
        self.minimax_regret_plan = None
        self.df_efficient_alternatives = None  # set by function 'summarize_efficient_alternatives'

    def execute_build_pareto_front(self, save_to_excel=False, excel_filename=''):
        self.start_time = time.clock()
        self.build_pareto_front()
        self.elapsed_seconds = time.clock() - self.start_time
        logging.info("Finished building pareto front, proceeding to summarize results..")
        self.summarize()
        if save_to_excel:
            self.to_excel(excel_filename)

    def build_pareto_front(self):
        # type: () -> List[StaticTePlan]
        pass

    def summarize(self):
        self.summary_pareto_front['Elapsed time'] = str(timedelta(seconds=self.elapsed_seconds))
        self.summary_pareto_front['Number of Efficient Alternatives'] = len(self.efficient_alternatives)
        for scenario in self.tep_model.tep_system.scenarios:
            self.summary_pareto_front['Optimal Plan {}'.format(scenario.name)] = str(self.optimal_plans[scenario])
            self.summary_pareto_front['Optimal Cost {}'.format(scenario.name)] = \
                self.optimal_plans[scenario].total_costs[scenario]
        # Process and summarize efficient alternatives
        self.summarize_efficient_alternatives()
        self.summary_pareto_front['Minimax Cost Plan'] = str(self.minimax_cost_plan)
        self.summary_pareto_front['Minimax Cost'] = self.minimax_cost
        self.summary_pareto_front['Minimax Regret Plan'] = str(self.minimax_regret_plan)
        self.summary_pareto_front['Minimax Regret'] = self.minimax_regret
        self.df_summary_pareto_front = pd.DataFrame(self.summary_pareto_front, index=['Value']).transpose()

    def summarize_efficient_alternatives(self):
        # type: (List[StaticTePlan], List[StaticTePlan], float, bool) -> object
        # TODO save details for each plan in the pareto solver, do not construct details again (it's idiotic)
        list_summaries_alternatives = []
        for alternative in self.efficient_alternatives:
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
        writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
        self.to_excel_sheet(writer, sheetname=sheetname)
        writer.save()

    def to_excel_sheet(self, writer, sheetname=['ParetoAlternatives', 'ParetoSummary']):
        Utils.df_to_excel_sheet_autoformat(self.df_summary_pareto_front, writer, sheetname[1])
        Utils.df_to_excel_sheet_autoformat(self.df_efficient_alternatives, writer, sheetname[0])

    @staticmethod
    def alternatives_to_plot_list(s1, s2, alternatives):
        x = list(alternative.total_costs[s1]
                 for alternative in alternatives)
        y = list(alternative.total_costs[s2]
                 for alternative in alternatives)
        tags = list(alternative.get_plan_id()
                    for alternative in alternatives)
        return [x, y, tags]

    def build_plot_alternatives(self):
        assert len(self.tep_model.tep_system.scenarios) == 2
        first_scenario = self.tep_model.tep_system.scenarios[0]
        second_scenario = self.tep_model.tep_system.scenarios[1]
        # plot efficient alternatives with frontier
        efficient_alts = sorted(self.efficient_alternatives, key=lambda a: a.total_costs[first_scenario])
        [x_efficient, y_efficient, tags_efficient] = ScenariosTepParetoFrontBuilder.alternatives_to_plot_list(
            first_scenario, second_scenario,
            efficient_alts)
        trace_efficient = Scatter(x=x_efficient,
                                  y=y_efficient,
                                  name='Efficient',
                                  mode='lines+markers',
                                  marker=dict(size=10),
                                  text=tags_efficient)
        self.data_plotly = [trace_efficient]
        # render plot
        show_grid_lines = True
        self.plot_layout = Layout(
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

    def plot_alternatives_offline(self, filename):
        plotly.offline.plot({"data": self.data_plotly,
                             "layout": self.plot_layout},
                            filename=filename)


class ScenariosTepParetoFrontByBruteForce(ScenariosTepParetoFrontBuilder):
    def __init__(self, tep_model,
                 pareto_brute_force_params=ScenariosTepParetoFrontByBruteForceParams()):
        # type: (TepScenariosModel, ScenariosTepParetoFrontByBruteForceParams) -> None
        ScenariosTepParetoFrontBuilder.__init__(self, tep_model)
        self.pareto_brute_force_params = pareto_brute_force_params
        if self.pareto_brute_force_params.save_extra_alternatives:
            self.extra_alternatives = []  # type: list[StaticTePlan]
        # enumerate and evaluate all possible alternatives
        self.number_of_maximum_possible_plans = self.tep_model.get_numberof_possible_expansion_plans()
        self.number_of_processed_plans = 0

    def build_pareto_front(self):
        # type: () -> List[StaticTePlan]
        equivalent_lines_idx = pwsp.PowerSystemTransmissionPlanning.get_sets_of_equivalent_lines_idx(
            self.tep_model.tep_system.candidate_lines)
        iterator_plan_ids = (i for i in range(self.number_of_maximum_possible_plans)
                             if StaticTePlan.is_first_unique_by_id(i, equivalent_lines_idx))
        for plan_id in iterator_plan_ids:
            # build and solve plan
            alternative = StaticTePlan.from_id(self.tep_model, plan_id)
            # update set of efficient plans with the newly created plan
            self.update_set_of_efficient_plans(alternative)
            # record plan anyway if all alternatives are being saved
            if self.pareto_brute_force_params.save_extra_alternatives \
                    and self.pareto_brute_force_params.save_alternative(self, alternative):
                self.extra_alternatives.append(alternative)
            # report progress
            self.number_of_processed_plans += 1
            if (self.number_of_processed_plans % self.pareto_brute_force_params.plans_processed_for_reporting) == 0:
                elapsed_time = time.clock() - self.start_clock
                logging.info("Processed {0} alternatives (elapsed: {3}; current ID={1} / {2})...".
                             format(self.number_of_processed_plans,
                                    plan_id, self.number_of_maximum_possible_plans,
                                    str(timedelta(seconds=elapsed_time))
                                    )
                             )
        return self.efficient_alternatives

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
                if self.optimal_plans[scenario] is None \
                        or alternative.total_costs[scenario] < self.optimal_plans[scenario].total_costs[scenario]:
                    self.optimal_plans[scenario] = alternative

    def get_dominated_alternatives(self):
        return (alt for alt in self.extra_alternatives if alt not in self.efficient_alternatives)

    def summarize(self):
        ScenariosTepParetoFrontBuilder.summarize(self)
        # Summarize extra alternatives
        self.df_extra_alternatives = None
        if self.pareto_brute_force_params.save_extra_alternatives:
            list_summaries_extra_alternatives = []
            logging.info("Proceeding to summarize {} extra alternatives".format(len(self.extra_alternatives)))
            startsumm = time.clock()
            for idx, alternative in enumerate(self.extra_alternatives):
                summary = StaticTePlanDetails(alternative).plan_summary
                summary['Kind'] = 'Efficient' if alternative in pareto_brute.efficient_alternatives else 'Dominated'
                summary = pd.DataFrame(summary, index=['Plan{0}'.format(summary['Plan ID'])])
                list_summaries_extra_alternatives.append(summary)
                if idx % 100 == 0:
                    elapsed_secs = time.clock() - startsumm
                    logging.info("Summarized {} / {} extra alternatives (elapsed: {})".
                                 format(idx, len(pareto_brute.extra_alternatives), timedelta(seconds=elapsed_secs))
                                 )
            self.df_extra_alternatives = pd.concat(list_summaries_extra_alternatives)

    def to_excel_sheet(self, writer, sheetname=['ParetoAlternatives', 'ParetoSummary', 'DominatedAlternatives']):
        ScenariosTepParetoFrontBuilder.to_excel_sheet(writer, sheetname[0:1])
        if self.df_extra_alternatives is not None:
            Utils.df_to_excel_sheet_autoformat(self.df_extra_alternatives, writer, sheetname[2])

    def build_plot_alternatives(self):
        ScenariosTepParetoFrontBuilder.build_plot_alternatives(self)
        # plot dominated alternatives, if all alternatives were recorded
        if self.pareto_brute_force_params.save_extra_alternatives:
            first_scenario = self.tep_model.tep_system.scenarios[0]
            second_scenario = self.tep_model.tep_system.scenarios[1]
            dominated_alternatives = list(self.get_dominated_alternatives())
            [x_dominated, y_dominated, tags_dominated] = ScenariosTepParetoFrontBuilder.alternatives_to_plot_list(
                first_scenario, second_scenario, dominated_alternatives)
            trace_dominated = Scatter(x=x_dominated,
                                      y=y_dominated,
                                      mode='markers',
                                      name='Dominated',
                                      marker=dict(size=3, color='rgb(200,200,200)'),
                                      text=tags_dominated)
            self.data_plotly.append(trace_dominated)
