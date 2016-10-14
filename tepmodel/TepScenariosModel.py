import OpfModel as opf
import powersys.PowerSystemPlanning as pwsp
import Utils
import plotly
from plotly.graph_objs import Scatter, Layout
import pandas as pd
import xlsxwriter


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
        opf_model_params = opf.OpfModelParameters(load_shedding_cost=params['load_shedding_cost'])
        tep_scenarios_model_parameters = \
            TepScenariosModelParameters(opf_model_params,
                                        investment_costs_multiplier=params['investment_costs_multiplier'],
                                        operation_costs_multiplier=params['operation_costs_multiplier'])
        imported_tep_model = TepScenariosModel(imported_tep_system, tep_scenarios_model_parameters)
        return imported_tep_model

    def evaluate_plan(self, plan, tep_scenarios_model_parameters=None):
        # type: (StaticTePlan) -> [dict[powersys.PowerSystemScenario.PowerSystemScenario, float], dict[powersys.PowerSystemScenario.PowerSystemScenario, float]]
        params = tep_scenarios_model_parameters
        if params is None:
            params = self.tep_scenarios_model_parameters
        total_costs = dict()
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

    def __init__(self, tep_model, candidate_lines_built=[],
                 tep_scenarios_model_parameters=None):
        # type: (TepScenariosModel) -> None
        self.tep_model = tep_model
        self.candidate_lines_built = candidate_lines_built
        self.tep_scenarios_model_parameters = tep_scenarios_model_parameters
        [self.scenario_simulation_model, self.total_costs, self.operation_costs] = self.tep_model.evaluate_plan(self,
                                                                                                                self.tep_scenarios_model_parameters)

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


class StaticTePlanDetails(object):
    """Details for a given static transmission expansion plan, mainly performance assessment under scenarios"""

    def __init__(self, plan):
        self.plan = plan  # type: StaticTePlan
        # get detailed solution
        self.scenarios_results = opf.ScenariosOpfModelResults(self.plan.scenario_simulation_model)

    def to_excel(self, filename):
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        self.to_excel_sheets(writer)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_excel_sheets(self, writer):
        self.scenarios_results.to_excel_sheets(writer)


class ScenariosTepParetoFrontByBruteForce(object):
    def __init__(self, tep_model):
        # type: (TepScenariosModel) -> None
        self.tep_model = tep_model
        self.alternatives = []  # type: list[StaticTePlan]
        self.efficient_alternatives = []  # type: list[StaticTePlan]
        # enumerate and evaluate all possible alternatives
        for plan_id in range(self.tep_model.get_numberof_possible_expansion_plans()):
            alternative = StaticTePlan.from_id(self.tep_model, plan_id)
            self.alternatives.append(alternative)
        # build pareto front
        for plan in self.alternatives:
            if plan.is_nondominated(self.alternatives):
                self.efficient_alternatives.append(plan)

    def get_dominated_alternatives(self):
        return (alt for alt in self.alternatives if alt not in self.efficient_alternatives)

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
        [x_efficient, y_efficient, tags_efficient] = alternatives_to_plot_list(first_scenario, second_scenario,
                                                                               self.efficient_alternatives)
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
        # plot dominated alternatives
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
            "data": [trace_efficient, trace_dominated],
            "layout": plot_layout
        },
            filename=filename)


class ScenariosTepParetoFrontByBruteForceSummary(object):
    """Summarizes results of a TEP pareto front"""

    def __init__(self, pareto_brute):
        # type: (ScenariosTepParetoFrontByBruteForce) -> None
        # dataframe with the summary solution for each alternative
        df_column_names = ['Plan ID',
                           'Kind',
                           'Number of Built Lines',
                           'Built Lines',
                           'Investment Cost [MMUS$]']
        for scenario in pareto_brute.tep_model.tep_system.scenarios:
            df_column_names.append('Operation Costs {0} [MMUS$]'.format(scenario.name))
            df_column_names.append('Total Costs {0} [MMUS$]'.format(scenario.name))
        self.df_alternatives = pd.DataFrame(columns=df_column_names)
        n = 0
        for alternative in pareto_brute.alternatives:
            kind = 'Efficient' if alternative in pareto_brute.efficient_alternatives else 'Dominated'
            row = [alternative.get_plan_id(),
                   kind,
                   len(alternative.candidate_lines_built),
                   map(str, alternative.candidate_lines_built),
                   alternative.get_total_investment_cost()]
            for scenario in pareto_brute.tep_model.tep_system.scenarios:
                row.append(alternative.operation_costs[scenario])
                row.append(alternative.total_costs[scenario])
            self.df_alternatives.loc[n] = row
            n += 1

    def to_excel(self, excel_filename, sheetname='Sheet1'):
        # self.df_alternatives.to_excel(excel_filename)

        # from: http://xlsxwriter.readthedocs.io/example_pandas_column_formats.html
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
        self.to_excel_sheet(writer, sheetname=sheetname)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_excel_sheet(self, writer, sheetname='Sheet1'):
        # Convert the dataframe to an XlsxWriter Excel object.
        self.df_alternatives.to_excel(writer, sheet_name=sheetname)
        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets[sheetname]
        # Add some cell formats.
        format_costs = workbook.add_format({'num_format': '$#,##0.00'})
        format_costs.set_text_wrap()
        # Set the column width and format.
        worksheet.set_column('F:J', 18, format_costs)
        worksheet.set_column('C:E', 18, None)
