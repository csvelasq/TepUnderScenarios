import pandas as pd
import PowerSystem as pws
import PowerSystemState as pwstate
import logging
import numbers


class PowerSystemScenario(object):
    def __init__(self, system, name):
        # type: (powersys.PowerSystem.PowerSystem, unicode) -> None
        self.system = system
        self.name = name
        self.states = []

    def __str__(self):
        return self.name

    @staticmethod
    def import_scenarios_from_excel(excel_filepath, system=None):
        # type: (str) -> list[PowerSystemScenario]
        if system is None:
            imported_system = pws.PowerSystem.import_from_excel(excel_filepath)
        else:
            imported_system = system
        imported_scenarios = []
        # import definition of scenarios (consisting of a name only)
        df_scenarios_definition = pd.read_excel(excel_filepath, sheetname="ScenariosDefinition")
        scenario_names = df_scenarios_definition.ScenarioName.unique()
        for name in scenario_names:
            scenario = PowerSystemScenario(imported_system, name)
            imported_scenarios.append(scenario)
        # import definition of each state for each scenario
        for index, row in df_scenarios_definition.iterrows():
            # find the scenario which matches the scenario name in this row
            scenario = next(x for x in imported_scenarios if x.name == row['ScenarioName'])
            # create power system state and add to selected scenario
            state = pwstate.PowerSystemState(imported_system,
                                             row['StateName'],
                                             row['Duration'])
            scenario.states.append(state)
        # import load scenario data
        df_scenarios_load_data = pd.read_excel(excel_filepath, sheetname="ScenarioLoadData")
        for index, row in df_scenarios_load_data.iterrows():
            for scenario in imported_scenarios:
                for state in scenario.states:
                    node_name = row['Node']
                    if isinstance(node_name, numbers.Number):
                        node_name = str(int(node_name))
                    node_state = next(x for x in state.node_states
                                      if x.node.name == node_name)  # type: pwstate.NodeState
                    node_state.load_state = row["Load-{0}-{1}".format(scenario.name, state.name)]
        # import generators scenario data
        df_scenarios_gen_data = pd.read_excel(excel_filepath, sheetname="ScenarioGenData")
        for index, row in df_scenarios_gen_data.iterrows():
            for scenario in imported_scenarios:
                for state in scenario.states:
                    generator_state = next(x for x in state.generators_states
                                           if x.generator.name == row['Generator'])  # type: pwstate.GeneratorState
                    generator_state.available_generating_capacity = row["Gx-{0}-{1}".format(scenario.name, state.name)]
                    generator_state.generation_marginal_cost = row["MG-{0}-{1}".format(scenario.name, state.name)]
        return imported_scenarios
