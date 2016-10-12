import pandas as pd
import PowerSystem as pws
import PowerSystemState as pwstate


class PowerSystemScenario:
    def __init__(self, system, name):
        assert isinstance(system, pws.PowerSystem)
        self.system = system
        self.name = name
        self.states = []

    # excel_filepath=r"C:\Users\cvelasquez\Google Drive\2016 Paper TEP IEEEGM2017\07 Casos de estudio\Python\4bus_PowerSystem.xlsx"
    @staticmethod
    def import_scenarios_from_excel(excel_filepath):
        imported_system = pws.PowerSystem.import_from_excel(excel_filepath)
        imported_scenarios = []
        # import ScenariosDefinition
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
        # import ScenarioData
        df_scenarios_data = pd.read_excel(excel_filepath, sheetname="ScenarioData")
        for index, row in df_scenarios_data.iterrows():
            for scenario in imported_scenarios:
                for state in scenario.states:
                    node_state = next(
                        x for x in state.node_states if x.node.name == row['Node'])  # type: pwstate.NodeState
                    node_state.load_state = row["Load-{0}-{1}".format(scenario.name, state.name)]
                    node_state.available_generating_capacity = row["Gx-{0}-{1}".format(scenario.name, state.name)]
                    node_state.generation_marginal_cost = row["MG-{0}-{1}".format(scenario.name, state.name)]
        return imported_scenarios
