"""Power system planning classes"""
from PowerSystemScenario import PowerSystemScenario
import pandas as pd


class CandidateTransmissionLine(object):
    """A candidate transmission line, not currently built"""

    def __init__(self, transmission_line, investment_cost):
        self.transmission_line = transmission_line
        self.investment_cost = investment_cost

    @staticmethod
    def import_from_excel(system, excel_filepath):
        # type: (powersys.PowerSystem.PowerSystem, str) -> list[CandidateTransmissionLine]
        df_candidate_lines = pd.read_excel(excel_filepath, sheetname="CandidateTransmissionLines")
        candidate_lines = []
        for index, row in df_candidate_lines.iterrows():
            transmission_line = next(line for line in system.transmission_lines
                                     if line.name == row['name'])
            candidate_line = CandidateTransmissionLine(transmission_line,
                                                       row['investment_cost'])
            candidate_lines.append(candidate_line)
        return candidate_lines

    def __str__(self):
        return self.transmission_line.name


class PowerSystemTransmissionPlanning(object):
    """A transmission expansion planning setup"""

    def __init__(self, scenarios, candidate_lines):
        # type: (List[powersys.PowerSystemScenario.PowerSystemScenario], List[powersys.PowerSystemPlanning.CandidateTransmissionLine]) -> None
        self.system = scenarios[0].system
        self.scenarios = scenarios
        self.candidate_lines = candidate_lines

    def commit_build_lines(self, lines_to_build):
        """Modifies internal scenarios and their states to activate only the selected candidate transmission lines,
        and deactivating all other candidate transmission lines"""
        # type: (list[CandidateTransmissionLine]) -> None
        for scenario in self.scenarios:
            for state in scenario.states:
                for line in state.transmission_lines_states:
                    line.isavailable = line in lines_to_build

    @staticmethod
    def import_from_excel(excel_filepath):
        # type: (str) -> PowerSystemTransmissionPlanning
        # import scenarios
        scenarios = PowerSystemScenario.import_scenarios_from_excel(excel_filepath)  # type: list[PowerSystemScenario]
        # import candidate transmission lines
        candidate_lines = CandidateTransmissionLine.import_from_excel(scenarios[0].system, excel_filepath)
        # build imported system
        imported_planning_system = PowerSystemTransmissionPlanning(scenarios, candidate_lines)
        return imported_planning_system
