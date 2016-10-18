"""Power system planning classes"""
from PowerSystemScenario import PowerSystemScenario
import pandas as pd
import itertools


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

    def is_equivalent(self, other_candidate):
        """Determines whether two candidate transmission lines are equivalent
        (despite being modeled as two alternative candidate lines)"""
        # type: (CandidateTransmissionLine) -> bool
        return self.transmission_line.is_equivalent(other_candidate.transmission_line) \
               and self.investment_cost == other_candidate.investment_cost

    def __str__(self):
        return str(self.transmission_line)


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
        # commit to each state in each scenario
        for scenario in self.scenarios:
            for state in scenario.states:
                # check only for candidate transmission lines
                for line_state in state.transmission_lines_states:
                    if self.transmission_line_is_candidate(line_state.transmission_line):
                        line_state.isavailable = line_state.transmission_line in (l.transmission_line
                                                                                  for l in lines_to_build)

    def transmission_line_is_candidate(self, transmission_line):
        return transmission_line in (candidate.transmission_line
                                     for candidate in self.candidate_lines)

    @staticmethod
    def get_sets_of_equivalent_lines_idx(candidates_lines):
        # type: (List[CandidateTransmissionLine]) -> List[List[CandidateTransmissionLine]]
        n = len(candidates_lines)
        equivalent_lines_idx = []
        for i in range(n):
            candidate1 = candidates_lines[i]  # type: CandidateTransmissionLine
            equivalent_lines_to_c1_idx = [i]
            for j in range(i, n):
                candidate2 = candidates_lines[j]  # type: CandidateTransmissionLine
                if candidate1.is_equivalent(candidate2):
                    equivalent_lines_to_c1_idx.append(j)
            if len(equivalent_lines_to_c1_idx) > 1:
                equivalent_lines_idx.append(equivalent_lines_to_c1_idx)
        return equivalent_lines_idx

    @staticmethod
    def get_sets_of_equivalent_lines(candidates_lines):
        # type: (List[CandidateTransmissionLine]) -> List[List[CandidateTransmissionLine]]
        n = len(candidates_lines)
        equivalent_lines = []
        for i in range(n):
            candidate1 = candidates_lines[i]  # type: CandidateTransmissionLine
            equivalent_lines_to_c1 = []
            for j in range(i, n):
                candidate2 = candidates_lines[j]  # type: CandidateTransmissionLine
                if candidate1.is_equivalent(candidate2):
                    equivalent_lines_to_c1.append(candidate2)
            if len(equivalent_lines_to_c1) > 0:
                equivalent_lines.append(equivalent_lines_to_c1)
        return equivalent_lines

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
