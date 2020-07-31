"""Power system planning classes"""
from tempfile import _candidate_tempdir_list

from .PowerSystemScenarios import PowerSystemScenario
from .PowerSystems import *
import pandas as pd
import itertools


class CandidateTransmissionLine(object):
    """A candidate transmission line, not currently built"""

    def __init__(self, transmission_line, investment_cost):
        self.transmission_line = transmission_line
        self.investment_cost = investment_cost

    @staticmethod
    def create_candidate_lines(system, excel_filepath):
        # type: (PowerSystem, unicode) -> List[List[CandidateTransmissionLine]]
        """Import candidate transmission lines (both in new and existing corridors)"""
        df_candidate_lines = pd.read_excel(excel_filepath, sheet_name="TransmissionLines")
        candidate_lines = []
        for index, row in df_candidate_lines.iterrows():
            max_new_lines = row['max_new_lines']
            if max_new_lines > 0:
                investment_cost = row['investment_cost']
                if row['is_new']:
                    transmission_line = TransmissionLine(system,
                                                             row['name'],
                                                             system.find_node(row['node_from']),
                                                             system.find_node(row['node_to']),
                                                             row['susceptance'],
                                                             row['thermal_capacity'])
                else:
                    transmission_line = next(line for line in system.transmission_lines
                                             if line.name == row['name'])
                # Creates new transmission lines in the power system model (one for each new candidate)
                candidates_lines_this_group = []
                for i in range(max_new_lines):
                    new_line_name = transmission_line.name + "_TC{}".format(i + 1)
                    new_line = TransmissionLine.copy_line(transmission_line, new_line_name)
                    system.transmission_lines.append(new_line)
                    candidates_lines_this_group.append(CandidateTransmissionLine(new_line, investment_cost))
                candidate_lines.append(candidates_lines_this_group)
        return candidate_lines

    def is_equivalent(self, other_candidate):
        """Determines whether two candidate transmission lines are equivalent
        (despite being modeled as two alternative candidate lines)"""
        # type: (CandidateTransmissionLine) -> bool
        return self.transmission_line.is_equivalent(other_candidate.transmission_line) \
               and self.investment_cost == other_candidate.investment_cost

    def __str__(self):
        return str(self.transmission_line)

    def to_str_nodes(self):
        return self.transmission_line.to_str_nodes()


class PowerSystemTransmissionPlanning(object):
    """A transmission expansion planning setup"""

    def __init__(self, scenarios, candidate_lines_groups):
        # type: (List[powersys.PowerSystemScenario.PowerSystemScenario], List[List[powersys.PowerSystemPlanning.CandidateTransmissionLine]]) -> None
        self.system = scenarios[0].system
        self.scenarios = scenarios
        self.candidate_lines_groups = candidate_lines_groups
        self.candidate_lines_flat_list = [item for sublist in self.candidate_lines_groups for item in sublist]

    def commit_build_lines(self, lines_to_build):
        """Modifies internal scenarios and their states to activate only the selected candidate transmission lines,
        and deactivating all other candidate transmission lines"""
        # type: (list[CandidateTransmissionLine]) -> None
        # commit to each state in each scenario
        for scenario in self.scenarios:
            for state in scenario.states:
                # check only for candidate transmission lines
                for line_state in state.transmission_lines_states:
                    line = line_state.transmission_line
                    if self.transmission_line_is_candidate(line):
                        line_state.isavailable = line in (l.transmission_line for l in lines_to_build)

    def transmission_line_is_candidate(self, transmission_line):
        return transmission_line in (candidate.transmission_line
                                     for candidate in self.candidate_lines_flat_list)

    def find_candidate_line_by_name(self, candidate_line_name):
        found_candidates = [l for l in self.candidate_lines_flat_list
                            if l.transmission_line.name == candidate_line_name]
        return found_candidates[0]

    @staticmethod
    def import_from_excel(excel_filepath):
        # type: (str) -> PowerSystemTransmissionPlanning
        # import power system
        system = PowerSystem.import_from_excel(excel_filepath)
        # import candidate transmission lines
        #   A list of groups of equivalent candidate transmission lines is created
        #   For each candidate line, a new transmission line is also created in the base power system
        candidate_lines_groups = CandidateTransmissionLine.create_candidate_lines(system, excel_filepath)
        # import scenarios
        scenarios = PowerSystemScenario.import_scenarios_from_excel(excel_filepath,
                                                                    system)  # type: list[PowerSystemScenario]
        # build imported system
        imported_planning_system = PowerSystemTransmissionPlanning(scenarios, candidate_lines_groups)
        return imported_planning_system
