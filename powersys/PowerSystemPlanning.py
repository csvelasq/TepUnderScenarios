import PowerSystem as pws
import pandas as pd


class PowerSystemTransmissionPlanning(object):
    def __init__(self, system, candidate_lines):
        self.system = system
        self.candidate_lines = candidate_lines

    @staticmethod
    def import_from_excel(excel_filepath):
        # type: (str) -> PowerSystemTransmissionPlanning
        system = pws.PowerSystem.import_from_excel(excel_filepath)
        df_candidate_lines = pd.read_excel(excel_filepath, sheetname="CandidateTransmissionLines")
        candidate_lines = []
        for index, row in df_candidate_lines.iterrows():
            transmission_line = next(line for line in system.transmission_lines
                                     if line.name == row['name'])
            candidate_line = CandidateTransmissionLine(transmission_line,
                                                       row['investment_cost'])
            candidate_lines.append(candidate_line)
        imported_planning_system = PowerSystemTransmissionPlanning(system, candidate_lines)
        return imported_planning_system


class CandidateTransmissionLine(object):
    def __init__(self, transmission_line, investment_cost):
        self.transmission_line = transmission_line
        self.investment_cost = investment_cost
