import os.path
import pandas as pd


class PowerSystem(object):
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.transmission_lines = []

    def find_node(self, node_name):
        return next(node for node in self.nodes
                    if node.name == node_name)

    @staticmethod
    def import_from_excel(excel_filepath):
        # type: (str) -> PowerSystem
        name = os.path.splitext(os.path.basename(excel_filepath))[0]
        imported_system = PowerSystem(name)
        # import nodes
        df_nodes = pd.read_excel(excel_filepath, sheetname="Nodes")
        for index, row in df_nodes.iterrows():
            node = Node(imported_system,
                        row['name'],
                        row['load'],
                        row['installed_generating_capacity'],
                        row['generation_marginal_cost'])
            imported_system.nodes.append(node)
        # import transmission lines
        df_lines = pd.read_excel(excel_filepath, sheetname="TransmissionLines")
        for index, row in df_lines.iterrows():
            line = TransmissionLine(imported_system,
                                    row['name'],
                                    imported_system.find_node(row['node_from']),
                                    imported_system.find_node(row['node_to']),
                                    row['susceptance'],
                                    row['thermal_capacity'])
            imported_system.transmission_lines.append(line)
        return imported_system

    def __str__(self):
        return self.name


class PowerSystemElement(object):
    def __init__(self, system, name):
        self.system = system
        self.name = name

    def __str__(self):
        return self.name


class Node(PowerSystemElement):
    def __init__(self, system, name, load, installed_generating_capacity, generation_marginal_cost):
        PowerSystemElement.__init__(self, system, name)
        self.name = name
        self.load = load
        self.installed_generating_capacity = installed_generating_capacity
        self.generation_marginal_cost = generation_marginal_cost

    def get_maximum_gen_energy(self, hours):
        return self.installed_generating_capacity * hours


class TransmissionLine(PowerSystemElement):
    def __init__(self, system, name, node_from, node_to, susceptance, thermal_capacity):
        PowerSystemElement.__init__(self, system, name)
        self.name = name
        self.node_from = node_from
        self.node_to = node_to
        self.susceptance = susceptance
        self.thermal_capacity = thermal_capacity

    def is_equivalent(self, other_line):
        # type: (TransmissionLine) -> bool
        return self.node_from == other_line.node_from and self.node_to == other_line.node_to \
               and self.thermal_capacity == other_line.thermal_capacity \
               and self.susceptance == other_line.susceptance

    def __str__(self):
        return self.name + "({}-{})".format(self.node_from.name, self.node_to.name)
