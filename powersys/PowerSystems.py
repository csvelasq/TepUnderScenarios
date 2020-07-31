import os.path
import pandas as pd
import numbers


class PowerSystem(object):
    """A simple power system model consisting of buses with inelastic loads;
    generators with marginal costs and a given installed capacity;
    and transmission lines with capacity and susceptance"""

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.generators = []
        self.transmission_lines = []

    def find_node(self, node_name):
        if isinstance(node_name, numbers.Number):
            node_name = str(int(node_name))
        return next(node for node in self.nodes
                    if node.name == node_name)

    def find_generator(self, gen_name):
        return next(gen for gen in self.generators
                    if gen.name == gen_name)

    @staticmethod
    def import_from_excel(excel_filepath):
        # type: (str) -> PowerSystem
        name = os.path.splitext(os.path.basename(excel_filepath))[0]
        imported_system = PowerSystem(name)  # type: PowerSystem
        # import nodes
        df_nodes = pd.read_excel(excel_filepath, sheet_name="Nodes")
        for index, row in df_nodes.iterrows():
            node_name = row['name']
            if isinstance(node_name, numbers.Number):
                node_name = str(int(node_name))
            node = Node(imported_system,
                        node_name,
                        row['load'])
            imported_system.nodes.append(node)
        # import generators
        df_generators = pd.read_excel(excel_filepath, sheet_name="Generators")
        for index, row in df_generators.iterrows():
            connection_node = imported_system.find_node(row['node'])
            generator = Generator(imported_system,
                                  connection_node,
                                  row['name'],
                                  row['installed_generating_capacity'],
                                  row['generation_marginal_cost'])
            imported_system.generators.append(generator)
        # import transmission lines
        df_lines = pd.read_excel(excel_filepath, sheet_name="TransmissionLines")
        for index, row in df_lines.iterrows():
            if not row['is_new']:  # new transmission lines are added elsewhere
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

    def __lt__(self, other):
        # Only needed to avoid Pyomo errors when it attempts to print the model to an *.LP file (it attempts to sort sets)
        return self.name < other.name


class Node(PowerSystemElement):
    def __init__(self, system, name, load):
        PowerSystemElement.__init__(self, system, name)
        self.name = name
        self.load = load


class Generator(PowerSystemElement):
    def __init__(self, system, connection_node,
                 name, installed_generating_capacity, generation_marginal_cost):
        PowerSystemElement.__init__(self, system, name)
        self.connection_node = connection_node
        self.name = name
        self.installed_generating_capacity = installed_generating_capacity
        self.generation_marginal_cost = generation_marginal_cost


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

    @staticmethod
    def copy_line(other_line, name=None):
        # type: (TransmissionLine, unicode) -> TransmissionLine
        if name is None:
            my_name = other_line.name
        else:
            my_name = name
        return TransmissionLine(other_line.system, my_name,
                                other_line.node_from, other_line.node_to,
                                other_line.susceptance, other_line.thermal_capacity)

    def to_str_nodes(self):
        return "({}-{})".format(self.node_from.name, self.node_to.name)

    def __str__(self):
        return self.name  # + self.to_str_nodes()
