import PowerSystem


class PowerSystemState(object):
    def __init__(self, system, name, duration):
        self.system = system
        self.name = name
        self.duration = duration
        self.node_states = []
        for node in self.system.nodes:
            self.node_states.append(NodeState(self, node, 0.0, 0.0, 0.0))
        self.transmission_lines_states = []
        for line in self.system.transmission_lines:
            self.transmission_lines_states.append(TransmissionLineState(self, line, True))
            # self.available_lines_states = (line_state for line_state
            #                                in self.transmission_lines_states
            #                                if line_state.isavailable)

    def find_node_state(self, node):
        return next(node_state for node_state in self.node_states if node_state.node == node)

    def __str__(self):
        return self.name


class PowerSystemElementState(object):
    def __init__(self, system_state, element):
        assert isinstance(system_state, PowerSystemState)
        self.system_state = system_state
        assert isinstance(element, PowerSystem.PowerSystemElement)
        self.element = element

    def __str__(self):
        return self.element.name


class NodeState(PowerSystemElementState):
    def __init__(self, system_state, node,
                 load_state, available_generating_capacity, generation_marginal_cost):
        PowerSystemElementState.__init__(self, system_state, node)
        self.node = node
        self.load_state = load_state
        assert available_generating_capacity <= node.installed_generating_capacity
        self.available_generating_capacity = available_generating_capacity
        self.generation_marginal_cost = generation_marginal_cost

    def find_incoming_lines_states(self):
        return (line_state for line_state in self.system_state.transmission_lines_states
                if line_state.node_to_state == self)

    def find_outgoing_lines_states(self):
        return (line_state for line_state in self.system_state.transmission_lines_states
                if line_state.node_from_state == self)

    def __str__(self):
        return self.node.name


class TransmissionLineState(PowerSystemElementState):
    def __init__(self, system_state, transmission_line, isavailable):
        assert isinstance(transmission_line, PowerSystem.TransmissionLine)
        PowerSystemElementState.__init__(self, system_state, transmission_line)
        self.transmission_line = transmission_line
        self.node_from_state = system_state.find_node_state(transmission_line.node_from)
        self.node_to_state = system_state.find_node_state(transmission_line.node_to)
        self.isavailable = isavailable