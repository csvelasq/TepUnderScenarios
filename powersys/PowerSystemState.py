import PowerSystem


class PowerSystemState(object):
    # TODO produce matpower case file for validation
    def __init__(self, system, name, duration):
        self.system = system
        self.name = name
        self.duration = duration
        self.node_states = []  # type: List[NodeState]
        for node in self.system.nodes:
            self.node_states.append(NodeState(self, node, None))
        self.generators_states = []  # type: List[GeneratorState]
        for generator in self.system.generators:
            self.generators_states.append(GeneratorState(self, generator, None, None))
        self.transmission_lines_states = []  # type: List[TransmissionLineState]
        for line in self.system.transmission_lines:
            self.transmission_lines_states.append(TransmissionLineState(self, line, True))

    def find_node_state(self, node):
        """Find the state object of a given node object"""
        return next(node_state for node_state in self.node_states if node_state.node == node)

    def find_node_state_by_name(self, node_name):
        """Find the state object of a given node name"""
        return next(node_state for node_state in self.node_states if node_state.node.name == node_name)

    def __str__(self):
        return self.name

    def get_total_demand_power(self):
        return sum(n.load_state for n in self.node_states)

    def get_total_demand_energy(self):
        return self.get_total_demand_power() * self.duration / 1e3

    def get_total_gen_capacity(self):
        return sum(n.available_generating_capacity for n in self.node_states)


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
                 load_state=None):
        PowerSystemElementState.__init__(self, system_state, node)
        self.node = node
        if load_state is None:
            self.load_state = node.load
        else:
            self.load_state = load_state

    def find_connected_generators_states(self):
        return (generator_state for generator_state in self.system_state.generators_states
                if generator_state.generator.connection_node_state == self)

    def find_incoming_lines_states(self):
        return (line_state for line_state in self.system_state.transmission_lines_states
                if line_state.node_to_state == self)

    def find_outgoing_lines_states(self):
        return (line_state for line_state in self.system_state.transmission_lines_states
                if line_state.node_from_state == self)

    def __str__(self):
        return self.node.name


class GeneratorState(PowerSystemElementState):
    def __init__(self, system_state, generator,
                 available_generating_capacity=None,
                 generation_marginal_cost=None):
        PowerSystemElementState.__init__(self, system_state, generator)
        self.generator = generator
        self.connection_node_state = system_state.find_node_state(self.generator.connection_node)
        if available_generating_capacity is None:
            self.available_generating_capacity = generator.installed_generating_capacity
        else:
            self.available_generating_capacity = available_generating_capacity
        if generation_marginal_cost is None:
            self.generation_marginal_cost = generator.generation_marginal_cost
        else:
            self.generation_marginal_cost = generation_marginal_cost


class TransmissionLineState(PowerSystemElementState):
    def __init__(self, system_state, transmission_line, isavailable):
        assert isinstance(transmission_line, PowerSystem.TransmissionLine)
        PowerSystemElementState.__init__(self, system_state, transmission_line)
        self.transmission_line = transmission_line
        self.node_from_state = system_state.find_node_state(transmission_line.node_from)
        self.node_to_state = system_state.find_node_state(transmission_line.node_to)
        self.isavailable = isavailable
