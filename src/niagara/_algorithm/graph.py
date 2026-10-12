"""
DAG object & event classes
"""
import os
import json
import math
from collections import deque

import numpy as np


try:
    from graphviz import Digraph
    has_graphviz = True
except:
    has_graphviz = False

from .distributions import ExponentialDistribution


class Graph:
    def __init__(self):
        self.events = []
        self.src_nodes = []
        self.edges = []
        self.nodes = []
        self.logical_nodes = []
        self.non_logical_nodes = []

    def as_dict(self):
        graph_dict = dict()
        graph_dict['events'] = [e.name for e in self.events]
        graph_dict['causal_mechs'] = dict()
        for event in self.events:
            node = self.get_node(event.name, StartNode)
            if len(node.parents) > 0:
                graph_dict['causal_mechs'][event.name] = [node.parents[0].name]
        return graph_dict

    def init_from_events(self, events, s_i_edges=False):
        """
        Initialize a graph with SIE nodes for each event and no edges
        :param events: a list of event names
        :return: None
        """
        self.events = [Event(e) for e in events]
        for e in self.events:
            self.nodes += [e.s_node, e.i_node, e.e_node]
            if s_i_edges:
                self.edges.append(Edge(e.s_node, e.i_node, None))

    def update_node_lists(self):
        self.non_logical_nodes = [n for n in self.nodes if not isinstance(n, LogicalNode)]
        self.logical_nodes = [n for n in self.nodes if isinstance(n, LogicalNode)]

    def get_logical_nodes(self):
        if len(self.nodes) != len(self.logical_nodes) + len(self.non_logical_nodes):
            self.update_node_lists()
        return self.logical_nodes

    def get_non_logical_nodes(self):
        if len(self.nodes) != len(self.logical_nodes) + len(self.non_logical_nodes):
            self.update_node_lists()
        return self.non_logical_nodes

    def show(self, filename="graph", e_and=False):
        if not has_graphviz:
            print("Graph visualization requires Graphviz.")
            return
        ln = [n for n in self.nodes if isinstance(n, LogicalNode)]
        h_space = 1.5 if len(ln) > 0 else 1
        dot = Digraph(format="png", engine='neato', graph_attr={'splines': 'true'})

        with dot.subgraph() as sub:
            for ie, event in enumerate(self.events):
                dot.node(str(id(event.s_node)), f"{event.name}_S", shape="ellipse", pos=f"0,{str(ie * h_space)}!")
                dot.node(str(id(event.i_node)), f"{event.name}_I", shape="ellipse", pos=f"2,{str(ie * h_space)}!")
                dot.node(str(id(event.e_node)), f"{event.name}_E", shape="ellipse", pos=f"4,{str(ie * h_space)}!")
            #taken = {(r, c): True if r%2 == 1 else False for r in range(len(self.events)) for c in [1, 3]}
            taken = {(r+0.5, c): False for r in range(len(self.events)) for c in [1, 3]}
            for i, n in enumerate(ln):
                if not isinstance(n.children[0], EndNode):
                    pos_1 = self.events.index(n.parents[0].event)
                    pos_2 = self.events.index(n.parents[1].event)
                    pos = max(0.5, math.floor((pos_1 + pos_2) / 2) - 0.5)
                    success = False
                    attempt = int(i % 2 != 0)
                    while not success and False in taken.values():
                        if attempt % 2 == 0:
                            col = 1
                            pos = (pos + 1) % len(self.events)
                        else:
                            col = 3
                        if not taken[(pos, col)]:
                            position = f"{str(col)},{str(pos * h_space)}!"
                            taken[(pos, col)] = True
                            success = True
                        else:
                            attempt += 1
                    if False not in taken.values():
                        dot.node(str(id(n)), n.operator, shape="ellipse")
                    else:
                        dot.node(str(id(n)), n.operator, shape="ellipse", pos=position)

        # Add edges
        for edge in self.edges:
            if isinstance(edge.parent, LogicalNode) and isinstance(edge.child, EndNode) and edge.child.event.i_node in edge.parent.parents:
                continue
            if isinstance(edge.child, LogicalNode) and isinstance(edge.child.children[0], EndNode):
                if edge.parent.event == edge.child.children[0].event: continue
                dot.edge(str(id(edge.parent)), str(id(edge.child.children[0])))
            else:
                dot.edge(str(id(edge.parent)), str(id(edge.child)))

        # Render and display
        dot.render(filename, view=True)

    def topologicalSort(self):
        """Performs topological sort on a graph represented as an adjacency list."""
        graph = {n: n.children for n in self.nodes}  # adjacency list

        for n in graph.keys():
            if isinstance(n, IntervalNode): graph[n.event.e_node].append(n)  # we need to have ended the event to know when it is ongoing # because of that, we always have S -> E

        # Compute in-degree of each node
        in_degree = {node: 0 for node in graph}
        for neighbors in graph.values():
            for neighbor in neighbors:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

        # Collect nodes with zero in-degree (starting nodes)
        queue = deque([node for node in in_degree if in_degree[node] == 0])

        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)

            # Reduce in-degree of neighbors
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        for n in graph.keys():
            if isinstance(n, IntervalNode): graph[n.event.e_node].remove(n)

        # If the topological order contains all nodes, return it
        if len(topo_order) == len(graph):
            return topo_order
        else:
            return None

    def get_event(self, name):
        for e in self.events:
            if e.name == name: return e

    def get_edge(self, parent, child):
        if isinstance(child.parents[0], LogicalNode) and not isinstance(parent, LogicalNode):
            return [e for e in self.edges if e.parent == parent and e.child == child.parents[0]][0]
        return [e for e in self.edges if e.parent == parent and e.child == child][0]

    def get_node(self, event, type):
        """
        Return the node with corresponding type (SIE) and event name, if any
        :param event: str
        :param type: StartNode/IntervalNode/EndNode
        :return: Node
        """
        # do not return logical nodes
        res = [n for n in self.nodes if not isinstance(n, LogicalNode) and n.event.name == event and isinstance(n, type)]
        return res[0] if res else None

    def as_adj_matrix(self, e_with_i=False):
        """
        Creates an adjacency matrix where a cell is 1 if the row node has the column node as children
        :param e_with_i: when another event X causes Y^E, Y^I is also counted as parent
        :return: adjacency numpy matrix
        """
        n = len(self.get_non_logical_nodes())
        matrix = np.zeros(shape=(n, n))
        for i, node_1 in enumerate(self.get_non_logical_nodes()):
            children = [c if not isinstance(c, LogicalNode) else c.children[0] for c in node_1.children]
            for j, node_2 in enumerate(self.get_non_logical_nodes()):
                if node_2 in children:
                    matrix[i, j] = 1 # children must be correctly filled for logical nodes
                if isinstance(node_2, EndNode) and e_with_i:
                    matrix[j-1, j] = 1 # will only work if nodes are ordered by triplets SIE
                else:
                    matrix[j-1, j] = 0  # will only work if nodes are ordered by triplets SIE
                if isinstance(node_2, IntervalNode):
                    matrix[j-1, j] = 1 # will only work if nodes are ordered by triplets SIE
        return matrix

    def as_simple_dag_adj_matrix(self):
        """
        Creates an adjacency matrix for a DAG with one node per event (no SIE!) where a cell is 1 if the row event has the column event as children
        :return: adjacency numpy matrix
        """
        n = len([node for node in self.get_non_logical_nodes() if isinstance(node, StartNode)])
        matrix = np.zeros(shape=(n, n))
        i = -1
        for node_1 in self.get_non_logical_nodes():
            if isinstance(node_1, StartNode): i += 1
            j = -1
            children = [c if not isinstance(c, LogicalNode) else c.children[0] for c in node_1.children]
            for node_2 in self.get_non_logical_nodes():
                if isinstance(node_2, StartNode): j += 1
                if node_1.event == node_2.event: continue
                if node_2 in children:
                    matrix[i, j] = 1 # children must be correctly filled for logical nodes
        return matrix

    def as_adj_matrix_without_I(self):
        """
        Creates an adjacency matrix where a cell is 1 if the row node has the column node as children
        :return: adjacency numpy matrix
        """
        n = len([node for node in self.get_non_logical_nodes() if not isinstance(node, IntervalNode)])
        matrix = np.zeros(shape=(n, n))
        i = -1
        for node_1 in self.get_non_logical_nodes():
            if not isinstance(node_1, IntervalNode): i += 1
            j = -1
            children = [c if not isinstance(c, LogicalNode) else c.children[0] for c in node_1.children]
            for node_2 in self.get_non_logical_nodes():
                if not isinstance(node_2, IntervalNode): j += 1
                if isinstance(node_1, StartNode) and node_2 == node_1.event.i_node: continue
                if isinstance(node_1, IntervalNode) and node_2 == node_1.event.e_node: continue
                if node_2 in children:
                    matrix[i, j] = 1 # children must be correctly filled for logical nodes
        return matrix

    def export_graph(self, path, filename):
        g_dict = self.as_dict()
        with open(os.path.join(path, filename + '.json'), 'w') as f:
            json.dump(g_dict, f)
        dag = self.as_simple_dag_adj_matrix()
        np.save(os.path.join(path, filename + '.npy'), dag)

class Edge():
    def __init__(self, parent, child, distrib):
        self.parent = parent
        self.child = child
        self.distrib = distrib

    def get_edge_cost(self):
        if self.distrib is None:
            return 0  # S -> I
        else:
            return self.distrib.get_parameter_cost()

    def __repr__(self) -> str:
        return f"Edge({self.parent} -> {self.child}, distrib={self.distrib})"

    def is_eq(self, other):
        return type(self) == type(other) and self.parent.is_eq(other.parent) and self.child.is_eq(other.child)


class Node:
    def __init__(self, event, intensity=None, parents=None):
        self.event = event
        self.parents = [parents] if parents else []
        self.children = []
        self.distrib = ExponentialDistribution(intensity)  # for noise or source nodes
        self.name = event

    def add_child(self, child):
        if self not in child.parents:
            child.parents.append(self)
        if child not in self.children:
            self.children.append(child)

    def del_child(self, child):
        child.parents.remove(self)
        self.children.remove(child)

    def get_cost(self):
        return self.distrib.get_parameter_cost()

    def is_eq(self, other):
        return type(self) == type(other) and self.event.name == other.event.name

    def __repr__(self) -> str:
        return self.name


class LogicalNode(Node):
    def __init__(self, parents, operator):
        self.parents = parents  # list(Node)
        self.children = []
        self.operator = operator
        self.name = f"Node({self.parents[0]} {self.operator.upper()} {self.parents[1]})"

    def __repr__(self) -> str:
        return self.name

    def is_eq(self, other):
        return type(self) == type(other) and self.operator == other.operator and [
            any([pa.is_eq(opa) for pa in self.parents]) for opa in other.parents] == [True, True]


class StartNode(Node):
    def __init__(self, event, parents=None):
        super().__init__(event, parents)
        self.name = f"Node({self.event.name}_S)"


class EndNode(Node):
    def __init__(self, event, parents=None):
        super().__init__(event, parents)
        self.name = f"Node({self.event.name}_E)"


class IntervalNode(Node):
    def __init__(self, event, parents=None):
        super().__init__(event, parents)
        self.name = f"Node({self.event.name}_I)"


class Event:
    def __init__(self, name):
        self.name = name
        self.s_node = StartNode(self) # corresponds to the start of an occurrence of the event
        self.i_node = IntervalNode(self, self.s_node) # corresponds to the event being ongoing
        self.e_node = EndNode(self) # corresponds to the end of an occurrence of the event

    def __repr__(self):
        return f"Event({self.name})"

class EventOcc():
    def __init__(self, event, start, end=None, ts_index=0):
        self.event = event
        self.start = start
        self.end = end
        self.ts_index = ts_index

    def __repr__(self):
        return f"{self.event}({self.start}:{self.end})"

    def is_during(self, timestamp):
        return (self.start <= timestamp and self.end is None) or (self.start <= timestamp <= self.end)

