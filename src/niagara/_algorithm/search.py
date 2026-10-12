"""
Causal graph search algorithm
"""
import math
import numpy as np
import networkx as nx

from .graph import Graph, LogicalNode, StartNode, Node, Edge, EndNode, IntervalNode
from .scorer import Scorer
from .options import args

class Searcher():
    def __init__(self, n, data, empty_graph, exhaustive_search=True, verbose=False):
        self.n = n
        self.data = data
        self.graph = Graph()
        self.graph.init_from_events([e.name for e in empty_graph.events], s_i_edges=True)
        self.empty_graph = empty_graph
        self.stored_intervals = dict()
        self.exhaustive_search = exhaustive_search
        self.verbose = verbose

        self.targ_candidates = [n for n in self.graph.nodes]
        self.src_candidates = [n for n in self.graph.nodes if isinstance(n, StartNode)]
        self.scores = dict()

    def main(self):
        self.score_all_edges(all=self.exhaustive_search)
        priority_list = self.order_edges()
        for edge, _ in priority_list: # 5.1 Causal Mechanism Selection in paper
            if len(self.graph.get_node(edge.child.event.name, type(edge.child)).parents) > 0:
                continue
            if self.introduce_cycle(edge):
                if self.verbose: print(f'Edge discarded because would introduce a cycle: {edge}')
                continue
            self.add_edge(edge)

    def order_edges(self):
        edges = []
        for child, ch_scores in self.scores.items():
            if isinstance(child, StartNode):
                base_score = ch_scores[child]
            elif isinstance(child, EndNode):
                if args.start_only:
                    edges.append(([e for e in ch_scores if e.parent == child.event.s_node][0], np.inf))
                    continue
                base_score = ch_scores[[e for e in ch_scores if e.parent == child.event.s_node][0]]
            for edge, score in ch_scores.items():
                if isinstance(edge, Node) or edge is None: continue
                gain = base_score - score
                if gain > 0:
                    edges.append((edge, gain))
        edges.sort(key=lambda x: x[1], reverse=True)
        return edges

    def introduce_cycle(self, edge):
        g = nx.DiGraph()
        for e in self.graph.edges + [edge]:
            if isinstance(e.child, LogicalNode): continue
            child = self.graph.get_non_logical_nodes().index(e.child)
            if isinstance(e.parent, LogicalNode):
                pa1 = self.graph.get_non_logical_nodes().index(e.parent.parents[0])
                pa2 = self.graph.get_non_logical_nodes().index(e.parent.parents[1])
                g.add_edge(pa1, child)
                g.add_edge(pa2, child)
            else:
                g.add_edge(self.graph.get_non_logical_nodes().index(e.parent), child)
        try:
            cycles = nx.find_cycle(g, orientation='original')
            return True
        except nx.NetworkXNoCycle:
            return False

    def add_edge(self, edge):
        scorer = Scorer(self.n, self.data, self.empty_graph, self.graph)
        if isinstance(edge.parent, LogicalNode):
            ln = edge.parent
            edge.parent.parents[0].children.append(edge.child)
            edge.parent.parents[1].children.append(edge.child)
            # edge.child.parents += [edge.parent.parents[0], edge.parent.parents[1]]
            edge.child.parents.append(edge.parent)
            if ln.operator == "xor":
                e, _, other = scorer.xor_fit(ln.parents[0], ln.parents[1], edge.child)
            elif ln.operator == 'and':
                e, _, other = scorer.and_fit(ln.parents[0], ln.parents[1], edge.child)
            else:
                e, _, other = scorer.or_fit(ln.parents[0], ln.parents[1], edge.child)
            edge.child.distrib = other["distrib_noise"]
            self.graph.edges.append(Edge(parent=edge.parent.parents[0], child=ln, distrib=other['distrib_1']))
            self.graph.edges.append(Edge(parent=edge.parent.parents[1], child=ln, distrib=other['distrib_2']))
            self.graph.edges.append(edge)
            self.graph.nodes.append(ln)
            edge.parent.children.append(edge.child)
            edge.child.parents.append(edge.parent)
        else:
            self.graph.edges.append(edge)
            edge.parent.children.append(edge.child)
            edge.child.parents.append(edge.parent)

    def score_all_edges(self, all=True):
        scorer = Scorer(self.n, self.data, self.empty_graph, self.graph, verbose=self.verbose)
        if all:
            scores = scorer.score_all_edges() # without causal superset estimation
        else:
            scores = scorer.score_promising_edges() # with causal superset estimation
        for k, v in scores.items():
            for kk, vv in v.items():
                if isinstance(kk, Node) or kk is None: model_cost = 0
                else: model_cost = self.cost_of_model(kk.parent, kk.child)
                scores[k][kk] += model_cost
        self.scores = scores

    def cost_of_model(self, parent, child):
        cost = 0
        if not isinstance(parent, LogicalNode):
            if type(parent) in [StartNode, EndNode]:
                cost = np.log2(len(self.graph.events))
        else:
            conditioning_pa = [pa for pa in parent.parents if isinstance(pa, IntervalNode)]
            triggering_pa = [pa for pa in parent.parents if not isinstance(pa, IntervalNode)]
            if len(triggering_pa) > 0: cost += np.log2(len(self.graph.events))
            if len(conditioning_pa) > 0: cost += np.log2(math.comb(len(self.graph.events), len(conditioning_pa))) + len(conditioning_pa)
        return cost
