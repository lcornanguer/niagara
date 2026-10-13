"""
Class to score all possible edges
"""
import math
import os
from multiprocessing import Pool
from itertools import combinations
import numpy as np
from bisect import bisect_right

from .distributions import ExponentialDistribution
from .options import args
from .graph import Edge, Graph, LogicalNode, StartNode, EndNode, IntervalNode, Node
from .intervals import *


class Scorer:
    def __init__(self, n, data, empty_graph, graph=None, verbose=False):
        self.n = n
        self.data = data
        if graph is None:
            self.graph = Graph()
            self.graph.init_from_events([e.name for e in empty_graph.events])
        else:
            self.graph = graph
        self.empty_graph = empty_graph
        self.stored_intervals = dict()
        self.verbose = verbose
        self.sequences = []
        for occurrences in self.data.values():
            for occurrence in occurrences:
                if occurrence.ts_index not in self.sequences:
                    self.sequences.append(occurrence.ts_index)

        n = len(self.graph.events)
        comb = math.comb(n-1, 2)
        test_nb = n + n*(3*(n-1)) + n + n*(3*(n-1)) + n*14*comb + n*(n-1)
        if args.start_only: test_nb = n + n*(3*(n-1)) + n*14*comb + n*(n-1)
        self.exhaustive_test_nb = test_nb

    def score_all_edges(self):
        """
        Compute and return the data and parameter cost for all possible edges
        :return: A dictionary with the event names as key and the occurrences as value
        """
        scores = dict()
        n_test = 0
        for child in self.graph.get_non_logical_nodes():
            if isinstance(child, IntervalNode): continue
            scores[child] = dict()
            if isinstance(child, EndNode) and args.start_only:
                scores[child][Edge(child.event.s_node, child.event.e_node, None)] = -np.inf
                continue
            for parent in self.graph.get_non_logical_nodes():
                if args.start_only and isinstance(parent, EndNode): continue
                if parent == child: continue
                if parent.event == child.event and (not isinstance(parent, StartNode)): continue
                if isinstance(child, EndNode) and parent != child.event.s_node: continue
                e, cost, _ = self.edge_fit(parent, child)
                if e is not None: scores[child][e] = cost
                if self.verbose: print(f"{parent} -> {child}: {cost:.3f}")
                n_test += 1
            if isinstance(child, StartNode):
                cost = self.src_node_fit(child)
                scores[child][child] = cost
                if self.verbose: print(f"{child}: {cost:.3f}")
                n_test += 1
            for pa1, pa2 in combinations([n for n in self.graph.get_non_logical_nodes() if n != child and not (args.start_only and isinstance(n, EndNode))], 2):
                    if isinstance(child, EndNode):
                        if child.event.i_node not in [pa1, pa2]: continue
                        if child.event.s_node in [pa1, pa2]: continue
                        e, cost, _ = self.and_fit(pa1, pa2, child)
                        if e is not None: scores[child][e] = cost
                        if self.verbose: print(f"{pa1} AND {pa2} -> {child}: {cost:.3f}")
                        n_test += 1
                    else:
                        if isinstance(child, StartNode) and child.event in [pa1.event, pa2.event]: continue
                        if child.event.i_node in [pa1, pa2] or (pa1.event == pa2.event and IntervalNode in [type(pa1), type(pa2)]): continue
                        if not IntervalNode in [type(pa1), type(pa2)]: # SE + SE
                            if args.or_heuristic:
                                # We only test OR if both parents alone already improve the score
                                edges_alone = [e for e in scores[child].keys() if e is not None and not isinstance(e, Node) and e.parent in [pa1, pa2]]
                                if any([scores[child][child] < scores[child][e] for e in edges_alone]):
                                    if self.verbose: print(f"{pa1} OR {pa2} -> {child} SKIPPED")
                                    continue
                            e, cost, _ = self.or_fit(pa1, pa2, child)
                            if e is not None: scores[child][e] = cost
                            if self.verbose: print(f"{pa1} OR {pa2} -> {child}: {cost:.3f}")
                            n_test += 1
                        if not (type(pa1) in [StartNode, EndNode] and type(pa2) in [StartNode, EndNode]): # SE/I + I
                            if type(pa1) == IntervalNode and type(pa2) == IntervalNode: #I + I
                                e, cost, _ = self.or_fit(pa1, pa2, child)
                                if e is not None: scores[child][e] = cost
                                if self.verbose: print(f"{pa1} OR {pa2} -> {child}: {cost:.3f}")
                                n_test += 1
                            e, cost, _ = self.and_fit(pa1, pa2, child)
                            if e is not None: scores[child][e] = cost
                            if self.verbose: print(f"{pa1} AND {pa2} -> {child}: {cost:.3f}")
                            n_test += 1
                            e, cost, _ = self.xor_fit(pa1, pa2, child)
                            if e is not None: scores[child][e] = cost
                            if self.verbose: print(f"{pa1} XOR {pa2} -> {child}: {cost:.3f}")
                            n_test += 1
        print(f'Predicted test nb: {self.exhaustive_test_nb}, realized: {n_test}')
        return scores

    def score_promising_edges(self):
        scores = {n: {} for n in self.graph.get_non_logical_nodes()}
        promising_edges = self.causal_superset_estimation(width=args.window_length) #  5.3 Causal Superset Estimation in paper
        for event in self.graph.events:
            s_node = event.s_node
            e_node = event.e_node
            # source node
            cost = self.src_node_fit(s_node)
            scores[s_node][s_node] = cost
            if self.verbose: print(f"{s_node}: {cost:.3f}")
            # S -> E
            if not args.start_only:
                e, cost, _ = self.edge_fit_S_to_E(s_node, e_node)
                if e is not None: scores[e_node][e] = cost
                if self.verbose: print(f"{s_node} -> {e_node}: {cost:.3f}")
            else:
                scores[e_node][Edge(s_node, e_node, None)] = -np.inf
        for edge in promising_edges:
            child = edge.child
            if not isinstance(edge.parent, LogicalNode):
                e, cost, _ = self.edge_fit(edge.parent, child)
                if e is not None: scores[child][e] = cost
                if self.verbose: print(f"{edge.parent} -> {child}: {cost:.3f}")
            else:
                ln = edge.parent
                pa1, pa2 = ln.parents
                if ln.operator == 'and':
                    e, cost, _ = self.and_fit(pa1, pa2, child)
                    if e is not None: scores[child][e] = cost
                    if self.verbose: print(f"{pa1} AND {pa2} -> {child}: {cost:.3f}")
                elif ln.operator == 'or':
                    e, cost, _ = self.or_fit(pa1, pa2, child)
                    if e is not None: scores[child][e] = cost
                    if self.verbose: print(f"{pa1} OR {pa2} -> {child}: {cost:.3f}")
                elif ln.operator == 'xor':
                    e, cost, _ = self.xor_fit(pa1, pa2, child)
                    if e is not None: scores[child][e] = cost
                    if self.verbose: print(f"{pa1} XOR {pa2} -> {child}: {cost:.3f}")
        print(f'Predicted test nb: {self.exhaustive_test_nb}, realized: {sum([len(v.values()) for v in scores.values()])}')
        return scores

    def get_ongoing_range(self, occs):
        """
        Return ranges where the event is ongoing
        :param occs: occurences of the event
        :return: list of list (intervals)
        """
        intervals = [(o.start, o.end) if o.end is not None else (o.start, np.inf) for o in occs]  # Extract intervals
        merged = []

        for begin, end in sorted(intervals):  # Sort intervals by start time
            if merged and merged[-1][1] >= begin:  # Check for overlap (no need for `begin - 1`)
                merged[-1][1] = max(merged[-1][1], end)  # Merge intervals
            else:
                merged.append([begin, end])  # Append new interval

        return merged  # Return merged list

    def get_nb_ongoing(self, occs, n):
        events = []  # Store event start and end times
        for o in occs:
            events.append((o.start, 1))  # Start of an event (+1)
            events.append((o.end if o.end is not None else np.inf, -1))  # End of an event (-1)

        events.sort()  # Sort by time

        ongoing = 0
        result = IntervalList(n=n)
        prev_time = 0  # Start from 0

        for time, change in events:
            if time != prev_time:  # Only add when time actually changes
                result.create_and_add_interval(prev_time, time, ongoing)
            ongoing += change
            prev_time = time

        result.fill_gaps()  # Ensure complete coverage
        for elem in result.intervals:
            elem.count = min(elem.count, 1)
        result.intervals[-1].end = n
        return result


    def get_nb_ongoing_logical(self, occs1, occs2, n, mode="and"):
        """
        Compute the number of ongoing events in time intervals with AND or XOR condition

        Parameters:
            occs1 (list): List of event occurrences for type A.
            occs2 (list): List of event occurrences for type B.
            mode (str): "and" for both events occurring together, "xor" for only one of the events occurring.

        Returns:
            List(s) of tuples (start_time, end_time, count).
        """
        int1 = self.get_nb_ongoing(occs1, n)
        int2 = self.get_nb_ongoing(occs2, n)
        if mode == 'and':
            result = int1.and_op(int2)
        elif mode == 'xor':
            result = int1.xor_op(int2)
        elif mode == 'or':
            result = int1.or_op(int2)
        else:
            print("Invalid mode (and/or/xor).")
            return
        if isinstance(result, IntervalList):
            for elem in result.intervals: elem.count = min(elem.count, 1)
        return result

    def manage_stored_intervals(self, node, sequence):
        key = [k for k in self.stored_intervals.keys() if k[1] == sequence and k[0].is_eq(node)]
        if len(key) > 0:
            return self.stored_intervals[key[0]]
        else:
            if isinstance(node, LogicalNode):
                occs1 = [occ for occ in self.data[node.parents[0].event.name] if occ.ts_index == sequence]
                occs2 = [occ for occ in self.data[node.parents[1].event.name] if occ.ts_index == sequence]
                nb_ongoing = self.get_nb_ongoing_logical(occs1, occs2, self.n[sequence], node.operator)
                if node.operator in ['xor']:
                    nb_ongoing = {node.parents[0].name: nb_ongoing[0], node.parents[1].name: nb_ongoing[1], 'both': nb_ongoing[2]}
            else:
                occurrences = [occ for occ in self.data[node.event.name] if occ.ts_index == sequence]
                nb_ongoing = self.get_nb_ongoing(occurrences, self.n[sequence])
            self.stored_intervals[(node, sequence)] = nb_ongoing
            return self.stored_intervals[(node, sequence)]

    def get_times_by_sequence(self, event, moment, intervals=None, positive=True, offset=0):
        times = {sequence: [] for sequence in self.sequences}
        for occurrence in self.data[event]:
            time = getattr(occurrence, moment)
            if time is None: continue
            if intervals is None or intervals[occurrence.ts_index].positive_count(time-offset) == positive:
                times[occurrence.ts_index].append(time)
        for values in times.values(): values.sort()
        return times

    def and_fit(self, pa1, pa2, child):
        """
        Fit edge pa1 AND pa2 -> child and compute score
        :param pa1:
        :param pa2:
        :param child:
        :return: (Edge, cost)
        """
        ln = LogicalNode([pa1, pa2], 'and')
        mom = 'start' if isinstance(child, StartNode) else 'end'
        if isinstance(pa1, IntervalNode) and isinstance(pa2, IntervalNode):
            nb_ongoing = {sequence: self.manage_stored_intervals(ln, sequence) for sequence in self.sequences}
            var = 10**-10 if isinstance(child, EndNode) else 0 # because at end time, the event isn't ongoing anymore...
            # cost of noise
            t = self.get_times_by_sequence(child.event.name, mom, nb_ongoing, positive=False, offset=var)
            if sum([len(values) for values in t.values()])/len(self.data[child.event.name]) > args.err_max: return None, np.inf, None
            distrib_noise, noise_cost = self.additive_noise_cost(t) #self.additive_noise_cost(t)
            # cost of causal link
            t = self.get_times_by_sequence(child.event.name, mom, nb_ongoing, offset=var)
            distrib = ExponentialDistribution()
            delay_cost = self.fit_and_cost_PP(t, nb_ongoing, distrib)
            e = Edge(ln, child, None)
            other = {"distrib_1": distrib, "distrib_2": distrib, "distrib_noise": distrib_noise}
            return e, delay_cost + noise_cost + distrib_noise.get_parameter_cost() + distrib.get_parameter_cost(), other
        else: # [I, S/E]
            i_parent = pa1 if isinstance(pa1, IntervalNode) else pa2
            o_parent = pa1 if not isinstance(pa1, IntervalNode) else pa2
            intervals = {sequence: self.get_ongoing_range([occ for occ in self.data[i_parent.event.name] if occ.ts_index == sequence]) for sequence in self.sequences}
            _, _, other = self.match_occ_optimal(o_parent, child, intervals)
            if other is None: return None, np.inf, None
            distrib_noise = other["distrib_noise"]
            distrib = other["distrib"]
            delay_cost = other["delay_cost"]
            noise_cost = other["noise_cost"]
            skipped_cost = other["skipped_cost"]
            e = Edge(ln, child, None)
            other = {"distrib_1": distrib, "distrib_2": distrib, "distrib_noise": distrib_noise}
            return e, delay_cost + noise_cost + distrib_noise.get_parameter_cost() + distrib.get_parameter_cost() + skipped_cost, other


    def or_fit(self, pa1, pa2, child):
        if IntervalNode in [type(pa1), type(pa2)]: #I OR I
            e, cost, other = self.I_OR_I(pa1, pa2, child)
            return e, cost, other
        else:  # SE OR SE
            if isinstance(child, EndNode) and child.event.s_node in [pa1, pa2]: return None, np.inf, None
            ln = LogicalNode([pa1, pa2], 'or')
            cost, other = self.match_occ_optimal_dual(pa1, pa2, child)
            if other is None: return None, np.inf, None
            e = Edge(ln, child, None)
            return e, cost, other

    def xor_fit(self, pa1, pa2, child):
        """
        Fit edge pa1 XOR pa2 -> child and compute score
        :param pa1:
        :param pa2:
        :param child:
        :return: (Edge, cost)
        """
        ln = LogicalNode([pa1, pa2], 'xor')
        mom = 'start' if isinstance(child, StartNode) else 'end'
        distrib = [None, None]
        t = [None, None]
        delay_cost = 0
        if isinstance(pa1, IntervalNode) and isinstance(pa2, IntervalNode):
            nb_ongoing = {sequence: self.manage_stored_intervals(ln, sequence) for sequence in self.sequences}
            # cost of noise
            nb_notongoing = {sequence: nb_ongoing[sequence]['both'].get_inverse() for sequence in self.sequences}
            t_noise = self.get_times_by_sequence(child.event.name, mom, nb_notongoing)
            if sum([len(values) for values in t_noise.values()]) / len(self.data[child.event.name]) > args.err_max: return None, np.inf, None
            distrib_noise, noise_cost = self.additive_noise_cost(t_noise)
            # cost of causal link
            for i, pa in enumerate([pa1, pa2]):
                pa_ongoing = {sequence: nb_ongoing[sequence][pa.name] for sequence in self.sequences}
                t[i] = self.get_times_by_sequence(child.event.name, mom, pa_ongoing)
                if sum([len(values) for values in t[i].values()]) == 0: return None, np.inf, None
                distrib[i] = ExponentialDistribution()
                delay_cost += self.fit_and_cost_PP(t[i], pa_ongoing, distrib[i])
            e = Edge(ln, child, None)
            other = {"distrib_1": distrib[0], "distrib_2": distrib[1], "distrib_noise": distrib_noise}
            return e, delay_cost + noise_cost + distrib_noise.get_parameter_cost() + distrib[0].get_parameter_cost() + distrib[1].get_parameter_cost(), other
        else: # [I, S/E]
            i_parent = pa1 if isinstance(pa1, IntervalNode) else pa2
            o_parent = pa1 if not isinstance(pa1, IntervalNode) else pa2
            nb_ongoing = {sequence: self.manage_stored_intervals(IntervalNode(i_parent.event), sequence) for sequence in self.sequences}
            intervals_off = {sequence: nb_ongoing[sequence].get_inverse().convert_to_interval_list() for sequence in self.sequences}
            # Matches S/E with child
            _, _, other = self.match_occ_optimal(o_parent, child, intervals_off) #self.edge_fit_from_S_E_iter(o_parent, child, intervals_off)
            if other is None: return None, np.inf, None
            distrib_noise = other["distrib_noise"]
            se_distrib = other["distrib"]
            delay_cost = other["delay_cost"]
            noise_cost = other["noise_cost"]
            skipped_cost = other["skipped_cost"]
            i_distrib =  None
            e = Edge(ln, child, None)
            distrib = [i_distrib, se_distrib if isinstance(ln.parents[0], IntervalNode) else se_distrib, i_distrib]
            other = {"distrib_1": distrib[0], "distrib_2": distrib[1], "distrib_noise": distrib_noise}
            return e, delay_cost + noise_cost + distrib_noise.get_parameter_cost() + se_distrib.get_parameter_cost() + skipped_cost, other

    def I_OR_I(self, pa1, pa2, child): # 5.2 Occurrence Partitioning in paper
        """
        """
        assert isinstance(pa1, IntervalNode) and isinstance(pa2, IntervalNode)
        partition = {pa1: {sequence: [] for sequence in self.sequences},
                     pa2: {sequence: [] for sequence in self.sequences},
                     'noise': {sequence: [] for sequence in self.sequences}}
        mom = 'start' if isinstance(child, StartNode) else 'end'
        ln = LogicalNode([pa1, pa2], 'or')
        nb_ongoing = {pa1.name: {sequence: self.manage_stored_intervals(IntervalNode(pa1.event), sequence) for sequence in self.sequences},
                      pa2.name: {sequence: self.manage_stored_intervals(IntervalNode(pa2.event), sequence) for sequence in self.sequences},
                      'both': {sequence: self.manage_stored_intervals(ln, sequence) for sequence in self.sequences}}
        # rate of noise
        nb_notongoing = {sequence: nb_ongoing['both'][sequence].get_inverse() for sequence in self.sequences}
        t = self.get_times_by_sequence(child.event.name, mom, nb_notongoing)
        if sum([len(values) for values in t.values()]) / len(self.data[child.event.name]) > args.err_max: return None, np.inf, None
        partition['noise'] = t
        rate_noise = sum([len(values) for values in t.values()]) / sum([intervals.duration() for intervals in nb_notongoing.values()])

        nb_ongoing_pure = {sequence: self.manage_stored_intervals(LogicalNode([pa1, pa2], 'xor'), sequence) for sequence in self.sequences} # to get pure intervals for rate fitting
        # rate of pa1
        pa_ongoing = {sequence: nb_ongoing_pure[sequence][pa1.name] for sequence in self.sequences}
        t = self.get_times_by_sequence(child.event.name, mom, pa_ongoing) # TODO: actually some of them can be noise
        partition[pa1] = t
        d = sum([intervals.duration() for intervals in pa_ongoing.values()])
        if d > 0:
            rate_pa1 = max(0, sum([len(values) for values in t.values()]) / d - rate_noise)
        else:
            return None, np.inf, None
        # rate of pa2
        pa_ongoing = {sequence: nb_ongoing_pure[sequence][pa2.name] for sequence in self.sequences}
        t = self.get_times_by_sequence(child.event.name, mom, pa_ongoing) # TODO: actually some of them can be noise
        partition[pa2] = t
        d = sum([intervals.duration() for intervals in pa_ongoing.values()])
        if d > 0:
            rate_pa2 = max(0, sum([len(values) for values in t.values()]) / d - rate_noise)
        else:
            return None, np.inf, None
        if rate_pa2 < rate_noise or rate_pa1 < rate_noise or rate_pa1 == 0 or rate_pa2 == 0: return None, np.inf, None

        nb_ongoing_mixed = {sequence: self.manage_stored_intervals(LogicalNode([pa1, pa2], 'and'), sequence) for sequence in self.sequences}  # to get intervals where both are active
        mixed = self.get_times_by_sequence(child.event.name, mom, nb_ongoing_mixed)
        for sequence, t in mixed.items():
            if len(t) == 0: continue
            n_pa1 = round(rate_pa1/(rate_pa1+rate_pa2) * len(t))
            n_pa2 = round(rate_pa2/(rate_pa1+rate_pa2) * len(t))

            min_pa = pa1 if n_pa1 < n_pa2 else pa2
            max_pa = pa2 if n_pa1 < n_pa2 else pa1
            min_n = n_pa1 if n_pa1 < n_pa2 else n_pa2
            if min_n == 0:
                labels = [max_pa] * len(t)
            else:
                avg_space = (len(t) - min_n) / min_n
                # Assign label
                labels = []
                rest = 0
                while len(labels) < len(t):
                    if rest >= 1:
                        space = math.ceil(avg_space)
                        rest += avg_space - math.ceil(avg_space)
                    else:
                        space = math.floor(avg_space)
                        rest += avg_space - math.floor(avg_space)
                    labels += [max_pa] * space
                    labels += [min_pa]
                labels = labels[:len(t)]
            for i, l in enumerate(labels):
                partition[l][sequence].append(t[i])
            partition[pa1][sequence].sort()
            partition[pa2][sequence].sort()
        # Compute cost
        distrib_noise, noise_cost = self.additive_noise_cost(partition['noise'])
        # cost of causal link
        distribs = dict()
        delay_costs = dict()
        for pa in [pa1, pa2]:
            distribs[pa] = ExponentialDistribution()
            delay_costs[pa] = self.fit_and_cost_PP(partition[pa], nb_ongoing[pa.name], distribs[pa])
        e = Edge(ln, child, None)
        other = {"distrib_1": distribs[pa1], "distrib_2": distribs[pa2], "distrib_noise": distrib_noise}
        return e, sum(delay_costs.values()) + noise_cost + distrib_noise.get_parameter_cost() + distribs[pa1].get_parameter_cost() + \
                  distribs[pa2].get_parameter_cost(), other

        return e, delay_cost + noise_cost + distrib_noise.get_parameter_cost() + e.distrib.get_parameter_cost(), other

    def edge_fit_from_I(self, parent, child):
        """
        Fit edge parent_I -> child and compute score
        :param parent:
        :param child:
        :return: (Edge, cost)
        """
        assert isinstance(parent, IntervalNode) and not isinstance(child, IntervalNode)
        mom = 'start' if isinstance(child, StartNode) else 'end'
        nb_ongoing = {sequence: self.manage_stored_intervals(IntervalNode(parent.event), sequence) for sequence in self.sequences}
        # cost of noise
        t = self.get_times_by_sequence(child.event.name, mom, nb_ongoing, positive=False)
        if sum([len(values) for values in t.values()]) / len(self.data[child.event.name]) > args.err_max: return None, np.inf, None
        distrib_noise, noise_cost = self.additive_noise_cost(t)
        # cost of causal link
        t = self.get_times_by_sequence(child.event.name, mom, nb_ongoing)
        distrib = ExponentialDistribution()
        delay_cost = self.fit_and_cost_PP(t, nb_ongoing, distrib)
        e = Edge(parent, child, distrib)
        other = {"distrib_noise": distrib_noise}
        return e, delay_cost + noise_cost + distrib_noise.get_parameter_cost() + e.distrib.get_parameter_cost(), other

    def fit_and_cost_PP(self, t, nb_ongoing, distrib, fit=True):
        """Compute the delays with time occurrences and constraining intervals, fit the delay distribution,
        and compute the cost"""
        if isinstance(t, dict):
            if sum([len(values) for values in t.values()]) == 0: return 0
            if fit:
                duration = sum([nb_ongoing[sequence].duration() for sequence in self.sequences])
                distrib.intensity = sum([len(values) for values in t.values()]) / duration
            cost = 0
            for sequence in self.sequences:
                if len(t[sequence]) == 0:
                    if nb_ongoing[sequence].duration() > 0:
                        cost += sum(distrib.cost_per_delay([nb_ongoing[sequence].duration()]))
                else:
                    cost += self.fit_and_cost_PP(t[sequence], nb_ongoing[sequence], distrib, fit=False)
            return cost
        if len(t) == 0: return 0
        delays = list()
        for i, occ in enumerate(t):
            if i == 0:
                delays.append(t[i] - nb_ongoing.get_last_inactive(t[i]).end)
            else:
                delays.append(nb_ongoing.duration_between(t[i - 1], t[i]))
        max_t = nb_ongoing.get_last_active(nb_ongoing.n).end
        delays.append(max_t-t[-1]) # Phantom occ to penalize nothing occurring at the end
        if fit:
            distrib.intensity = len(t) / nb_ongoing.duration()
        cost = sum(distrib.cost_per_delay(delays))
        return cost

    def edge_fit_S_to_E(self, parent, child):
        """
        Fit edge event_S -> event_E and compute score
        :param parent:
        :param child:
        :return: (Edge, cost)
        """
        assert parent.event == child.event and isinstance(parent, StartNode) and isinstance(child, EndNode)
        delays = [o.end - o.start for o in self.data[parent.event.name] if o.end is not None]
        if len(delays) == 0: return None, np.inf, None
        if args.duration_distrib == "LogNormal":
            distrib = LogNormalDistribution()
        else:
            distrib = WeibullDistribution()
        distrib.fit_to_delays(delays)
        delay_cost = sum(distrib.cost_per_delay(delays))
        e = Edge(parent, child, distrib)
        other = None
        return e, delay_cost + e.distrib.get_parameter_cost(), other

    def match_occ_optimal(self, parent, child, intervals=None): # 5.2 Occurrence Partitioning in paper
        matches = []
        delays = []

        mom_pa = 'start' if isinstance(parent, StartNode) else 'end'
        mom_ch = 'start' if isinstance(child, StartNode) else 'end'
        valid_parent_found = False
        valid_child_found = False

        for sequence in self.sequences:
            pa_occs = [occ for occ in self.data[parent.event.name] if occ.ts_index == sequence]
            ch_occs = [occ for occ in self.data[child.event.name] if occ.ts_index == sequence]

            # Precompute timestamps for performance
            pa_timestamps_raw = [(pa, getattr(pa, mom_pa)) for pa in pa_occs if getattr(pa, mom_pa) is not None]
            ch_timestamps = [(ch, getattr(ch, mom_ch)) for ch in ch_occs if getattr(ch, mom_ch) is not None]

            pa_timestamps_raw.sort(key=lambda x: x[1]) # necessary if parent is EndNode

            # If intervals are used, optimize interval lookup
            if intervals is not None and intervals[sequence]:
                # Sort intervals and convert to tuple of bounds for binary search
                interval_bounds = sorted(intervals[sequence])
                starts, ends = zip(*interval_bounds)

                def is_in_intervals(ts):
                    # Binary search for interval containment
                    i = bisect_right(starts, ts) - 1
                    return 0 <= i < len(interval_bounds) and ts < ends[i]

                valid_pa_occs = [pa for pa, ts in pa_timestamps_raw if ts is not None and is_in_intervals(ts)]
            elif intervals is not None:
                valid_pa_occs = []
            else:
                valid_pa_occs = [pa for pa, ts in pa_timestamps_raw if ts is not None]

            valid_ch_occs = [ch for ch, _ in ch_timestamps]
            valid_parent_found = valid_parent_found or bool(valid_pa_occs)
            valid_child_found = valid_child_found or bool(valid_ch_occs)

            # Extract timestamps again after filtering
            pa_timestamps = [getattr(occ, mom_pa) for occ in valid_pa_occs]
            ch_timestamps = [getattr(occ, mom_ch) for occ in valid_ch_occs]

            matched_pa = set()
            matched_ch = set()

            nb_pa_matched = 0
            for ch_i, ch_occ in enumerate(valid_ch_occs):
                ch_t = ch_timestamps[ch_i]
                idx = bisect_right(pa_timestamps, ch_t)
                if idx == 0 or pa_timestamps[idx-1] == np.inf: # child time is smaller than all parent times
                    matches.append((None, ch_occ, np.inf, None))
                else:
                    pa_t = pa_timestamps[idx-1]
                    pa_j = idx - 1 + nb_pa_matched
                    matched_pa.add(valid_pa_occs[pa_j])
                    matched_ch.add(ch_occ)
                    delay = ch_timestamps[ch_i] - pa_t
                    delays.append(delay)
                    matches.append((valid_pa_occs[pa_j], ch_occ, delay))
                    pa_timestamps = np.delete(pa_timestamps, idx-1, 0)
                    nb_pa_matched += 1 # to keep the relation between pa_timestamps and valid_pa_occs despite the deletions

            # Unmatched parent occurrences
            for pa in valid_pa_occs:
                if pa not in matched_pa:
                    matches.append((pa, None, np.inf))
                    delays.append(np.inf)

        if not valid_parent_found or not valid_child_found:
            return None, np.inf, None

        def fit_model(delays, matches):
            distrib = ExponentialDistribution()
            # noise cost for child (child triggered without parents)
            unmatched_ch_occs = [m[1] for m in matches if m[0] is None]
            mom = 'start' if isinstance(child, StartNode) else 'end'
            t = {sequence: [getattr(occ, mom) for occ in unmatched_ch_occs if occ.ts_index == sequence and getattr(occ, mom) is not None] for sequence in self.sequences}
            if sum([len(values) for values in t.values()]) / len(self.data[child.event.name]) > args.err_max: return np.inf, None
            for values in t.values(): values.sort()
            distrib_noise, noise_cost = self.additive_noise_cost(t)
            # cost of causal link
            distrib.fit_to_delays(delays)
            delay_cost = sum(distrib.cost_per_delay(delays))
            skipped_cost = distrib.skipped_cost(delays)
            full_cost = delay_cost + noise_cost + distrib_noise.get_parameter_cost() + distrib.get_parameter_cost() + skipped_cost
            other = {"distrib_noise": distrib_noise, "distrib": distrib, "delay_cost": delay_cost, "noise_cost": noise_cost, "skipped_cost": skipped_cost}
            return full_cost, other

        full_cost, other = fit_model(delays, matches)

        improvement = True

        while improvement and any([d != np.inf for d in delays]):
            max_m = max([m for m in matches if m[2] is not np.inf], key=lambda m: m[2])
            max_delay = max_m[2]
            max_pa = max_m[0]
            max_ch = max_m[1]

            # remove max delay
            new_delays = delays.copy()
            new_delays.remove(max_delay)
            new_delays.append(np.inf)
            new_matches = matches.copy()
            new_matches.remove(max_m)
            new_matches.append((max_pa, None, np.inf))
            new_matches.append((None, max_ch, np.inf))

            new_cost, new_other = fit_model(new_delays, new_matches)

            if new_cost < full_cost:
                full_cost = new_cost
                delays = new_delays
                matches = new_matches
                other = new_other
            else:
                improvement = False

        if other is not None:
            if other['distrib'].alpha < args.err_max: return None, np.inf, None
            e = Edge(parent, child, other['distrib'])
        else:
            e = None

        return e, full_cost, other

    def additive_noise_cost(self, t_occ):
        distrib_noise = ExponentialDistribution()
        if isinstance(t_occ, dict):
            t_occ = {sequence: [t for t in t_occ[sequence] if t != np.inf] for sequence in self.sequences}
            intervals = {sequence: IntervalList([Interval(0, self.n[sequence], 1)], self.n[sequence]) for sequence in self.sequences}
        else:
            t_occ = [t for t in t_occ if t != np.inf]
            intervals = IntervalList([Interval(0, self.n, 1)], self.n)
        noise_cost = self.fit_and_cost_PP(t_occ, intervals, distrib_noise)
        return distrib_noise, noise_cost

    def edge_fit(self, parent, child):
        """
        Fit edge parent -> child and compute score
        :param parent:
        :param child:
        :return: (Edge,cost)
        """
        if isinstance(parent, IntervalNode):
            e, cost, other = self.edge_fit_from_I(parent, child)
        elif parent.event == child.event and isinstance(parent, StartNode) and isinstance(child, EndNode):
            e, cost, other = self.edge_fit_S_to_E(parent, child)
        else:
            e, cost, other = self.match_occ_optimal(parent, child) #self.edge_fit_from_S_E_iter(parent, child)# self.edge_fit_from_S_E(parent, child)
        return e, cost, other

    def src_node_fit(self, node):
        """
        Fit source nodes and compute score
        :param node:
        :return: cost
        """
        assert isinstance(node, StartNode)
        t = {sequence: [occ.start for occ in self.data[node.event.name] if occ.ts_index == sequence] for sequence in self.sequences}
        intervals = {sequence: IntervalList([Interval(0, self.n[sequence], 1)], self.n[sequence]) for sequence in self.sequences}
        distrib = ExponentialDistribution()
        delay_cost = self.fit_and_cost_PP(t, intervals, distrib)
        return delay_cost + distrib.get_parameter_cost()

    def match_occ_optimal_dual(self, pa1, pa2, child): # 5.2 Occurrence Partitioning in paper
        matches = []
        delays = {pa1: [], pa2: []}

        # Extract timestamps
        mom_pa1 = 'start' if isinstance(pa1, StartNode) else 'end'
        mom_pa2 = 'start' if isinstance(pa2, StartNode) else 'end'
        mom_ch = 'start' if isinstance(child, StartNode) else 'end'
        valid_parent_found = False
        valid_child_found = False

        for sequence in self.sequences:
            # Get occurrences from data
            pa1_occs = [occ for occ in self.data[pa1.event.name] if occ.ts_index == sequence]
            pa2_occs = [occ for occ in self.data[pa2.event.name] if occ.ts_index == sequence]
            ch_occs = [occ for occ in self.data[child.event.name] if occ.ts_index == sequence]

            # Filter by valid intervals
            valid_pa_obj = [(getattr(pa, mom_pa1), pa, pa1) for pa in pa1_occs if getattr(pa, mom_pa1) is not None] # t, occ, node
            valid_pa_obj += [(getattr(pa, mom_pa2), pa, pa2) for pa in pa2_occs if getattr(pa, mom_pa2) is not None] # t, occ, node
            valid_pa_obj.sort(key=lambda x: x[0])

            valid_ch_occs = [ch for ch in ch_occs if getattr(ch, mom_ch) is not None]
            valid_parent_found = valid_parent_found or bool(valid_pa_obj)
            valid_child_found = valid_child_found or bool(valid_ch_occs)

            ch_timestamps = [getattr(occ, mom_ch) for occ in valid_ch_occs]

            matched_pa1 = set()
            matched_pa2 = set()
            matched_ch = set()

            for ch_i, ch_occ in enumerate(valid_ch_occs):
                ch_t = ch_timestamps[ch_i]
                idx = bisect_right([o[0] for o in valid_pa_obj], ch_t)
                if idx == 0 or valid_pa_obj[idx-1] == np.inf:  # child time is smaller than all parent times
                    matches.append((None, ch_occ, np.inf, None))
                else:
                    pa_obj = valid_pa_obj[idx-1]
                    pa_t, pa_occ, pa_node = pa_obj
                    if pa_node == pa1:
                        matched_pa = matched_pa1
                    else:
                        matched_pa = matched_pa2
                    matched_pa.add(pa_occ)
                    matched_ch.add(ch_occ)
                    delay = ch_t - pa_t
                    delays[pa_node].append(delay)
                    matches.append((pa_occ, ch_occ, delay, pa_node))
                    valid_pa_obj.remove(pa_obj) # parent occurrence is taken

            # Unmatched parent occurrences
            for pa_obj in valid_pa_obj:
                matches.append((pa_obj[1], None, np.inf, pa_obj[2]))
                delays[pa_obj[2]].append(np.inf)

        if not valid_parent_found or not valid_child_found:
            return None, None # If no valid occurrences, return

        def fit_model(delays, matches):
            distrib = {pa: ExponentialDistribution() for pa in [pa1, pa2]}
            delay_cost, skipped_cost = dict(), dict()
            # noise cost for child (child triggered without parents)
            unmatched_ch_occs = [m[1] for m in matches if m[0] is None]
            mom = 'start' if isinstance(child, StartNode) else 'end'
            t = {sequence: [getattr(occ, mom) for occ in unmatched_ch_occs if occ.ts_index == sequence and getattr(occ, mom) is not None] for sequence in self.sequences}
            if sum([len(values) for values in t.values()]) / len(self.data[child.event.name]) > args.err_max: return np.inf, None
            for values in t.values(): values.sort()
            distrib_noise, noise_cost = self.additive_noise_cost(t)
            # cost of causal link
            for pa in [pa1, pa2]:
                distrib[pa].fit_to_delays(delays[pa])
                delay_cost[pa] = sum(distrib[pa].cost_per_delay(delays[pa]))
                skipped_cost[pa] = distrib[pa].skipped_cost(delays[pa])
            full_cost = sum(delay_cost.values()) + noise_cost + distrib_noise.get_parameter_cost() + distrib[
                pa1].get_parameter_cost() + distrib[pa2].get_parameter_cost() + sum(skipped_cost.values())
            other = {"distrib_1": distrib[pa1], "distrib_2": distrib[pa2], "distrib_noise": distrib_noise}
            return full_cost, other

        full_cost, other = fit_model(delays, matches)
        if full_cost == np.inf: return full_cost, other

        improvement = True

        while improvement:
            max_m = max([m for m in matches if m[2] is not np.inf], key=lambda m: m[2])
            max_delay = max_m[2]
            max_pa_node = max_m[3]
            max_pa = max_m[0]
            max_ch = max_m[1]

            # remove max delay
            new_delays = {k: v.copy() for k, v in delays.items()}
            new_delays[max_pa_node].remove(max_delay)
            new_delays[max_pa_node].append(np.inf)
            new_matches = matches.copy()
            new_matches.remove(max_m)
            new_matches.append((max_pa, None, np.inf, max_pa_node))
            new_matches.append((None, max_ch, np.inf, None))

            new_cost, new_other = fit_model(new_delays, new_matches)

            if new_cost < full_cost:
                full_cost = new_cost
                delays = new_delays
                matches = new_matches
                other = new_other
            else:
                improvement = False

        if other is not None and (other['distrib_1'].alpha < args.err_max or other['distrib_2'].alpha < args.err_max): return np.inf, None
        return full_cost, other

    def estimate_parents_intervals(self, child, backward_only=False, forward_only=False, width=30):
        intervals = {sequence: IntervalList(n=self.n[sequence]) for sequence in self.sequences}
        mom = 'start' if isinstance(child, StartNode) else 'end'
        for occ in self.data[child.event.name]:
            if getattr(occ, mom) is None: continue
            if backward_only:
                start, end = getattr(occ, mom) - width, getattr(occ, mom)
            elif forward_only:
                start, end = getattr(occ, mom), getattr(occ, mom) + width
            else:
                start, end = getattr(occ, mom) - width / 2, getattr(occ, mom) + width / 2
            intervals[occ.ts_index].add_interval(Interval(start, end), sum_count=True)
        return intervals


    def causal_superset_estimation(self, width=30):
        self.graph.update_node_lists()
        # initialization of the stats
        stats = np.zeros(shape=(len(self.graph.non_logical_nodes), len(self.graph.non_logical_nodes)))
        stats.fill(np.nan) # IN PAPER: p1
        ipos = np.zeros(shape=(len(self.graph.non_logical_nodes), len(self.graph.non_logical_nodes)))
        ipos.fill(np.nan) # IN PAPER: p4
        ineg = np.zeros(shape=(len(self.graph.non_logical_nodes), len(self.graph.non_logical_nodes)))
        ineg.fill(np.nan) # IN PAPER: (1-p3)
        # data preparation
        data = dict.fromkeys(self.graph.non_logical_nodes)
        for node in self.graph.non_logical_nodes:
            if isinstance(node, StartNode):
                data[node] = {sequence: [occ.start for occ in self.data[node.event.name] if occ.ts_index == sequence] for sequence in self.sequences}
            elif isinstance(node, EndNode):
                data[node] = {sequence: [occ.end for occ in self.data[node.event.name] if occ.ts_index == sequence and occ.end is not None] for sequence in self.sequences}
            else:
                data[node] = {sequence: self.get_ongoing_range([occ for occ in self.data[node.event.name] if occ.ts_index == sequence]) for sequence in self.sequences}
            for d in data[node].values(): d.sort()

        def data_len(node):
            return sum([len(values) for values in data[node].values()])
        # stat computation
        for ch_i, child in enumerate(self.graph.non_logical_nodes):
            if isinstance(child, IntervalNode): continue
            if args.start_only and isinstance(child, EndNode): continue
            intervals = self.estimate_parents_intervals(child, backward_only=True, width=width)
            for pa_i, parent in enumerate(self.graph.non_logical_nodes):
                if parent == child or parent.event == child.event: continue
                if not isinstance(parent, IntervalNode):
                    count = 0
                    for sequence in self.sequences:
                        pa_occs = data[parent][sequence]
                        for i in intervals[sequence].intervals:
                            idx1 = bisect_right(pa_occs, i.start)
                            idx2 = bisect_right(pa_occs, i.end)
                            count += min(i.count, idx2 - idx1)
                    stats[ch_i, pa_i] = count / data_len(child)
                else: # IntervalNode
                    if child.event == parent.event: continue
                    cpos = 0
                    cneg = 0
                    for sequence in self.sequences:
                        pa_intervals = self.manage_stored_intervals(parent, sequence)
                        for t in data[child][sequence]:
                            if pa_intervals.positive_count(t) or pa_intervals.duration_between(t-width, t) > 0:
                                cpos += 1
                            if not pa_intervals.positive_count(t):
                                cneg += 1
                    ipos[ch_i, pa_i] = cpos / data_len(child)
                    ineg[ch_i, pa_i] = cneg / data_len(child)
        # interesting causal mechs selection
        promising = list()
        for ch_i, child in enumerate(self.graph.get_non_logical_nodes()):
            if isinstance(child, IntervalNode): continue
            high_p = stats[ch_i, :] > (1-args.err_max) * 0.9
            high_p_idx = np.argwhere(high_p)
            high_p_i = ipos[ch_i, :] > (1-args.err_max) * 0.9
            high_p_i_idx = np.argwhere(high_p_i)
            low_p_i = ineg[ch_i, :] > args.err_max
            low_p_i_idx = np.argwhere(low_p_i)
            for id in high_p_idx:
                parent = self.graph.non_logical_nodes[id[0]]
                if isinstance(child, EndNode):
                    ln = LogicalNode([parent, child.event.i_node], 'and')
                    promising.append(Edge(ln, child, None))
                    continue
                if data_len(child) / data_len(parent) > (1-args.err_max) * 0.9: # alpha
                    promising.append(Edge(parent, child, None)) # S/E -> child
                # else:
                for id2 in low_p_i_idx:
                    parent2 = self.graph.non_logical_nodes[id2[0]]
                    ln = LogicalNode([parent, parent2], 'xor')
                    promising.append(Edge(ln, child, None)) # S/E XOR I -> child
                for id2 in high_p_i_idx:
                    parent2 = self.graph.non_logical_nodes[id2[0]]
                    ln = LogicalNode([parent, parent2], 'and')
                    promising.append(Edge(ln, child, None)) # S/E AND I -> child
            for index, id1 in enumerate(high_p_i_idx):
                parent = self.graph.non_logical_nodes[id1[0]]
                if isinstance(child, EndNode):
                    parent2 = child.event.i_node
                    ln = LogicalNode([parent, parent2], 'and')
                    promising.append(Edge(ln, child, None)) # I -> child
                else:
                    promising.append(Edge(parent, child, None)) # I -> child
                if not isinstance(child, EndNode):
                    for id2 in high_p_i_idx[index:]:
                        if id2 == id1: continue
                        parent2 = self.graph.non_logical_nodes[id2[0]]
                        ln = LogicalNode([parent, parent2], 'and')
                        promising.append(Edge(ln, child, None))  # I AND I -> child
                        ln = LogicalNode([parent, parent2], 'xor')
                        promising.append(Edge(ln, child, None))  # I XOR I -> child
            if not isinstance(child, EndNode):
                # I XOR I -> child
                row_p = ipos[ch_i, :]
                i, j = np.triu_indices(len(row_p), k=1)  # Avoid (i, i) and duplicate pairs
                sums = row_p[i] + row_p[j]
                mask = sums > (1-args.err_max) * 0.9 # Get pairs where the sum is above threshold
                result_indices = list(zip(i[mask], j[mask]))
                for r in result_indices:
                    parent1 = self.graph.non_logical_nodes[r[0]]
                    parent2 = self.graph.non_logical_nodes[r[1]]
                    if data_len(child) / data_len(parent1) > (1-args.err_max) * 0.9 and data_len(child) / data_len(parent2) > (1-args.err_max) * 0.9:
                        ln = LogicalNode([parent1, parent2], 'xor')
                        promising.append(Edge(ln, child, None))
                # S/E OR S/E -> child
                row_p = stats[ch_i, :]
                i, j = np.triu_indices(len(row_p), k=1)  # Avoid (i, i) and duplicate pairs
                sums = row_p[i] + row_p[j]
                mask = sums > (1-args.err_max) * 0.9 # Get pairs where the sum is above threshold
                result_indices = list(zip(i[mask], j[mask]))
                for r in result_indices:
                    parent1 = self.graph.non_logical_nodes[r[0]]
                    parent2 = self.graph.non_logical_nodes[r[1]]
                    if (data_len(child) * stats[ch_i, r[0]]) / data_len(parent1) > (1-args.err_max) * 0.9 and (data_len(child) * stats[ch_i, r[1]]) / data_len(parent2) > (1-args.err_max) * 0.9:
                        ln = LogicalNode([parent1, parent2], 'or')
                        promising.append(Edge(ln, child, None))
                # I OR I -> child
                row_p = (1 - ineg[ch_i, :])
                i, j = np.triu_indices(len(row_p), k=1)  # Avoid (i, i) and duplicate pairs
                sums = row_p[i] + row_p[j]
                mask = sums > (1-args.err_max) * 0.9 # Get pairs where the sum is above threshold
                result_indices = list(zip(i[mask], j[mask]))
                for r in result_indices:
                    parent1 = self.graph.non_logical_nodes[r[0]]
                    parent2 = self.graph.non_logical_nodes[r[1]]
                    ln = LogicalNode([parent1, parent2], 'or')
                    promising.append(Edge(ln, child, None))
        if self.verbose: print(f'Number of promising edges: {len(promising)}')
        return promising
