"""
Interval class to ease interval manipulation
"""

from bisect import bisect_right
import numpy as np


class Interval:
    def __init__(self, start, end, count=1):
        self.start = start
        self.end = end
        self.count = count

    def __repr__(self):
        return f"Interval({self.start}, {self.end}, {self.count})"

    def overlaps(self, other):
        """Check if this interval overlaps with another."""
        return not (self.end <= other.start or self.start >= other.end)

    def merge(self, other, sum_count=False):
        """Merge overlapping intervals without increasing count."""
        count = self.count + other.count if sum_count else max(self.count, other.count)
        if self.overlaps(other):
            return Interval(min(self.start, other.start), max(self.end, other.end), count)
        return None

    def get_inverse(self):
        """ Return a new interval with count 0 if the count was positive and 1 id the count was 0. """
        new_count = 1 if self.count == 0 else 0
        return Interval(self.start, self.end, new_count)


class IntervalList:
    def __init__(self, intervals=None, n=1000):
        self.intervals = sorted(intervals, key=lambda x: x.start) if intervals else []
        self.n = n

    def __repr__(self):
        return f"IntervalList({self.intervals})"

    def add_interval(self, new_interval, sum_count=False):
        """Add an interval and merge if necessary (without increasing count)."""
        self.intervals.append(new_interval)
        self.intervals.sort(key=lambda x: x.start)
        self._merge_intervals(sum_count=sum_count)

    def create_and_add_interval(self, start, end, count):
        self.intervals.append(Interval(start, end, count))

    def _merge_intervals(self, sum_count=False):
        """Merge overlapping intervals."""
        merged = []
        for interval in self.intervals:
            if merged and merged[-1].overlaps(interval):
                merged[-1] = merged[-1].merge(interval, sum_count=sum_count)
            else:
                merged.append(interval)
        self.intervals = merged

    def fill_gaps(self):
        """Ensure the intervals cover the full range [0, max_time] by inserting missing zero-count intervals."""
        new_intervals = []
        prev_end = 0

        for interval in self.intervals:
            if interval.start > prev_end:
                new_intervals.append(Interval(prev_end, interval.start, 0))  # Fill gap
            new_intervals.append(interval)
            prev_end = interval.end

        if prev_end < self.n:
            new_intervals.append(Interval(prev_end, self.n, 0))  # Ensure full coverage

        self.intervals = new_intervals

    def xor_op(self, other):
        """Compute XOR: return (A active while B inactive, B active while A inactive) as separate lists."""
        a_active, b_active, either_active = [], [], []
        i, j = 0, 0

        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i], other.intervals[j]

            start = max(a.start, b.start)
            end = min(a.end, b.end)

            if start < end:  # Overlapping region
                if a.count > 0 and b.count == 0:
                    a_active.append(Interval(start, end, a.count))
                    either_active.append(Interval(start, end, a.count))
                if b.count > 0 and a.count == 0:
                    b_active.append(Interval(start, end, b.count))
                    either_active.append(Interval(start, end, b.count))

            if a.end <= b.end:
                i += 1
            if b.end <= a.end:
                j += 1

        resA = IntervalList(a_active, self.n)
        resA.fill_gaps()
        resB = IntervalList(b_active, self.n)
        resB.fill_gaps()
        resE = IntervalList(either_active, self.n)
        resE.fill_gaps()

        return resA, resB, resE

    def and_op(self, other):
        """Compute AND: return intervals where both A and B are active."""
        both_active = []
        i, j = 0, 0

        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i], other.intervals[j]

            start = max(a.start, b.start)
            end = min(a.end, b.end)

            if start < end:  # Overlapping region
                if a.count > 0 and b.count > 0:
                    both_active.append(Interval(start, end, min(a.count, b.count)))  # Choose min count

            if a.end <= b.end:
                i += 1
            if b.end <= a.end:
                j += 1

        res = IntervalList(both_active, self.n)
        res.fill_gaps()

        return res

    def or_op(self, other):
        """Compute OR: return intervals where A and/or B are active."""
        active = []
        i, j = 0, 0

        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i], other.intervals[j]

            start = max(a.start, b.start)
            end = min(a.end, b.end)

            if start < end:  # Overlapping region
                if a.count > 0 or b.count > 0:
                    active.append(Interval(start, end, a.count + b.count))

            if a.end <= b.end:
                i += 1
            if b.end <= a.end:
                j += 1

        res = IntervalList(active, self.n)
        res.fill_gaps()

        return res

    def convert_to_interval_list(self):
        """ Convert to a simple list of interval lists without counts. """
        return [[elem.start, elem.end] for elem in self.intervals if elem.count > 0]

    def weighted_duration(self):
        """ Return the sum of all intervals weighted with the counts. """
        return sum(
            [(t.end - t.start) * t.count if t.end != np.inf else (self.n - t.start) * t.count for t in self.intervals])

    def duration(self):
        """ Return the sum of all intervals. """
        return sum(
            [(t.end - t.start) if t.end != np.inf and t.count > 0 else (self.n - t.start) * min(1, t.count) for t in self.intervals])

    def get_intervals(self):
        """Returns a sorted list of intervals."""
        return sorted(self.intervals, key=lambda x: x.start)

    def find_interval(self, time):
        """Binary search to quickly find which interval contains 'time'."""
        idx = bisect_right([iv.start for iv in self.intervals], time) - 1
        if idx < len(self.intervals) and self.intervals[idx].start <= time < self.intervals[idx].end:
            return self.intervals[idx]
        if time == self.n: return self.intervals[-1]
        return None

    def duration_between(self, t1, t2, weighted=False):
        """ Duration with positive count between to time values. """
        def weight(count):
            return 0 if count == 0 else (1 if not weighted else max(1, count * weighted))

        int_starts = [iv.start for iv in self.intervals]
        i1 = bisect_right(int_starts, t1) - 1
        i2 = bisect_right(int_starts, t2) - 1

        if i1 == i2: return (t2 - t1) * weight(self.intervals[i1].count)

        dur = (self.intervals[i1].end - t1) * weight(self.intervals[i1].count)
        dur += (t2 - self.intervals[i2].start) * weight(self.intervals[i2].count)
        for iv in self.intervals[i1+1:i2]:
            dur += (iv.end - iv.start) * weight(iv.count)

        return dur

    def get_inverse(self):
        """ Return a new IntervalList with count 0 for any interval with positive count and 1 for intervals with count 0. """
        return IntervalList([i.get_inverse() for i in self.intervals], self.n)

    def positive_count(self, time):
        """ Return whether the interval that contains time has positive count. """
        return self.find_interval(time).count > 0

    def get_last_inactive(self, time):
        """ Return the last time the event count was equal to 0. """
        curr_interval = self.find_interval(time)
        i = self.intervals.index(curr_interval)
        for index in range(i, -1, -1):
            if self.intervals[index].count == 0:
                return self.intervals[index]
        return Interval(0, 0, 0)

    def get_last_active(self, time):
        """ Return the last time the event count was equal to 0. """
        curr_interval = self.find_interval(time)
        i = self.intervals.index(curr_interval)
        for index in range(i, -1, -1):
            if self.intervals[index].count > 0:
                return self.intervals[index]
        return Interval(0, 0, 0)

    def default(self):
        self.intervals = [Interval(0, self.n, 1)]
        return self

    def jaccard_index(self, other):
        intersec = self.and_op(other)
        union = self.or_op(other)
        iou = intersec.weighted_duration() / union.weighted_duration()
        return iou

    def set_counts(self, val=1, inplace=True):
        """ Set a given value to all interval counts greater than 0 ."""
        if inplace:
            for interval in self.intervals:
                if interval.count > 0:
                    interval.count = val
        else:
            new = IntervalList()
            for interval in self.intervals:
                count = val if interval.count >= 1 else interval.count
                new.intervals.append(Interval(interval.start, interval.end, count))
            return new

