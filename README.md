# Niagara

Niagara is a causal discovery method for interval-based event sequences. 
It models event starts, active intervals, and event ends, and selects causal mechanisms between events using a minimum-description-length score.

This repository contains the reusable implementation accompanying the AAAI 2026 paper *Causal Discovery from Interval-Based Event Sequences* (see citation below). 
Version 0.1.0 preserves the behavior of the implementation used for the publication.

## Installation

```console
pip install niagara-cd
```

## Five-minute example

Provide one row per event occurrence.
The columns are the event name, start time, and end time; an optional fourth column identifies a sequence.

```python
import numpy as np

from niagara import fit

intervals = np.array(
    [
        ["rain", 0.0, 2.0],
        ["traffic", 1.0, 3.0],
        ["rain", 5.0, 7.0],
        ["traffic", 6.0, 8.0],
    ],
    dtype=object,
)

result = fit(intervals, horizon=10.0)

print(result.feature_names)
print(result.adjacency)
print(result.edges)
print(result.mechanisms)
```

This example produces an event-level edge from `rain` to `traffic`:

```text
('rain', 'traffic')
[[0. 1.]
 [0. 0.]]
```

The feature order gives the row and column labels of the adjacency matrix.
Rows are causes (parents), columns are effects (children), and a value of `1` at position `[i, j]` means `feature_names[i] → feature_names[j]`.
Here, the value at row `rain` and column `traffic` is `1`.

`result.edges` provides the same event-level result without requiring matrix indexing:

```python
print(result.edges)
# (('rain', 'traffic'),)
```

To inspect whether the mechanism concerns an event starting, being ongoing, or ending, use `result.mechanisms`:

```python
mechanism = result.mechanisms[0]

print(mechanism.parents)
# (Endpoint(event='rain', kind='interval'),)

print(mechanism.child)
# Endpoint(event='traffic', kind='start')

print(mechanism.operator)
# None
```

This means that **rain being ongoing is selected as a cause of traffic starting**.
Endpoint kinds are `start`, `interval`, and `end`.
For a logical mechanism with two parents, `operator` is `and`, `or`, or `xor`; it is `None` for a single-parent mechanism.

A Pandas DataFrame may be supplied instead when it has `event`, `start`, and `end` columns, plus an optional `sequence` column.
Rows are kept in their supplied order, and event names are preserved.

## Public API

`fit(data, *, horizon, feature_names=None, config=None)` fits Niagara and returns a `NiagaraResult`.

`NiagaraConfig` controls exhaustive or partial search, the maximum error proportion, start-only mode, the OR heuristic, and MDL precision.
Defaults match the publication implementation.

`NiagaraResult.adjacency` is the event-level parent-by-child adjacency matrix, labeled by `NiagaraResult.feature_names` on both axes.
`NiagaraResult.edges` contains the corresponding named `(parent, child)` pairs.
`NiagaraResult.mechanisms` contains the more detailed start/interval/end mechanisms as `CausalMechanism` objects.
Use `result.to_dict()` or `result.to_json(path)` for serialization.

## Command line

Given `events.csv`:

```csv
event,start,end
rain,0,2
traffic,1,3
rain,5,7
traffic,6,8
```

Run:

```console
niagara events.csv --horizon 10 --output result.json --adjacency-output adjacency.csv
```

## Citation

If you use Niagara, please cite:

> Lénaïg Cornanguer, Joscha Cüppers, and Jilles Vreeken.
> Causal Discovery from Interval-Based Event Sequences.
> In: Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), AAAI, 2026.

Citation metadata is also available in `CITATION.cff`.

## License

Niagara is distributed under the BSD 3-Clause License.
