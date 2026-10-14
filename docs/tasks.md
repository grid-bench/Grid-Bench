# Task definitions

An instance specifies a bounded integer-valued grid, its initial state, and a
complete event sequence. Apply every event in order. The answer is the entire
state after the final supplied event; no future event needs to be predicted.

## Shared conventions

The domain is $X=\prod_i\{0,\ldots,L_i-1\}$. Coordinates and axis indices
start at zero. The world does not wrap. Unlisted cells have value zero.
$W$ denotes the state before the current event, and $W'$ the state after it.
Cells outside an operation's update set remain unchanged.

Let $N(x)$ be the in-bounds cells at Manhattan distance one from $x$,
$B(a,r)=\{x\in X:\|x-a\|_1\le r\}$, and $\partial C$ the in-bounds
cells outside $C$ adjacent to it. Outside the world is not an additional layer
of empty cells. All coordinates in an event must be in bounds; radii and motion
limits are nonnegative integers, and axes are valid indices.

All families support `SET(p,v)`, which assigns `v` to coordinate `p`.
The other events in a history belong to one family. The first seven families
use an alphabet $\{0,\ldots,M-1\}$, with $M\ge2$; MIX requires $M\ge3$.
COMPRESS permits zero and a declared set of positive values.

## PAINT — metric replacement

Family ID: `metric_paint`. Event: `PAINT(a,r,u,v)`.

For all cells $x\in B(a,r)$ with $W(x)=u$, set $W'(x)=v$ simultaneously.
Both `u` and `v` must be allowed values. Radius zero selects only the anchor.
`u=0` can fill empty cells; `v=0` clears cells; `u=v` is a valid no-op.

## REFLECT — box reflection

Family ID: `box_reflection`. Event: `REFLECT(lo,hi,axis)`.

Within the inclusive box `lo <= x <= hi`, move each value, including zero,
to its reflected coordinate:

$$x'_{axis}=lo_{axis}+hi_{axis}-x_{axis},\qquad x'_j=x_j\quad(j\ne axis).$$

All reads use the old state and all writes are simultaneous. Each lower bound
must be at most its upper bound. A box of width one along the reflected axis
is unchanged; applying the same reflection twice restores the state.

## STENCIL — local aggregation

Family ID: `local_stencil`. Event: `STENCIL(a,r)`.

For every $x\in B(a,r)$, simultaneously set

$$W'(x)=\left(W(x)+\sum_{y\in N(x)}W(y)\right)\bmod M.$$

Neighbors outside the ball are read but not updated. Every update reads the
same old world. Radius zero updates only the anchor; an all-zero neighborhood
remains zero.

## MIX — aggregation through occupied cells

Family ID: `occluded_mix`. Event: `MIX(p,r)`.

An empty anchor is a no-op. Otherwise, let $S$ contain the cells reachable
from `p` by at most `r` axis-adjacent steps through cells with nonzero old
values. The anchor is reached in zero steps; empty cells cannot be entered
or crossed. For every $x\in S$, simultaneously set

$$W'(x)=1+\left(W(x)+\sum_{y\in N(x)}W(y)\right)\bmod(M-1).$$

The sum reads all in-bounds neighbors, reached or not. Every read uses the
old world. A reached cell remains nonzero, so a MIX event preserves occupancy.
Radius zero recomputes only a nonzero anchor. A nearby cell separated by empty
space is not necessarily reached.

## RELABEL — component relabeling

Family ID: `component_relabel`. Event: `RELABEL(p,v)`.

Let $u=W(p)$ and let $C$ be the maximal axis-connected component of cells
with value $u$ containing `p`. Set all cells in $C$ to `v`. The anchor may be
zero, in which case the empty component is relabeled. `v=0` deletes a component;
`v=u` is a no-op. Writing a neighboring component's label can merge components.

## LAYER — component growth and erosion

Family ID: `boundary_layer`. Event: `LAYER(p,s)`, with `s` equal to `-1` or `+1`.

An empty anchor is a no-op. Otherwise, let $u=W(p)$ and $C$ be its maximal
same-valued component.

- For `s=-1`, set to zero every cell of $C$ with an in-bounds zero neighbor.
- For `s=+1`, set to $u$ every zero cell in $\partial C$.

All reads use the old state. The world boundary alone does not expose a cell
to erosion. Erosion can split a component; growth can join components sharing
the anchor's label. Growth and erosion are not inverses: growing the middle
cell of `[0,1,0]` produces `[1,1,1]`, which erosion leaves unchanged because
there are no in-bounds zero neighbors.

## MOVE — obstructed motion

Family ID: `obstructed_motion`. Event: `MOVE(v,axis,direction,k)`.

`v` is nonzero and `direction` is `-1` or `+1`. Select all cells of value `v`
and process them from the leading edge: decreasing axis coordinate for `+1`,
increasing for `-1`. Each moves one cell at a time, at most `k` cells, stopping
before the boundary or the first currently occupied cell. Clear the old
position and write `v` at its destination. Earlier moves are visible to later
ones; values never merge.

Different axis-parallel lines do not interact, so ties between those lines
have no effect. `k=0` is a no-op. An adjacent block of selected values can move
together because its leading value moves first.

## COMPRESS — packing and merging

Family ID: `aggregating_compression`. Event: `COMPRESS(axis,direction)`.

Process each line parallel to `axis` independently. The destination edge is
coordinate zero for `direction=-1` and the highest coordinate for `+1`.

1. List the line's nonzero values starting at the destination edge.
2. Scan that list in order. Replace each consecutive equal pair by its sum and
   continue after the pair. A merged value cannot merge again in this event.
3. Write the result from the destination edge and fill the rest with zeros.

For example, toward coordinate zero, `[2,2,2,0]` becomes `[4,2,0,0]`, and
`[2,2,4,4]` becomes `[4,8,0,0]`. Another event may merge values again.
Each line's sum is conserved. A merge producing a value outside the declared
alphabet is rejected during reference execution.

## Reference workload accounting

Every event records `effect`, the number of changed cells, and `scan`, the
family's canonical read-accounting count:

| Operation | `scan` |
| --- | --- |
| SET | 1 |
| PAINT | $|B(a,r)|$ |
| REFLECT | Inclusive box volume |
| STENCIL | $\sum_{x\in B(a,r)}(1+|N(x)|)$ |
| MIX | $\sum_{x\in S}(1+|N(x)|)$; 1 for an empty anchor |
| RELABEL | $|C|+|\partial C|$ |
| LAYER | $|C|+|\partial C|$; 1 for an empty anchor |
| MOVE | Selected entities plus in-bounds destination occupancy checks |
| COMPRESS | Total number of grid cells |

A blocked destination check counts for MOVE; a step outside the world does
not. STENCIL and MIX count repeated neighbor reads separately. These are
reference execution measurements, not measurements of model computation.
History length counts all events, including SET.

The [generation guide](generation.md) describes how instances are sampled.
Sampling preferences do not change the valid transitions above. Executable
implementations are in [src/grid_bench/families](../src/grid_bench/families).
