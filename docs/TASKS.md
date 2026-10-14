# Task Families

This document specifies the eight v0.5 transition families: exact semantics,
the canonical `scan` accounting, the explicit predicate where one exists, the
released rule-card wording, edge cases, and the generation defaults. Names and
event shapes are neutral and mathematical; no game name appears in a prompt.

Every family carries a one-dimensional worked example rendered after its rule
card. Its after-state is computed by the reference simulator; agreement with
that same implementation is a consistency check, not independent validation.
The example is omitted when the alphabet cannot express its values. These
examples support the one-step anchor at `T=1` but do not eliminate rule-reading
or output-format errors. See `tests/test_prompt.py`.

All families use a bounded domain $\mathcal X_d=\prod_i\{0,\ldots,L_i-1\}$,
zero-based coordinates, no wrapping, and default value `0`. `W(x)` denotes the
value immediately before the current event. Every family supports
`SET(p,v)`: scan 1, effect 1 exactly when the stored value changes.

For every event, `effect` is the Hamming distance between input and output
states. Each family defines a canonical `scan`; it controls workload within a
family, not a common unit of difficulty across families. A core event is
*effective* when its predicate holds, or, without a predicate, when
`effect > 0`. The reference scan accounting is not a proven lower bound on the smallest
sufficient read. `STENCIL`, `MIX` and `RELABEL` admit valid no-op cases in their rules,
but their generators do not deliberately propose an ineffective branch.

The generation notes below describe default policies. Numeric development
defaults are maintained in `spatial_worlds.generation.config.FAMILY_DEFAULTS`;
the selected settings and overrides remain in the source configurations and
released instance metadata. See [evaluation conditions](evaluation.md) and
[release provenance](reproduction.md) for the current study.

## 1. Metric Paint — `PAINT(a, r, u, v)`

**Spatial role.** Membership in a metric region: the update support is
selected by distance rather than connectivity, direction, or motion.

**State.** `0, ..., M-1` with `M >= 2` (default `M = 4`).

**Semantics.** Let $B_1(a,r)=\{x\in\mathcal X_d:\|x-a\|_1\le r\}$. All cells
in $B_1(a,r)$ whose old value is `u` are simultaneously set to `v`; other
cells are unchanged. **Scan** $=|B_1(a,r)|$ (in-bounds cells only).
**Predicate**: the ball contains at least one cell with value `u`.

**Edge cases.** `u = v` is valid and changes nothing (the generator never emits
it). `u = 0` fills empty cells; `v = 0` clears. `r = 0` selects only `a`.

**Rule card.**
> PAINT(a,r,u,v): simultaneously replace u by v at every coordinate x with sum_i |x_i-a_i| <= r. All other cells remain unchanged.

**Generation.** Uniform initializer, density 0.45–0.65, `set_rate` 0.2,
`noop_rate` 0.1. The workload target is clipped ball size; discrete geometry
prevents exact matching at every anchor. Ineffective candidates take `u` from
values absent in the ball.

## 2. Box Reflection — `REFLECT(lo, hi, axis)`

**Spatial role.** A pure coordinate bijection before obstruction, collision,
or aggregation.

**State.** `0, ..., M-1`, `M >= 2` (default 4).

**Semantics.** For every `x` in the inclusive box, the new value at the
mirrored position (`x'_axis = lo_axis + hi_axis - x_axis`, other coordinates
unchanged) is the old value at `x`, zeros included; writes are simultaneous.
**Scan** = box volume $\prod_i(hi_i-lo_i+1)$. No predicate.

**Edge cases.** `lo_axis = hi_axis` is the identity (valid, effect 0; the
generator requires an extent of at least 2 on the reflected axis). Applying the
same event twice restores the state.

**Rule card.**
> REFLECT(lo,hi,axis): for every cell x in the inclusive box lo <= x <= hi, the new value at the mirrored position x' (x'_axis = lo_axis + hi_axis - x_axis, other coordinates unchanged) is the old value at x, including 0. All writes happen simultaneously. Cells outside the box remain unchanged.

**Generation.** Uniform initializer, density 0.45–0.65, `set_rate` 0.2,
`noop_rate` 0.1, workload target = box volume. Ineffective candidates prefer
boxes with no nonzero cell.

## 3. Local Stencil — `STENCIL(a, r)`

**Spatial role.** Synchronous local coupling inside a metric window. The
window uses the same Manhattan-ball definition as Metric Paint. The rule
contrast is conditional relabeling versus neighbourhood aggregation; their
selected workloads and generated trajectories are not a matched ablation.

**State.** `0, ..., M-1` (default `M = 4`), interpreted as residues mod `M`.

**Semantics.** Let $B_1(a,r)$ be the in-bounds ball and $N_1(x)$ the
in-bounds axis neighbours of `x`. For every `x` in the ball at once,
$W'(x)=(W(x)+\sum_{y\in N_1(x)}W(y))\bmod M$. Neighbours outside the ball
are read but never written; cells outside the ball are unchanged. **Scan**
$=\sum_{x\in B_1(a,r)}(1+|N_1(x)|)$, the ball size plus every neighbour
read. A ball covering the world gives $N+2E$ with $E$ the number of lattice
edges (241 on [81], 369 on [9,9], 448 on [4,4,5], 513 on [3,3,3,3]). The
**support** $|B_1(a,r)|$ is the family's workload level. No predicate.

**Edge cases.** `r = 0` updates only `a`. The all-zero world is a fixed point
of every event, and a ball whose cells and neighbours are all `0` changes
nothing. Rollouts in which a core event changes fewer than 10% of its ball are
rejected.

**Rule card.**
> STENCIL(a,r): simultaneously set every cell x with sum_i |x_i-a_i| <= r to (W(x) + sum W(y)) mod M, where y ranges over the in-bounds cells at Manhattan distance 1 from x, whether or not they lie within distance r of a. Every update reads the same old world; cells farther than r from a remain unchanged.

**Generation.** Uniform initializer, density 0.55–0.85, `set_rate` 0.2,
no ineffective-event control; workload axis = update-support size. A target
equal to the tier's nominal cell count selects the whole world, emitted as
`a = 0` and `r = sum(L_i - 1)`. Its support is the actual shape size.
Scan additionally counts every in-bounds neighbour read. Selected targets are retained in the source configurations; reference workloads
record realized supports. A common target does not guarantee equal support.

## 4. Occluded Mix — `MIX(p, r)`

**Spatial role.** Graph distance through occupied cells combined with
synchronous local aggregation. The support is the depth-limited breadth-first
region through nonzero cells (the geodesic ball, cut by empty cells wherever
occupancy occludes it) and the update is a modular neighbourhood sum
with a different modulus and an added offset, keeping reached cells nonzero. It is the state-determined counterpart of
`STENCIL` in the way that the geodesic ball is the state-determined
counterpart of the Manhattan ball, and it fills the state-determined
aggregation cell of the family taxonomy.

**State.** `0` empty; `1, ..., K` with $K = M-1 \ge 2$ (default `M = 4`, so
`K = 3`).

**Semantics.** If `W(p) = 0` nothing changes and scan is 1. Otherwise let $S$ be the set of
cells reachable from `p` by a path of at most `r` axis steps whose cells all
have nonzero old values (`p` is reached with 0 steps; an empty cell can never
be entered or passed through). For every $x \in S$ at once,
$W'(x) = 1 + \big(W(x) + \sum_{y \in N_1(x)} W(y)\big) \bmod K$, where
$N_1(x)$ are the in-bounds axis neighbours of `x`, reached or not (empty
neighbours contribute 0). All reads use the old world; cells outside $S$ are
unchanged, and a reached cell is never set to 0, so the occupancy mask and the
nonzero count are exactly conserved by a core event (interleaved `SET`
events can change occupancy). **Scan** $=\sum_{x\in S}(1+|N_1(x)|)$:
every reached cell plus every neighbour it reads, so the empty cells that stop
the region are read as neighbours of the cells they stop. The **support**
$|S|$ is the family's workload level. **Predicate**: none; the event is
effective when some reached cell changes (a cell keeps its value exactly when
its neighbour sum is $K-1 \bmod K$; no independence or uniformity of
these sums is assumed).

**Edge cases.** For a nonzero anchor, `r = 0` recomputes only `p` (scan
$1+|N_1(p)|$). A run of nonzero cells shorter than `r` in one dimension is
reached entirely and its end cells read the empty cells that bound it. A
nonzero cell inside the Manhattan ball of radius `r` but separated from `p`
by empty cells is not reached. With $K = 3$ the map on a region is affine over
$\mathbb Z_3$, which is an algebraic fact about the reference execution and
not a practical shortcut for a language model.

**Rule card** (rendered with the world's $K$; shown for `M = 4`).
> MIX(p,r): if W(p) = 0, nothing changes. Otherwise find every cell that can be reached from p by a path of at most r Manhattan-distance-1 steps whose cells all have nonzero old values (p itself counts as reached with 0 steps; a cell with old value 0 can never be entered or passed through). Then simultaneously set every reached cell x to 1 + ((W(x) + sum W(y)) mod 3), where y ranges over the in-bounds cells at Manhattan distance 1 from x, whether or not they were reached. Every update reads the old world; cells that were not reached remain unchanged.

**Generation.** Uniform initializer, density 0.55–0.70, `set_rate` 0.2
with the thermostat, no ineffective-event control (`noop_rate` null).
The proposer draws a nonzero anchor, runs a breadth-first pass and selects the
depth whose support is closest to the target. A rollout in which a core event
changes fewer than 10% of its support is rejected, as for `STENCIL`.
Approximate support matching does not match neighbour-read counts across
dimensions. In one dimension a region cannot leave its occupied run; short
runs under this initializer make the selected M-tier cell infeasible under
the generation policy. The rule itself remains well-defined in one dimension.

## 5. Component Relabel — `RELABEL(p, v)`

**Spatial role.** State-dependent connectivity without a boundary predicate.

**State.** `0` empty; `1, ..., K` labels (default 3). Adjacency is Manhattan
distance 1.

**Semantics.** Let `u = W(p)` (zero allowed) and $C$ the maximal
axis-connected component of cells with old value `u` containing `p`; set every
cell of $C$ to `v`. With $\partial C$ the in-bounds external vertex
boundary, **scan** $=|C|+|\partial C|$. No predicate.

**Edge cases.** `v = W(p)` is a no-op with unchanged scan; the generator never
emits it, because the anchor alone certifies that nothing changes. `W(p) = 0`
relabels the empty component (valid; the generator uses nonzero anchors only).
`v = 0` deletes the component.

**Rule card.**
> RELABEL(p,v): let u = W(p) (u may be 0). Find the maximal set of cells connected to p by Manhattan-distance-1 steps whose old values are all u, including p. Set every cell of that set to v.

**Generation.** Region initializer (random labelled partition with region
size calibrated to the target), density 0.50–0.65, `set_rate` 0.2 with the
distinct-label thermostat; SET writes avoid neighbouring labels when possible.
There is no ineffective-event control. Candidates within the relative
`tolerance` band tie and are sampled uniformly.

The proposer prefers a new label absent from the component's boundary. It can
instead delete a component holding at most a quarter of the occupied cells:
with probability 0.15 this is tried first, and deletion is also the fallback
when no separate label is available. Only when neither option is possible
does it use a merging label. `merge_contacts` records boundary cells already
holding the new label. A small alphabet can restrict merge avoidance,
especially with many boundary contacts; the selected study uses six nonzero
labels. This is a sampling choice, not a theorem that every high-dimensional
component touches every label.

## 6. Boundary Layer — `LAYER(p, s)`

**Spatial role.** Component tracking plus a per-cell test of exposure: which
cells of a component touch empty space, and which empty cells touch the
component. The two signs are the erosion and the dilation of mathematical
morphology, applied to one component.

**State.** `0` empty; `1, ..., K` labels (default `K = 6`). `s` is `-1` or
`+1`.

**Semantics.** If `W(p) = 0` nothing changes. Otherwise let `u = W(p)`, $C$
the maximal axis-connected set of cells with old value `u` containing `p`,
and $\partial C$ its in-bounds boundary. If `s = -1`, every cell of $C$ that
has an in-bounds neighbour with old value `0` becomes `0`. If `s = +1`, every
cell of $\partial C$ with old value `0` becomes `u`. All reads use the old
world. **Scan** $=|C|+|\partial C|$ for both signs (1 for an empty anchor).
**Predicate**: none; both signs are effective exactly when $\partial C$
contains an empty cell, and a fully enclosed component is a no-op for both
signs once the whole boundary has been checked.

**Edge cases.** A component touching the world edge is eroded only where it
touches an in-bounds empty cell; space outside the world is not empty space.
Erosion can split a component. Dilation can bring a component into contact
with other labels but never merges it with them, and it merges it with a
same-label component only through a filled cell that touched both. In one
dimension a run loses or gains at most its two end cells.

**Rule card.**
> LAYER(p,s): if W(p) = 0, nothing changes. Otherwise let u = W(p) and find the maximal Manhattan-connected set of cells with old value u, including p. If s = -1, set to 0 every cell of that set that has an in-bounds neighbor with old value 0. If s = +1, set to u every in-bounds cell with old value 0 that is a neighbor of a cell of that set. Positions outside the world are not neighbors; nothing else changes.

**Generation.** Region initializer with distinct neighbouring labels,
density 0.45–0.60, `set_rate` 0.2 with the distinct-label thermostat,
`noop_rate` 0.1. An ineffective proposal is an enclosed component whose scan
is at least half the target; if none exists, the slot becomes effective.
The proposer offers both signs for components with an empty boundary cell;
density enters the selection policy.

Growth and erosion are not inverse operations. For example, on the bounded
line `[0,1,0]`, growth from the centre gives `[1,1,1]`; erosion then leaves
it unchanged because there is no in-bounds empty neighbour. The generator's
recent-state return check rejects effective candidates that reproduce one of
the last eight states. This reduces short cycles without asserting
invertibility or ruling out all longer cycles.

## 7. Obstructed Motion — `MOVE(v, axis, direction, k)`

**Spatial role.** Occupancy-constrained transport with an explicit processing
order.

**State.** `0` empty; `1, ..., K` entity classes (default `K = 4`). `v` is
nonzero, `direction` in `{-1, +1}`, `k >= 0`.

**Semantics.** All entities of value `v` are processed in decreasing axis
coordinate for `direction = 1` and increasing for `direction = -1`. Each moves
one cell at a time along the axis, at most `k` cells, stopping before the
boundary or the first currently occupied cell; earlier moves are visible to
later ones and entities never merge. Because trajectories stay on their own
axis-parallel fiber and fibers share no cells, this equals processing every
fiber independently front to back. **Scan** = number of processed entities +
number of in-bounds destination cells whose occupancy was checked (a check
that finds the cell occupied counts; the boundary is not a cell). No
predicate.

**Edge cases.** `k = 0`: scan equals the entity count, no change. An entity
at the wall makes no check; one blocked immediately makes one.

**Rule card.**
> MOVE(v,axis,direction,k): take every cell with value v, ordered by decreasing coordinate on axis if direction=1 and increasing if direction=-1, so that the cells nearest the leading edge are handled first. In that order, move each one along axis in the sign of direction, one cell at a time and at most k cells, stopping as soon as the next cell is outside the world or nonzero (including cells already moved into). Each move clears the old cell and writes v to the new cell; values never merge, and a block of adjacent v cells therefore moves as a unit.

**Generation.** Uniform initializer, density 0.35–0.50, `set_rate` 0.3,
`noop_rate` 0.1, `k` in 1..min(4, side−1). The workload target is the
entity count plus realized destination checks, not maximum displacement.

## 8. Aggregating Compression — `COMPRESS(axis, direction)`

**Spatial role.** Global transport, collision, and aggregation along a
partition of the world into ordered axis-aligned fibers.

**State.** `0` empty; positive tiles. Generation uses powers of two up to
$2^{12}$. A merge outside the declared alphabet raises an error rather than
being clipped; replay validation checks this state-dependent constraint.

**Semantics.** Independently on every line parallel to `axis`: list the
nonzero values from the destination edge (coordinate 0 for `direction = -1`,
the highest coordinate for `+1`); walk the list, replacing each pair of
consecutive equal values by their sum and continuing after the pair; write the
result back from the destination edge and fill the rest with zeros. **Scan**
$=|\mathcal X_d|$. No predicate.

**Edge cases.** `[2,2,2,0]` toward 0 gives `[4,2,0,0]`; `[2,2,4,4]` gives
`[4,8,0,0]`; `[4,2,2,0]` gives `[4,4,0,0]`. Repeating the direction can
merge again: `[2,2,4]` becomes `[4,4,0]`, then `[8,0,0]`. A merged tile
cannot merge again within the same event, but can in the next one. Per-fiber
sums are conserved and the nonzero count never increases during a core event.

**Rule card.**
> COMPRESS(axis,direction): on every line parallel to axis, independently: list the nonzero values in order starting from the destination edge (coordinate 0 if direction=-1, the highest coordinate if direction=1). Walk that list from its start; whenever two consecutive values are equal, replace them by one value equal to their sum and continue after the pair, so a merged value cannot merge again in this event. Write the resulting list back starting at the destination edge and set every remaining cell of the line to 0.

**Generation.** Small-tile initializer (values 1 and 2, 9:1), density
0.15–0.60, `set_rate` 0.5 with the spawn policy, `noop_rate` 0.1, workload
axis = N level; this is the only family whose core event always scans the
whole world. In one
dimension the equilibrium density is low because every merge happens on a
single fiber; the realized density is recorded, and headline comparisons for
this family use $d \ge 2$.

## Family relationships

- Metric Paint and Local Stencil share the Manhattan ball and contrast
  conditional relabeling with modular aggregation over neighbourhoods.
- Local Stencil and Occluded Mix contrast metric and occupancy-constrained
  support, but also differ in their arithmetic. Even without empty cells their
  outputs need not agree: on `[1,1,1]`, centre/radius 1 and alphabet 0..3,
  STENCIL gives `[2,3,2]`, whereas MIX gives `[3,1,3]`. Their selected support
  targets are approximately calibrated, not identical matched conditions.
- Component Relabel, Boundary Layer, and Occluded Mix share the breadth-first
  primitive and differ in what is traced and written: the whole equal-valued
  component, its exposed layer or its empty rim, or a depth-limited region
  through any nonzero cell whose values are aggregated rather than copied.
- Box Reflection is collision-free transport; Obstructed Motion adds occupancy
  and sequential order; Aggregating Compression adds collision and
  aggregation.

These relationships motivate exploratory comparisons, not isolated causal
ablations. Raw accuracies and scan counts are not interchangeable measures
of difficulty across families.
