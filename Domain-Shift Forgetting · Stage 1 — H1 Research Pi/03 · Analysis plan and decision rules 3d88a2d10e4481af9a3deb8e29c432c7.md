# 03 · Analysis plan and decision rules

[Domain-Shift Forgetting · Stage 1 — H1 Research Pilot](../Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)

## Primary estimand

For each seed, let L(m,d,s) denote token-weighted full-development WEB CE, in nats per supervised token, after s continuation updates; d is web or code. Taper-minus is T; RMS is R.

$$
D(s)=[L(T,code,s)-L(T,web,s)]-[L(R,code,s)-L(R,web,s)].
$$

Primary endpoint D=D(6104). Define D-plus identically for optional Taper-plus. Never substitute a better-looking checkpoint for the primary endpoint.

Actual forgetting is F(m,d)=L(m,d,6104)−L(m,prefix). Report every F even if D is positive.

$$
G_{web}=[L(T,prefix)-L(R,prefix)]-[L(T,web,6104)-L(R,web,6104)].
$$

$$
D=[F(T,code)-F(R,code)]+G_{web}.
$$

G-web is a descriptive change in the between-condition gap during web continuation, not a causal recovery bound. Set Q=D−G-web to report direct differential code-branch forgetting. Validate this identity in analysis tests.

## Evaluation schedule

Include s=0 in both quick and full dev. Prefix full-dev: global updates 763, 3052, 6104; then 6409, 6714, 7019, 7324, 7629, 7934, 8239, 8544, 8849, 9156. Quick dev also at 9156.

Continuation quick-dev: deduplicated sorted union of {0,1,2,5,10,20}, every 50 through 1000, every 100 from 1100 through 6100, plus {1525,3050,6104}.

Continuation full-dev: union of {0,1,10,100}, every 305 from 305 through 6100, plus {1525,3050,6104}. Save weight checkpoints and per-document/per-class sufficient statistics at every full-dev point.

Persist D at 1525, 3050 and 6104 using full dev. Quick-dev trajectories describe transients; do not mix their numerical values into the full-dev primary contrast.

Fit slopes over the last five prefix full-dev points and report both slopes in nats per million tokens and their difference. Avoid an unstable ratio when the RMS slope is near zero.

## Matched adaptation, secondary

For each seed, set C-j to RMS-code non-whitespace quick-dev CE at s-j=1525,3050,6104. The target rule is frozen; its numerical value is derived separately within each seed.

Match Taper-code using the first chronological downward crossing of C-j between adjacent observed quick-dev checkpoints no more than 100 updates apart. An exact match uses the earliest observed exact checkpoint. Do not match a target already surpassed at s=0 by extrapolation; mark it as no post-switch downward crossing.

Interpolate Taper web quick-dev loss and tokens within that bracket in code-loss coordinates. If losses are flat, use an exact match only; do not divide by zero. Store both endpoint values and bracket length. No extrapolation, smoothing or choosing a later favorable recrossing.

Compute matched differential actual forgetting relative to each model's own s=0 web quick-dev loss. Primary matched target for the decision is the furthest of these three targets reached by all primary seeds. No common target means inconclusive adaptation comparison.

Report all three targets, nearest-endpoint sensitivity, and tokens to target. Interpolation is a descriptive curve estimate, not an actual evaluated model or evidence of equal training histories. Non-whitespace code CE measures adaptation proxies, not executable-code quality.

## Tail and token decomposition

For each web document i use mean token CE with fixed scored-token count n-i. Compute D-i with the same four-way contrast. Verify sum(n-i×D-i)/sum(n-i)=D within 1e-6 nats.

Rank documents by mean D-i for the unweighted median and a symmetric document-count 1%-trimmed token-weighted mean. Separately rank by signed token contribution c-i=n-i×D-i/N for top-contributor analysis. Select ceil(0.01×number_of_documents), ties by document ID.

Report signed top-contributor sum, its ratio to D when D>0, and its fraction of total positive contribution sum(max(c-i,0)). Ratios to a near-zero net D are unstable and may exceed 100%; never silently clamp them.

Outlier-concentrated flag: D≥0.015 and top 1% contribute >50% of net D. Include positive-mass fraction and plots before interpreting. It guides later investigation; it does not exclude layerwise causes.

For W/A/P/X report count, CE, and frequency-weighted contribution to D; contributions must sum to D. R is an overlapping frequency flag and is reported separately. Rare/control subgroups with <100 scored labels are descriptive only.

## Zero-shot audit and diagnostics

At the switch, evaluate source and code calibration pools without updates. For each of 12 sites measure energies of z entering QKV/MLP, log-k=0.5×log(E-web/E-code), and norm p99/p50. Use epsilon 1e-12 only as numerical protection; report any near-zero denominator. For RMS also record pre-norm h energy.

Report all positions and a sensitivity mask excluding window positions 0–15 and the first token after EOS; identify these as positional exclusions, not verified attention sinks. Record embedding norms by class/R.

At s=0,10,100,1525,6104 log activation RMS, branch-output/residual-input ratios and gains. Record gradient norm and clipping rate throughout training. Attention entropy runs on fixed small probe batches in a separate forward pass.

These observations do not establish a correction mechanism. No fixed log-k threshold can prove corrections are a no-op.

## Fixed-sample decision hierarchy

Apply rules in this order after the planned three primary seed groups. Keep every seed and failed attempt visible.

1. **INVALID OR INCOMPLETE:** missing primary seed, unresolved correctness/data failure, nonfinite primary run or resource termination. No clean H1 conclusion.
2. **COMPARABILITY / ADAPTATION LIMITED:** primary guardrails fail. Report observations and limitations; do not declare supported H1 or a null.
3. **OPPOSITE DIRECTION:** mean D≤−0.03. Stop the proposed positive-effect direction; report adaptation and F values.
4. **PROCEED TO DESIGN THE NEXT STUDY:** all of the following hold: mean D≥0.03; each seed D>0; mean D(3050)≥0.015; mean Q>0.5×mean D; mean matched differential forgetting>0 at the furthest common target. These are screening criteria, not hypothesis-test significance.
5. **TRANSIENT ONLY:** mean D<0.015 and mean quick-dev D(s)≥0.05 at at least one common scheduled s≤1000. Stop this persistent-effect direction; report the transient.
6. **STOP — SMALL OBSERVED EFFECT:** mean D<0.015. No convincing persistent excess detected within this pilot; not proof of equivalence.
7. **INCONCLUSIVE:** everything else. No automatic added seeds, changed endpoints or next-stage launch.

The Q criterion prevents advancement driven mainly by differing web-continuation gains, but remains a descriptive heuristic.

## Uncertainty and claims

Report all seed values, mean and sample SD. A descriptive fixed-n paired 95% t interval is mean D ±4.303×s-D/sqrt(3). With only three seeds, its assumptions and precision are fragile. Document-level bootstrap, if added, estimates evaluation sampling conditional on trained models and must never replace seed-level uncertainty.

All data here are development data. Do not claim confirmed H1, a population lower bound of 0.03, or statistical equivalence. Any later confirmation requires a newly frozen plan, independent seeds, reserved test evaluation, and prospectively justified power/sample size.

Do not use s-D≤0.015 as a guaranteed power gate. For planning, four-seed t-interval half-width 1.591×s-D is only a rough sensitivity calculation; pilot variance is uncertain.

Auxiliary results are secondary/exploratory. If all three plus seeds ran, report paired A=D-minus−D-plus with uncertainty. A favorable A suggests the auxiliary objective warrants study. It does not uniquely identify scale anchoring, and no RMS-plus condition is included. Apply no confirmatory significance claims to this family.

## Synthetic analysis acceptance

- [ ]  Four identical branches produce D=0.
- [ ]  Known synthetic shifts recover exact D, F and Q decomposition.
- [ ]  Weighted document and disjoint-class contributions reconstruct D.
- [ ]  Matching covers exact, crossing, flat, nonmonotonic and unreachable cases.
- [ ]  Decision rules cover boundaries 0.015/0.03 and missing data without overlapping classifications.
- [ ]  Plot labels distinguish full/quick dev and actual/interpolated checkpoints.