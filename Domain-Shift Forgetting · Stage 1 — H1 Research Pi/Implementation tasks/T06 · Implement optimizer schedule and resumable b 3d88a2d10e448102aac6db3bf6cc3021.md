# T06 · Implement optimizer schedule and resumable branching

Depends on: T05
Order: 6
Phase: Implementation
Priority: P0
Status: Not started

[Domain-Shift Forgetting · Stage 1 — H1 Research Pilot](../../Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)

[01 · Protocol v3 — H1 only](../01%20%C2%B7%20Protocol%20v3%20%E2%80%94%20H1%20only%203d88a2d10e44812ea734fdeff5891429.md)

## Objective

Implement optimizer schedule and resumable branching for the H1-only Stage 1 pilot.

## Acceptance criteria

- [ ]  Preserve all Adam moments, parameter step counters, LR/gate and data/RNG state.
- [ ]  Honor inactive-gain semantics and immutable switch-state branching.

## Dependencies

T05

## Evidence to attach

Commit or manifest URL, validation output or run-record links. Actual time and costs where relevant. Do not mark complete without evidence.

## Status

Not executed. Scheduling is dependency-based; no calendar deadline or assignee notification has been created.