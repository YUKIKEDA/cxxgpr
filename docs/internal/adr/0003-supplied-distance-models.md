# 0003: Supplied-distance models are part of the Exact GPR scope

- Status: Accepted
- Date: 2026-11-21
- Issue: #89, #104

## Context

[ADR 0001](0001-exact-gpr-cpp23-scope.md), REQ-01 and `AGENTS.md` named the models of the project as `Gpr` / `FittedGpr`, while `DistanceKernel` (supplied squared distances) was in scope only as a kernel. The persistence specification (FMT-05, FMT-07) already described `LoadedDistanceGpr<C>` and a `distance` section of `config.json`, but no model could be trained on a `DistanceKernel`: `Gpr` rejects it at compile time, and the loader rejected the `distance` key. Issue #89 asked which of the two is authoritative.

## Decision

1. The owner chose to implement the supplied-distance models (option B of #89). `DistanceGpr<O, P, C>` and `FittedDistanceGpr<O, P, C>` (DES-21), with `C` in `{DistanceOnly, WithPoints}`, all four precision policies and all five optimizers, are part of the Exact GPR scope, and so is their persistence through `LoadedDistanceGpr<C>` (FMT-05, FMT-07).
2. They are separate types, not a specialization of `Gpr`: the supplied tables replace the coordinates, so the input arguments, the query supply and the distance-cache policy differ, while the lifecycle, the error contract (`std::expected`, failure leaves every object unchanged), the numerics (DES-04, DES-05, DES-13, DES-14) and the allocation contract (REQ-05, REQ-11) are those of the coordinate model.
3. The work is split into the roadmap rows REL-5 (specification, #104), REL-6 (training and inference, #105) and REL-7 (persistence, #106).

## Consequences

- REQ-01, `AGENTS.md` and the `scope` rule list the supplied-distance models next to `Gpr` / `FittedGpr`. ADR 0001 stays as written; this ADR extends its scope decision.
- An independent reference is available for every derived quantity: Euclidean squared distances of a coordinate dataset fed to a `DistanceKernel` with the same leaves must reproduce the coordinate model (DES-21).
- `Sgpr` / `Svgp` / `OnlineGpr` and inducing-point coordinate derivatives remain out of scope.
