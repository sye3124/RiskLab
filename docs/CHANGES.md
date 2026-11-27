# 25 November 2026 — historical workflow update

The main entry point now uses the saved historical dataset (or downloads it when absent), runs both estimation windows, and shares defaults with the app. Production simulation generation, bundled simulated prices and the demo launchers have been retired. The 24 November audit below is historical, not the current workflow.

# Changes from the 21 November handoff

| Audit finding | Applied change | Verification |
|---|---|---|
| No included executable input/results | Bundled deterministic synthetic input, generator, ready report and one-command export | Default runner executed from another working directory; generated artifacts retained |
| No input-price view/export | Raw and cleaned previews/downloads plus preserved input snapshots | Byte/hash round-trip checks and full app-flow test |
| Covariance target differs from LN3 | Constant-correlation target with explicit configurable fixed delta | Target diagonal/off-diagonal identities and blend endpoints |
| ERC solver differs from supplied course formulation | Unconstrained ERC uses LN5 log constraint and rescaling; capped approximation explicitly labelled | Course example, risk residuals, scale invariance and low-correlation regression |
| Feasible solver failures silently accepted | Strict solver success and residual checks; candidate diagnostics and failure evidence | Injected failed optimizer result is rejected |
| Same-close information/execution ambiguity | One-observation close-to-close execution delay with initial cash | Independently reconstructed NAV and timing checks |
| Missing observations collapsed into daily returns | Reject incomplete interior rows and ambiguous dates | Missing/duplicate-input rejection tests |
| Original cleaning history lost | Preserve original bytes, original report, cleaned snapshot and separate hashes | Leading-history report survives import, simulation and export |
| Weak future-data test | Asset-specific time-varying future perturbation checked across earlier targets and NAV | Multi-signal independence regression |
| 2022 depends on 2020 availability | Check each observed year independently | Input without 2020 still produces 2022 table |
| Current target confused with held portfolio | Distinct target preview, executed target and drifted holdings with dates | Snapshot helper and default app-flow checks |
| Limited turnover treatment | Optional zero-default planned full-L1 turnover penalty; annualized rebalance turnover excludes initial purchase | Penalty arithmetic, feasibility and turnover accounting checks |
| Fragile environment/startup | Numerical dependencies pinned to tested versions; standard-library test runner; optional app/provider requirements | Standalone runner and regression suite |
| Failed rerun can leave stale apparent success | Preserve old generated outputs and write explicit current failure status/report | Success-then-failure regression |

## Deliberate limits

- The default input is synthetic and is labelled as such in the report, metadata and metrics. Actual historical ETF data are still absent from the provided source materials.
- Fixed delta is not an automatic Ledoit–Wolf coefficient. The coefficient formula was not established in the allowed material.
- The capped objective and optional turnover penalty are explicit project extensions built from the course quantities. Exact ERC/global optimality is not claimed for the capped objective.
- Actual Streamlit rendering and optional Yahoo connectivity were not tested because their dependencies were unavailable. The full default UI control flow was tested with a recording adapter. The offline HTML report's structure and local links were checked; an attempted browser render was blocked by the environment's socket permissions, so no browser layout verification is claimed.
- Cloud deployment and repository publication are not performed by this local repair. The blueprint's interactive app source is included and runnable after installing its optional dependency.

The exact original handoff is preserved separately in the local team package; it is not included in this code-only repository. Its old instructions and backlog describe the previous version; current startup and methods are in the root README and this docs directory.

## 26 November 2026 — client dashboard

- Client mandate, data coverage and dynamic caps promoted above raw data.
- Grouped capital/risk bars with equal-risk reference; actual holdings stacked on a 0–100% scale without normalizing cash away.
- Stable strategy colors, year axes, percentage drawdowns, mean subsequent-period risk tracking and 2020/2022 comparisons.
- Selected-strategy performance cards, shorter benchmark table and compact export section.
- Archive folder names replace machine-specific paths in provenance and regenerated reports.
- Windows test encoding correction retained; five extra tests. Mathematical engine unchanged.
- Missing saved CSV defaults to upload mode; no cloud dataset or redistribution rights assumed.
