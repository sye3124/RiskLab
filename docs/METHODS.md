# Methodology and implementation choices

The methodological sources are the supplied QARM II materials. The supplied blueprint defines project scope; it does not independently establish a new theoretical result. This document distinguishes the course formulas from modelling and software choices made to produce a runnable experiment.

## 1. Source map

| Element | Supplied basis | Implementation |
|---|---|---|
| Weighted portfolio return and quadratic variance | LN1, slides 8–9 | Daily simple returns with current holdings; `portfolio_risk` |
| Minimum variance and long-only/full-investment limits | LN1, slides 13–16 and 21 | Constrained minimum-variance reference |
| Structured/sample covariance combination | LN3, slides 9–10 | Constant-correlation target and explicit fixed delta |
| Turnover as sum of absolute weight changes | LN3, slides 21–22 | Full traded notional; one-way reporting convention separately labelled |
| Costs on bought/sold positions | LN3, slides 27–28; project blueprint | Simple proportional NAV haircut; an implementation approximation |
| Volatility risk contributions and Euler allocation | LN5, slides 4–8 | Marginal, absolute and percentage risk shares |
| Equal risk contributions | LN5, slide 15 | Equal risk across seven ETFs, not seven asset classes with aggregated budgets |
| Log-constraint ERC and rescaling | LN5, slides 23–28 and risk-budgeting lab | Unconstrained ERC reference |
| Capped relative-risk matching | Blueprint, page 2; direct use of LN5 risk shares | Explicit project approximation, not the course's exact log solver |

The seven ETFs, USD denomination, 35%/70%/25% limits, quarterly schedule, 504-day default calibration, 10-basis-point cost and fixed delta are project settings. They are not described as professor-imposed constants or empirically optimal values.

## 2. Covariance: exactly which shrinkage is implemented?

At a signal, use only the most recent configured number of return observations ending on that signal date. The sample estimator `S` uses denominator `T-1` (`ddof=1`). Let `sigma_i = sqrt(S_ii)` and compute sample correlations `rho_ij = S_ij/(sigma_i*sigma_j)`.

The LN3 structured target is:

\[
F_{ii}=S_{ii},\qquad F_{ij}=\bar\rho\sqrt{S_{ii}S_{jj}}\quad(i\ne j),
\]

where `rho_bar` is the mean of off-diagonal sample correlations. Then:

\[
\widehat\Sigma=\delta F+(1-\delta)S.
\]

This preserves the sample variances. It differs from the prior code's scaled-identity target. The project no longer depends on scikit-learn for covariance estimation.

**Delta defaults to 0.5 and is visible/configurable.** It is a fixed project assumption, not a statistically estimated optimum. LN3 refers to an appendix for automatic intensity calculation, but the needed implementation was not established from the supplied materials. No outside formula has been inserted. Delta=0 gives the sample covariance; delta=1 gives the constant-correlation target.

All three matrices are exported for each signal, so the blend can be reconstructed directly. Inputs with zero asset variance, nonfinite matrices, asymmetry or materially negative eigenvalues are rejected where relevant.

## 3. Risk contribution and portfolio optimization

With course weights `x` and covariance `Sigma`:

\[
\sigma(x)=\sqrt{x^\top\Sigma x},\quad
RC_i=x_i\frac{(\Sigma x)_i}{\sqrt{x^\top\Sigma x}},\quad
RC_i^\star=\frac{x_i(\Sigma x)_i}{x^\top\Sigma x}.
\]

The relative shares sum to one when the portfolio variance is positive. Individual contributions can be negative. Equal ETF risk targets each of the seven relative shares at `1/7`.

### Unconstrained ERC

The reference solver uses the LN5 construction:

\[
\min_y\frac12y^\top\Sigma y
\quad\text{subject to}\quad
\sum_i\ln y_i\ge c,\quad 0<y_i\le1,
\]

then rescales `x = y/sum(y)`. The arbitrary permitted scale `c` and numerical variable normalization are recorded in diagnostics. Normalizing objective units and changing variables do not change this mathematical problem. A second smaller scale is attempted only if the first fails the acceptance criteria.

Acceptance requires solver success, finite values, constraint residual at most `2e-7`, and maximum relative-risk-budget error at most `2e-5`. The last threshold is a numerical tolerance, not a theorem about returns. Nonuniform unconstrained risk budgets are explicitly unsupported in this version; the blueprint's custom-budget control was optional.

### Constrained risk-budget matching

Under caps, the code minimizes:

\[
\sum_i\left(RC_i^\star-\frac17\right)^2
\]

subject to full investment, long-only weights, and the ETF/group caps. This objective is directly constructed from the course risk shares but is a **project-specific approximation**. A nonzero residual can be unavoidable. SLSQP convergence is necessary but is not a guarantee of global optimality for this objective. Several distinct feasible starts are checked and retained; duplicate starts are removed.

### Minimum variance and optional turnover preference

The minimum-variance objective is `x' Sigma x` divided by the mean asset variance for numerical scaling. When the turnover penalty is zero, division by this positive constant does not change its minimizer.

For the two constrained strategies only, an optional nonnegative penalty adds:

\[
\lambda\sum_i|x_i-x_i^{\mathrm{signal}}|.
\]

Here the reference holdings are those already known at the signal close. The default is `lambda=0`. The absolute value is implemented using auxiliary variables constrained above both signed changes, an algebraic reformulation of the same expression. The objective's base units differ between minimum variance and risk-budget matching, so the same lambda is not a universal economic trade-off.

This is a preference for lower **planned** turnover, not a hard cap on executed turnover. Market drift before execution can change the actual amount traded. The initial all-cash entry has no prior invested weight reference and is not penalized by this term.

### Solver evidence

Every signal/strategy row records the formula used, signal/execution dates, calibration dates, iterations, objective and components, convergence, constraint residual, risk-budget error, planned turnover and candidate outcomes. Benchmarks are explicitly labelled fixed allocations, not numerical optimizations. Failed optimization raises an error with its diagnostics; feasible-but-failed results are no longer silently used.

## 4. Trading chronology

| Observation | Information and action | Return entitlement |
|---|---|---|
| Quarter-end close `t` | Estimate covariance through `t`; select the target | Existing holdings earned the interval ending at `t` |
| Next supplied close `t+1` | Execute that previously selected target; charge costs | Existing holdings first earn the interval ending at `t+1` |
| Following close `t+2` | Hold the new portfolio, with natural drift | New holdings earn the interval ending at `t+2` |

At the first entry there are no existing asset holdings: the model holds zero-return cash through `t+1`, pays its initial purchase cost at that close, and starts earning asset returns afterwards. No return is credited to new holdings before their execution.

Signals are identified at transitions between calendar quarters in the supplied observations. This requires the user to provide a suitable observation calendar. The program does not retrieve exchange holidays.

The chosen next-close execution is a project convention that makes information and trading chronology explicit. It still assumes execution at that observed close with the configured simple cost; it is not a claim of a full brokerage execution model.

## 5. Drift, turnover and costs

For an asset-return vector `r`, a portfolio with holdings `x` earns gross return `x' r`. Its pre-trade drifted weights are:

\[
x_i^{\mathrm{drift}}=\frac{x_i(1+r_i)}{1+x^\top r}.
\]

This follows directly from each position's value divided by the new total value. At a rebalance, full traded notional is:

\[
\tau=\sum_i|x_i^{\mathrm{target}}-x_i^{\mathrm{drift}}|.
\]

The project reports ordinary one-way turnover as `tau/2`, while the lecture turnover is the full sum. Initial entry from cash is separately reported as purchase fraction one. `annualized_rebalance_turnover` excludes that initial purchase; `total_one_way_turnover` includes it under the documented initial-entry convention.

With cost rate `q = basis_points/10000`, the model subtracts `q * tau * NAV_before_cost`. It then holds the target weights. This is the blueprint's simple proportional cost approximation; it is not an exact commission-funded trade-ticket model. It omits spread variation, impact and taxes. `cost_nav` uses units of initial wealth; multiplying by USD 10 million gives the reported dollar cost.

Caps constrain targets. They are not continuously enforced between trades. The two constrained strategies are eligible under the illustrated mandate at execution. Other strategies serve as references even when they exceed a cap.

## 6. Returns and reported statistics

Daily net portfolio return is computed from consecutive net NAV values, so trading costs enter once. Initial wealth is one; reported dollar values multiply by the blueprint's USD 10 million starting capital.

- CAGR is `ending_NAV ** (1/years) - 1`, where `years` is the calendar span from the initial signal close to the final evaluation close divided by 365.25 days. It includes the initial cash interval and initial cost.
- Annual volatility is the sample standard deviation of the observed daily net returns times `sqrt(252)`.
- Sharpe subtracts the configured constant daily risk-free assumption `(1+annual_rf)**(1/252)-1` from the mean daily return, divides by the daily standard deviation and multiplies by `sqrt(252)`.
- Maximum drawdown is the most negative NAV-to-previous-peak ratio minus one. The initial peak includes starting wealth one, so an initial cost contributes to drawdown.
- Year tables compound only the evaluation observations actually available in that year and show first/last dates. The years 2020 and 2022 are checked independently.

The weighted return, variance, Sharpe concept and compounding are supported by the supplied course. The exact reporting conventions above are explicit accounting/measurement choices from the project. They are not attributed to an unlocated course formula. No historical risk-free series is bundled.

## 7. Risk-budget evaluation

Ex ante risk shares use the fitted covariance available at the signal. Good fitting alone is not evidence of robust future diversification.

The subsequent holding-period diagnostic uses the sample covariance of returns **after execution through the next execution date, inclusive**. It applies that covariance to the old period's fixed target weights. Actual wealth and turnover use drifting holdings instead. This is a fixed-target risk diagnostic, not realized attribution of the evolving portfolio.

At least 20 observations are required for a reported period. The terminal incomplete period is identified by `partial_period`; means include the available reported periods. Compare complete periods separately when that is the desired empirical comparison. Future covariance is never returned to the historical optimizer.

## 8. Input policy and remaining limits

Original bytes and original cleaning evidence are retained. Leading pre-common history is documented and trimmed; explicit incomplete internal dates, duplicate dates and invalid values are rejected rather than silently repaired. Entire dates absent from every column cannot be independently detected without an authorized calendar. Long calendar intervals are recorded for review.

Historical runs remain labelled unverified until source authenticity and adjustment are reviewed. The retrospectively selected ETF universe, execution/cost approximation, constant risk-free assumption and lack of taxes/FX hedging remain scope limitations. No strategy superiority is assumed.
