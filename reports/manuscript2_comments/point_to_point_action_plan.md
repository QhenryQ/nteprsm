# Point-to-Point Action Plan for AJ-2026-01-0038-OA

This document is the consolidated execution plan for the revision. It preserves the useful strategy, evidence triage, and analysis planning from the earlier planning notes in one place.

## Working objective

Revise the paper into a statistically defensible single-site methods paper unless genuinely broader validation can be added before the resubmission deadline.

## Decision rule

Do not finalize the manuscript rewrite until the minimum validation package is completed.

Minimum validation package:

1. one real-data held-out predictive analysis on the NJ2 dataset
2. one harder-than-self-generated validation element, preferably posterior predictive checks for ordinal calibration and category use
3. one whole-dataset summary artifact across all entries
4. one corrected model-comparison table that reports differences and uncertainty rather than only raw scores

## Priority order

1. lock manuscript scope and claim level
2. produce the minimum validation package
3. repair model-comparison language and table structure
4. add whole-dataset results and operational interpretation
5. rewrite framing, methods, results, discussion, and limitations
6. clean figures, captions, terminology, and response-letter mechanics

## Point-to-point action plan

| ID | Priority | Issue to address | Cross-validation verdict | Concrete action | Evidence or analysis required | Planned manuscript output |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Critical | Abstract and discussion overclaim robustness, scalability, and future multi-location readiness | Strong reviewer point | Rewrite title, abstract, plain-language summary, discussion, and conclusion as a single-site methods contribution | no new analysis required, but final wording must match added validation results | revised title, abstract, conclusion, and scope statements |
| A2 | Critical | Validation is too dependent on same-family parameter recovery | Strong reviewer point | Add one real-data held-out predictive analysis and one posterior predictive validation block | held-out prediction plus PPC outputs | new validation subsection in Results and Discussion |
| A3 | Critical | Model comparison is confounded because multiple changes were introduced together | Strong reviewer point | Rebuild comparison language and, if feasible, run limited ablation ladder | ELPD differences with uncertainty; optional intermediate model fits | revised comparison table and causal interpretation text |
| A4 | Critical | Four-entry illustration is too selective for the practical claims | Strong reviewer point | Add a whole-dataset seasonality summary across all 89 entries | derived summary metrics for each entry | one new figure plus one compact summary table |
| A5 | High | Identification story is implicit rather than explicit | Strong but fixable by exposition | Add a compact identification paragraph explaining latent-scale anchoring, fixed discrimination, and threshold centering | no new analysis required unless sensitivity is added | revised Methods subsection |
| A6 | High | Fixed discrimination is weakly justified as written | Strong reviewer point, but not fatal | Reframe as a modeling choice for identifiability and tractability; add sensitivity only if feasible | optional sensitivity text or exploratory variant | revised Methods and Discussion text |
| A7 | High | Threshold irregularity for Rater E may indicate instability or genuine behavior | Justified but needs careful framing | Discuss as an empirical pattern under the current model and not over-interpret it | optional threshold summary or sensitivity note | revised Results and Discussion |
| A8 | High | Climate/biology versus rater-driven temporal variation is sometimes blurred | Justified reviewer point | Clarify that the estimated temporal component is a latent model-based seasonality term after adjustment, not direct proof of causal climate effects | no new analysis required | tightened Discussion language |
| A9 | High | Operational interpretation of the latent scale is still mostly conceptual | Strong reviewer point | Add one concrete translation from latent-scale results to an average-rater or standardized interpretation | summary mapping or worked example using real entries | expanded Results interpretation section |
| A10 | Medium | Computational gain is discussed more clearly than fit-complexity tradeoff | Justified reviewer point | Add explicit discussion of `p_loo`, uncertainty, and practical significance | revised comparison statistics | revised Results and Discussion paragraphs |
| A11 | Medium | Limitations are back-loaded | Strong reviewer point | Move key limitations earlier and distinguish present evidence from future work | no new analysis required | revised Introduction ending and Limitations section |
| A12 | Medium | Prior terminology and writing are inconsistent | Clear editorial issue | Reconcile weakly informative versus empirically informed language and line edit the paper | no new analysis required | copyedited Methods and Discussion |
| A13 | Medium | Figure axes, terminology, and typographic issues reduce credibility | Clear editorial issue | Correct labels, captions, and terminology after substantive edits are done; known typos: "Radical Basis Function" → "Radial Basis Function", "Seaonality" → "Seasonality", "Catogory" → "Category", "a entry" → "an entry" (line 358) | no new analysis required | corrected figures and final copyedit |

## Concrete analysis plan

### Minimum viable analysis package

This is the package to complete first. It is the shortest route to a defensible revision.

#### Analysis 1. Real-data held-out prediction on NJ2

Purpose:

- answer the central criticism that current validation mostly shows self-consistency under favorable assumptions
- support a narrower but stronger claim that the model improves predictive performance on real ordinal ratings

Recommended split:

- use blocked hold-out by rating event or rating date, not random row-wise splitting
- preferred primary split: leave out 20 percent of rating events across the season while preserving all entries and raters in training
- preferred sensitivity split: leave out one full rating event at a time if compute is manageable

Why this split:

- random row splits leak temporal structure
- event-blocked hold-out tests whether the temporal component generalizes to unseen field-rating occasions

Metrics to report:

- held-out log predictive density
- ranked probability score for ordinal predictions
- exact-category accuracy
- adjacent-category accuracy
- calibration of predicted versus observed category frequencies

Comparison target:

- compare the current seasonality model against the prior spatial model if both can be scored on the same hold-out partitions
- if that is infeasible, report held-out performance for the revised bundled model alone and avoid comparative causal claims

Planned outputs:

- Figure V1: observed versus predicted category distribution on held-out data, overall and by rating event
- Table V1: held-out predictive metrics for each model and each split

Implementation note:

- the existing LOO workflow in `notebooks/manuscript2_visualizations_nj2.ipynb` computes PSIS-LOO on the full dataset via `az.loo()`; it does **not** support event-blocked hold-out prediction
- PSIS-LOO is **not** true held-out cross-validation — it approximates LOO via importance sampling without ever refitting the model. The reviewer is asking for a genuine refit-and-predict workflow.
- **new infrastructure required before this analysis can run:**
  1. add train/test split logic to `DataHandler` in `nteprsm/utils.py` that can mask observations by rating event (see task I2)
  2. either (a) modify `annual_seasonality_model.stan` to accept a held-out index array and compute out-of-sample log-lik in `generated quantities`, or (b) fit on the training subset and score held-out observations in Python using `rsm_probability()` from `nteprsm/utils.py`
  3. option (a) is cleaner for ELPD-style scoring; option (b) avoids touching Stan code but requires a Python scoring loop
- Henry must decide which approach before Olena begins implementation
- once the split infrastructure exists, compute held-out log likelihoods in the same format used for current ELPD summaries so the reporting stays consistent

**Detailed implementation steps for Olena**:

**Step 1. Choose held-out events** (Henry decides, Olena implements):

- NJ2 has 35 rating events. Hold out 7 events (~20%), spread across the season.
- Sort events by `adj_time_of_year` and pick every 5th event so the held-out set covers all seasons.
- Example: events at indices 5, 10, 15, 20, 25, 30, 35 (1-indexed).

**Step 2. Generate training data**:

```python
from nteprsm.utils import DataHandler

dh = DataHandler(filepath="path/to/quality.csv")
dh.load_data()
dh.preprocess_data()

held_out = [5, 10, 15, 20, 25, 30, 35]  # Henry confirms these
train_data, test_data = dh.split_by_rating_event(held_out, M_f=8, pred_N=100, padding=5)
```

**Step 3. Fit the seasonality model on training data only**:

- Use the existing Slurm infrastructure. Copy `submit_job_array_seasonality.sh` to a new script `submit_holdout_seasonality.sh`.
- Modify to pass `train_data` instead of the full dataset.
- The model file is unchanged: `models/annual_seasonality_model.stan`.
- Save the fit output to a new directory: `model_runs/holdout_seasonality_nj2/`.

**Step 4. Score held-out observations** (two options):

*Option A — Stan-side scoring (preferred):*

Create `models/annual_seasonality_model_holdout.stan` by copying the original and adding to `data {}`:

```stan
int<lower=0> N_test;
array[N_test] int<lower=0, upper=y_max> y_test;
array[N_test] int<lower=1, upper=num_plots> plot_code_test;
array[N_test] int<lower=1, upper=num_entries> entry_code_test;
array[N_test] int<lower=1, upper=num_raters> rater_code_test;
array[N_test] real time_test;  // adj_time_of_year for each test obs
```

And adding to `generated quantities`:

```stan
vector[N_test] log_lik_test;
array[N_test] int y_rep_test;
for (n in 1:N_test) {
  // compute time_effect at the test observation's time point
  // (requires evaluating the Hilbert basis at the test time, not at a training event index)
  vector[2 * M_f] phi_test_n = PHI_periodic_single(M_f, 2 * pi() / period, (time_test[n] - mean_time) / sd_time);
  real te_n = intercept_f[entry_code_test[n]] + dot_product(diagSPD_f .* beta_f[entry_code_test[n]], phi_test_n);
  real theta_test_n = plot_effect[plot_code_test[n]] + te_n;

  vector[y_max + 1] unsummed_test = append_row(rep_vector(0, 1), theta_test_n - tau_rater[rater_code_test[n]]);
  vector[y_max + 1] probs_test = softmax(cumulative_sum(unsummed_test));
  log_lik_test[n] = categorical_lpmf(y_test[n] + 1 | probs_test);
  y_rep_test[n] = categorical_rng(probs_test) - 1;
}
```

Note: you'll need a helper function `PHI_periodic_single` that evaluates the basis for a single time point, or precompute the test-time basis matrix in `transformed data` by passing `time_test` as data.

*Option B — Python-side scoring (simpler, no Stan changes):*

```python
import numpy as np
from scipy.special import softmax
from cmdstanpy import CmdStanMCMC

# load training fit
fit = ...  # CmdStanMCMC from the training-only run

# extract posterior draws
plot_effect = fit.stan_variable("plot_effect")   # (draws, num_plots)
intercept_f = fit.stan_variable("intercept_f")   # (draws, num_entries)
beta_f = fit.stan_variable("beta_f")             # (draws, num_entries, 2*M_f)
tau_rater = fit.stan_variable("tau_rater")        # (draws, num_raters, y_max)
sigma_f = fit.stan_variable("sigma_f")            # (draws,)
lengthscale_f = fit.stan_variable("lengthscale_f")  # (draws,)

# reconstruct PHI basis for test time points
# (copy the PHI_periodic and diagSPD_periodic logic from the Stan model)

# for each test observation, for each draw, compute:
#   theta = plot_effect[draw, plot] + intercept_f[draw, entry] + PHI @ (diagSPD .* beta_f[draw, entry])
#   probs = softmax(cumsum([0, theta - tau_rater[draw, rater, :]]))
#   log_score = log(probs[y_test])
#   y_rep = np.random.choice(y_max+1, p=probs)

# this loop over (draws × N_test) is slow but correct
# vectorize the inner loop over N_test if possible
```

**Step 5. Compute metrics from held-out scores**:

```python
# held_out_log_scores: shape (draws, N_test)

# 1. held-out log predictive density (sum over observations, mean over draws)
elpd_holdout = np.mean(np.sum(held_out_log_scores, axis=1))

# 2. ranked probability score (requires full probability vectors, not just log_lik)
#    RPS = sum over k of (cumulative_predicted_prob_k - cumulative_observed_k)^2
#    compute per observation per draw, then average

# 3. exact-category accuracy
#    y_pred = argmax of mean predicted probabilities
#    accuracy = mean(y_pred == y_test)

# 4. adjacent-category accuracy
#    adj_accuracy = mean(abs(y_pred - y_test) <= 1)

# 5. calibration
#    for each category 0-8: predicted frequency vs observed frequency across test set
```

**Step 6. Repeat for spatial model** (if Henry approves comparative analysis):

- Fit `models/spatial_model.stan` on the same training split.
- Score the same held-out events.
- Report both models' held-out metrics side-by-side in Table V1.
- Note: the spatial model uses different codes (`beta[rating_event_code]` with per-event severity, not rater-specific thresholds), so the train/test split must also re-index `rating_event_code` consistently.

**Deliverable**: Table V1 (CSV) + Figure V1 (observed vs. predicted category distributions on held-out events).

#### Analysis 2. Posterior predictive checks for ordinal calibration and category use

Purpose:

- provide a harder validation layer without requiring a new external dataset
- show whether the model reproduces category-use behavior and ordinal calibration patterns actually seen in the field data

Checks to run:

1. overall category frequencies: observed counts for categories 1 through 9 versus posterior predictive intervals
2. category frequencies by rater: observed versus posterior predictive intervals for each rater
3. category frequencies by rating event: observed versus posterior predictive intervals for each event date
4. tail behavior: check under- or over-production of the highest and lowest rating categories

Planned outputs:

- Figure V2: posterior predictive check panels for category use overall, by rater, and by event
- short Results paragraph stating where the model reproduces ordinal behavior well and where it misses

Interpretation rule:

- do not present PPCs as proof of robustness
- present them as evidence that the model captures key distributional features of ordinal ratings better than a purely internal recovery study alone

Implementation note:

- the current Stan model produces `pred_time_effect` (latent-scale predictions), **not** simulated ordinal responses (`y_rep`); PPCs for category frequencies require `y_rep`
- **new infrastructure required:**
  1. preferred path: add a `y_rep` sampling block to `annual_seasonality_model.stan` generated quantities, then use `CmdStanModel.generate_quantities()` on existing posterior draws — this avoids refitting the model
  2. fallback path: sample `y_rep` in Python using posterior draws of `theta` and `tau_rater` combined with `rsm_probability()` from `nteprsm/utils.py`
- the `generate_quantities()` approach is recommended because it keeps the sampling within Stan's categorical distribution and avoids Python-side categorical sampling edge cases

**Detailed implementation steps for Olena**:

**Step 1. Create `models/annual_seasonality_model_ppc.stan`** (see task I3 for the full file):

- Copy the entire `annual_seasonality_model.stan`.
- Replace the `model {}` block with an empty block.
- Add `y_rep` sampling to `generated quantities` (see task I3 for exact Stan code).

**Step 2. Run `generate_quantities()` on existing NJ2 fit**:

```python
from cmdstanpy import CmdStanModel
import pickle
import numpy as np

# load the existing full-dataset fit (NOT a refit)
with open("fit_seasonality_nj2_quality.pkl", "rb") as f:
    fit = pickle.load(f)

# compile the PPC model
from gptools.stan import get_include
ppc_model = CmdStanModel(
    stan_file="models/annual_seasonality_model_ppc.stan",
    stanc_options={"include-paths": [get_include()]}
)

# generate y_rep from existing posterior draws
gq = ppc_model.generate_quantities(
    data=stan_data,          # exact same stan_data used for the original fit
    previous_fit=fit,
    seed=42
)

# extract y_rep: shape (num_draws, N)
y_rep = gq.stan_variable("y_rep")
np.save("data/model_output/nj2_y_rep.npy", y_rep)
```

**Step 3. Compute PPC summaries**:

```python
import numpy as np
import pandas as pd

y_obs = stan_data["y"]  # shape (N,), 0-indexed
rater_codes = stan_data["rater_code"]  # shape (N,), 1-indexed
event_codes = stan_data["rating_event_code"]  # shape (N,), 1-indexed
num_categories = stan_data["y_max"] + 1  # 9

# --- Overall category frequencies ---
obs_counts = np.bincount(y_obs, minlength=num_categories)
# for each draw, count category frequencies
rep_counts = np.array([np.bincount(y_rep[d], minlength=num_categories) for d in range(y_rep.shape[0])])
# 90% interval
lower = np.percentile(rep_counts, 5, axis=0)
upper = np.percentile(rep_counts, 95, axis=0)
median = np.median(rep_counts, axis=0)

# --- By rater ---
for r in range(1, stan_data["num_raters"] + 1):
    mask = rater_codes == r
    obs_r = np.bincount(y_obs[mask], minlength=num_categories)
    rep_r = np.array([np.bincount(y_rep[d, mask], minlength=num_categories) for d in range(y_rep.shape[0])])
    # store lower/median/upper for rater r

# --- By rating event ---
for e in range(1, stan_data["num_ratings"] + 1):
    mask = event_codes == e
    obs_e = np.bincount(y_obs[mask], minlength=num_categories)
    rep_e = np.array([np.bincount(y_rep[d, mask], minlength=num_categories) for d in range(y_rep.shape[0])])
    # store lower/median/upper for event e

# --- Tail behavior ---
# proportion of extreme categories (0, 1 at bottom; 7, 8 at top)
obs_low = np.sum(y_obs <= 1) / len(y_obs)
obs_high = np.sum(y_obs >= 7) / len(y_obs)
rep_low = np.mean(y_rep <= 1, axis=1)   # proportion per draw
rep_high = np.mean(y_rep >= 7, axis=1)
# compare obs_low to [5th, 95th percentile of rep_low]
```

**Step 4. Generate Figure V2**:

- Panel 1: bar chart of observed category counts with shaded 90% posterior predictive intervals, overall.
- Panel 2: same but faceted by rater (7 panels for raters A–G).
- Panel 3: same but faceted by rating event (35 panels or a heatmap-style summary).
- Panel 4: tail-behavior summary — observed vs. posterior predictive proportion for categories 0–1 and 7–8.

Use plotly or matplotlib — match the existing notebook style.

**Deliverable**: Figure V2 (multi-panel PPC plot) + `data/processed/ppc_summary.csv` with the numeric summaries + a one-paragraph write-up of where the model fits well and where it misses.

#### Analysis 3. Whole-dataset seasonality summary across all entries

Purpose:

- replace the current four-entry story as the main empirical summary
- show that the framework yields systematic outputs across the full trial rather than only illustrative examples

Summary statistics per entry:

- posterior mean seasonal amplitude
- day-of-year of posterior mean peak
- day-of-year of posterior mean minimum
- peak-to-trough spread with uncertainty
- posterior uncertainty in peak timing

Planned outputs:

- Figure R1: heatmap or dot-and-interval plot across all 89 entries showing peak timing and seasonal amplitude
- Table R1: compact top and bottom entries by seasonal amplitude or peak quality, with uncertainty intervals

Recommendation on format:

- use the figure for all 89 entries
- keep the table short and decision-oriented so the manuscript remains readable

Interpretation rule:

- keep the current four-entry curves only as a secondary illustration after the whole-dataset summary appears

Implementation note:

- use the same posterior seasonality summaries already visualized for selected entries in `notebooks/manuscript2_visualizations_nj2.ipynb`

#### Analysis 4. Corrected model comparison summary

Purpose:

- fix the current presentation that treats less negative ELPD as sufficient proof of superiority
- separate statistical evidence from interpretive overreach

Minimum required reporting:

- `elpd_loo`
- standard error of `elpd_loo`
- `p_loo`
- `elpd_diff` relative to baseline
- standard error of `elpd_diff`
- brief interpretation of whether the difference is large relative to its uncertainty

Planned outputs:

- Table M1: revised model comparison table with raw scores, differences, uncertainty, and complexity

Implementation note:

- the notebook already contains raw LOO outputs for the seasonality and spatial models, so the immediate task is to reframe and extend the reporting rather than invent a new workflow

### Stretch analysis package

Only do these if the minimum package is stable and compute time remains.

#### Analysis 5. Limited ablation ladder

Purpose:

- answer the reviewer request for deconfounding the contribution of seasonality from other model changes

Preferred ladder:

1. prior spatial model baseline
2. spatial model plus revised rater-threshold structure
3. spatial model plus seasonality component
4. full revised model with approximations

Fallback if only one additional fit is feasible:

- fit either the threshold-revised-no-seasonality variant or the seasonality-with-old-threshold variant, whichever is fastest to implement from current code

Planned outputs:

- Table M2: ablation comparison with `elpd_diff`, uncertainty, runtime, and `p_loo`

Implementation cost warning:

- the two existing Stan models (`spatial_model.stan` and `annual_seasonality_model.stan`) differ in fundamental ways: global vs. rater-specific thresholds, per-event severity vs. none, Cholesky GP vs. Hilbert+RFFT2 GPs
- creating intermediate variants requires **writing new Stan models**, not reconfiguring existing ones
- budget 1–2 days of Stan development per intermediate model variant
- this cost means the ablation decision should be made upfront, not treated as a last-minute stretch goal

Decision rule:

- if these models are not feasible in time, do not force them
- instead rewrite the manuscript to say that the comparison evaluates the bundled revised model rather than isolating seasonality alone
- **make this decision in Week 0 so the manuscript framing is not left ambiguous**

#### Analysis 6. Misspecification or reduced-information stress test

Purpose:

- provide a harder validation element if PPCs alone look too soft in the response letter

Best candidate stress tests given current repo assets:

1. simulate drifting rater thresholds over time, then fit the current static-threshold model
2. downsample rating events to create irregular seasonal coverage
3. reduce the number of raters or remove some rater identities

Primary summary metrics:

- bias in seasonal amplitude recovery
- bias in peak-timing recovery
- widening of uncertainty under reduced information

Planned outputs:

- Figure V3: bias or error under one misspecification scenario
- Table V2: recovery or predictive degradation under reduced information

Implementation note:

- adapt the existing parameter-recovery notebooks instead of building a new pipeline from scratch

## Figure and table plan

### New figures to add

| ID | Figure concept | Purpose | Minimum content |
| --- | --- | --- | --- |
| V1 | Held-out predictive performance | show real-data predictive validity | predicted versus observed category use on held-out events; optionally by event |
| V2 | Posterior predictive checks for ordinal calibration | show whether category use and ordinal behavior are reproduced | overall, by rater, and by event PPC panels |
| R1 | Whole-dataset seasonality summary across 89 entries | replace selective illustration as the main results artifact | amplitude and peak timing with uncertainty for every entry |
| I1 | Latent-scale interpretation example | make interpretation operational | one mapping from latent scale to average-rater or standardized expected rating |

### Optional stretch figures

| ID | Figure concept | Purpose |
| --- | --- | --- |
| V3 | Misspecification stress-test result | show behavior under broken assumptions |
| M2-Fig | Ablation comparison visual | visualize fit-complexity-runtime tradeoff |

### New tables to add

| ID | Table concept | Purpose | Minimum content |
| --- | --- | --- | --- |
| V1 | Held-out predictive metrics | support strengthened validation | log score, ranked probability score, exact and adjacent accuracy |
| M1 | Corrected model comparison | fix current comparison language | `elpd_loo`, SE, `elpd_diff`, diff SE, `p_loo`, runtime |
| R1 | Compact seasonality ranking summary | support decision relevance | selected entries with amplitude, peak timing, uncertainty |

### Optional stretch tables

| ID | Table concept | Purpose |
| --- | --- | --- |
| M2 | Ablation ladder comparison | isolate where predictive gains are coming from |
| V2 | Stress-test degradation summary | quantify robustness under misspecification |

## Manuscript rewrite plan tied to the analyses

### Title, abstract, and plain-language summary

Revise after Analyses 1 through 4 are complete.

Required changes:

- remove unsupported claims of robustness and scalability unless directly supported by new evidence
- state clearly that the paper demonstrates a Bayesian latent-scale seasonality model on one site-year dataset
- describe validation honestly: held-out prediction, posterior predictive checks, and parameter recovery if retained

### Methods

Required additions:

- one explicit identification paragraph explaining the role of fixed discrimination, threshold centering, and latent-scale anchoring
- one paragraph clarifying why discrimination was fixed and what is deferred to future work
- one short paragraph describing the held-out design and PPC workflow

### Results

Required restructuring:

1. model comparison with corrected uncertainty language
2. strengthened validation block
3. whole-dataset seasonality summary
4. illustrative entry-level curves as supporting detail
5. latent-scale interpretation example

Additional point-edits from Reviewer 2 (currently only in the tracker, must also be addressed in prose):

- R2-3: relocate the paragraph at lines 67–78 from Introduction to Methods
- R2-5: add rationale for the synthetic data parameter choices (lengthscale, sigma, grid size) in the parameter recovery subsection
- R2-6: remove or replace the word "large" when describing the synthetic dataset
- R2-9: explicitly address the SE increase in the new model (72.22 vs. 68.96) in the model comparison discussion; Table M1 should include a note or row addressing this

### Discussion and limitations

Required changes:

- distinguish what is demonstrated from what remains future work
- clarify that latent seasonality is inferred under model assumptions rather than directly observed climate effect
- discuss static thresholds, static spatial structure, event coverage dependence, and known-rater requirements as real scope limits

## Response-letter positioning

Use the following response posture.

### Concede directly

- overclaiming relative to single-site evidence
- need for stronger validation
- current model-comparison language overstates what the comparison isolates
- need for a whole-dataset summary
- writing, figure-label, and terminology problems

### Concede with clarification

- identifiability concern: explain that the paper already contains identification structure, but it was presented too diffusely
- fixed discrimination concern: acknowledge weak justification and reframe it as a deliberate modeling simplification
- climate versus rater-behavior concern: clarify interpretation instead of implying the temporal term is purely biological

### Do not over-concede

- do not say that a single-site methods paper is invalid
- do not claim the revised paper establishes cross-location generalizability unless new evidence actually does so
- do not imply that ELPD differences identify the causal contribution of seasonality if ablations are not run

## Henry and Olena work split

Working assumption:

- Henry owns scientific direction, manuscript decisions, reviewer-response strategy, and final editorial control
- Olena owns Slurm-based execution at UMN, reproducible run management, intermediate outputs, and first-pass analysis artifacts
- major modeling or scope decisions should not be made ad hoc during cluster work; Henry should approve them first

### Henry ownership

- decide the revision scope: strictly single-site methods paper versus broader claim set
- choose the minimum validation package and decide whether stretch analyses are worth the time
- define acceptance criteria for each analysis before Olena runs it
- write or heavily edit the title, abstract, introduction, methods justification, discussion, limitations, and conclusion
- own the responses to reviewers on identifiability, fixed discrimination, claim calibration, and interpretation
- review Olena's outputs and decide which figures and tables are manuscript-ready
- do the final prose pass for consistency, tone, and journal-facing polish
- own the final submission package, including tracked changes and point-by-point responses

### Olena ownership

- prepare and launch Slurm jobs on the UMN cluster
- keep run logs, output paths, configuration versions, and runtime summaries organized
- implement the agreed validation analyses once Henry signs off on the design
- generate first-pass figures and tables for held-out prediction, posterior predictive checks, whole-dataset summaries, and model-comparison outputs
- report failures, instability, or unexpected results quickly rather than trying to reinterpret them alone
- if stretch analyses are approved, run ablation or misspecification experiments and summarize what completed successfully
- maintain a short run memo after each work session: what ran, what failed, what files were produced, and what decisions are needed from Henry

### Shared responsibilities with clear lead

| Work item | Lead | Henry role | Olena role |
| --- | --- | --- | --- |
| Lock revision scope | Henry | decide claim level and minimum evidence bar | provide feasibility input from code and cluster constraints |
| Held-out validation design | Henry | specify split design, metrics, and success criteria | implement and run the approved design |
| Posterior predictive checks | Olena | approve which PPCs matter for the response letter | generate PPC outputs and summarize fit/misfit |
| Whole-dataset seasonality summary | Olena | define which entry-level metrics are decision-relevant | produce the summary dataset, figure, and draft table |
| Model comparison table | Olena | decide how conservative the interpretation should be | compute comparison statistics and draft table |
| Ablation or stress-test decision | Henry | decide whether extra runs are worth the time | estimate runtime/cost and execute if approved |
| Methods justification text | Henry | write identification and modeling rationale | supply technical notes from implementation if needed |
| Results figure selection | Henry | choose which outputs make the strongest paper | provide candidate figures and variants |
| Response letter draft | Henry | write the substantive responses | provide exact run outcomes, file names, and result summaries |
| Final manuscript cleanup | Henry | final wording and consistency pass | support with updated figure exports and table values |

### Working cadence for Henry and Olena

#### Before each analysis cycle

Henry should provide:

- the exact question to answer that week
- the model/configuration to run
- the output metrics needed
- a stop rule, such as "do not start ablation until held-out prediction is stable"

This keeps Olena's execution time focused on the highest-value computational work rather than open-ended exploration.

#### During each execution cycle

Olena should prioritize:

1. launching the highest-priority validated run set first
2. checking for failures early so time is not lost to a broken job array
3. producing one concise end-of-day summary with links or paths to outputs

#### After each execution cycle

Henry should:

1. review outputs and decide which results are worth keeping
2. decide the next week's run plan
3. convert usable outputs into manuscript text, figure captions, or reviewer-response language

### Recommended task split by phase

#### Phase 1. Scope and analysis design

- Henry leads
- Olena advises on feasibility and expected runtime on the UMN Slurm cluster

#### Phase 2. Computational execution

- Olena leads
- Henry only intervenes for decision points, failed assumptions, or interpretation questions

#### Phase 3. Manuscript rewriting

- Henry leads
- Olena supplies exact numbers, figure exports, and methodological clarifications as needed

#### Phase 4. Final response package

- Henry leads
- Olena verifies that all reported numbers match the latest outputs

### Practical rule for avoiding role drift

- if the task is "decide what claim the paper can support," it belongs to Henry
- if the task is "run, rerun, export, summarize, or troubleshoot code on Slurm," it belongs to Olena
- if the task is "interpret an ambiguous output and decide whether it changes the paper's story," Henry decides after reviewing Olena's summary

### Immediate recommended assignment

#### Henry should do next

1. finalize the exact held-out design and success metrics for Analysis 1
2. decide whether PPCs alone are enough as the second validation layer or whether a stress test is also required
3. define the entry-level summary metrics that will appear in Figure R1 and Table R1
4. draft the revised claim boundaries for abstract and conclusion so analysis targets are clear

#### Olena should do next

1. audit the current NJ2 notebook and run scripts to identify the fastest path for held-out prediction on the UMN cluster
2. **build the pre-execution infrastructure** (see new section below) before launching any analysis jobs
3. prepare the job scripts and output structure for Analysis 1 and Analysis 2
4. verify how current LOO and posterior outputs can be reused for Table M1 and Figure V2
5. return a short feasibility note with expected runtime, dependencies, and any blockers

### Pre-execution infrastructure tasks (must complete before Analyses 1–2 can run)

These tasks address implementation gaps that the current codebase does not cover. Each task includes the exact files to edit, what code to write, and what the deliverable is.

#### I1. Data access and path verification

**Status**: not an issue. Data is stored in a shared Google Drive folder accessible to both Henry and Olena. The NJ2 notebook's external path (`DATA_DIR / 'data_ingestion/parsed_data/kb2017/nj2/quality.csv'`) and the pickled fit objects (`fit_seasonality_nj2_quality.pkl`, `fit_spatial_nj2_quality.pkl`) are available there.

**Only action needed**: Olena should confirm the fit pickle files can be reloaded on the UMN cluster without path errors (CmdStanMCMC pickles store internal references to Stan CSV files; if those CSVs were moved, the pickle may fail to load). If this happens, re-extract the fit from the CSV files directly using `CmdStanMCMC(csv_files=[...])`.

**Deliverable**: a one-line confirmation that the pickles load cleanly, or a note if re-extraction from CSVs is needed.

#### I2. Train/test split infrastructure for held-out prediction

**Problem**: `DataHandler.generate_stan_data()` in `nteprsm/utils.py` (line ~278) puts every observation into one `stan_data` dict. There is no way to split by rating event.

**File to edit**: `nteprsm/utils.py`, inside the `DataHandler` class, after the `generate_stan_data()` method (after line ~327).

**Add this method** (adapt as needed):

```python
def split_by_rating_event(self, held_out_events: list[int], **kwargs) -> tuple[dict, dict]:
    """
    Split model_data into train and test stan_data dicts by rating_event_code.

    Parameters
    ----------
    held_out_events : list[int]
        rating_event_code values to hold out (1-indexed, matching Stan codes).
    **kwargs : passed to generate_stan_data() for additional fields like M_f, pred_N, padding.

    Returns
    -------
    train_data : dict  — stan_data dict for training observations only
    test_data  : dict  — stan_data dict for held-out observations (keeps original codes for mapping)
    """
    if self.model_data is None:
        raise ValueError("Call preprocess_data() first.")

    full_df = self.model_data
    train_mask = ~full_df["rating_event_code"].isin(held_out_events)
    test_mask = full_df["rating_event_code"].isin(held_out_events)

    train_df = full_df[train_mask].copy()
    test_df = full_df[test_mask].copy()

    # Re-index rating_event_code for training set so codes are contiguous 1..K_train
    # (Stan requires contiguous integer codes)
    old_to_new = {old: new for new, old in enumerate(sorted(train_df["rating_event_code"].unique()), 1)}
    train_df["rating_event_code"] = train_df["rating_event_code"].map(old_to_new)

    # Rebuild time array for training events only
    train_time = (
        train_df[["rating_event_code", "adj_time_of_year"]]
        .drop_duplicates()
        .sort_values("rating_event_code")
        .set_index("rating_event_code")
        .values.reshape(-1)
    )

    plot_data = full_df.groupby("plot_code")[["row", "col"]].mean()

    train_data = {
        "y": train_df[self.target].values,
        "N": len(train_df),
        "num_ratings": int(train_df["rating_event_code"].max()),
        "num_raters": int(train_df["rater_code"].max()),
        "num_entries": int(train_df["entry_code"].max()),
        "num_plots": int(train_df["plot_code"].max()),
        "y_max": int(train_df[self.target].max()),
        "rating_event_code": train_df["rating_event_code"].values,
        "entry_code": train_df["entry_code"].values,
        "plot_code": train_df["plot_code"].values,
        "rater_code": train_df["rater_code"].values,
        "DIST": self._calculate_distance_matrix(),
        "num_rows": int(plot_data["row"].max()),
        "num_cols": int(plot_data["col"].max()),
        "plot_row": plot_data["row"].astype(int).values,
        "plot_col": plot_data["col"].astype(int).values,
        "time": train_time,
    }
    train_data.update(kwargs)

    # Test data keeps original codes so we can map back to calendar dates
    test_data = {
        "y": test_df[self.target].values,
        "N_test": len(test_df),
        "entry_code": test_df["entry_code"].values,
        "plot_code": test_df["plot_code"].values,
        "rater_code": test_df["rater_code"].values,
        "rating_event_code": test_df["rating_event_code"].values,  # original codes
        "adj_time_of_year": test_df["adj_time_of_year"].values,
    }

    return train_data, test_data
```

**Key considerations**:
- `entry_code`, `plot_code`, and `rater_code` must stay on the same coding as the full dataset (all entries/plots/raters still exist in training, just some events are removed)
- `rating_event_code` must be re-indexed for the training dict because Stan iterates `1:num_ratings`
- `time` array length must equal the new `num_ratings`

**Test**: call `split_by_rating_event([5, 10, 15, 20, 25, 30, 35])` on the NJ2 data and verify:
- `train_data["N"] + test_data["N_test"] == original N (9612)`
- `train_data["num_ratings"] == 35 - 7 == 28`
- all rater codes and entry codes still present in training

**Deliverable**: working method in `utils.py` + a short script or notebook cell that runs the test above and prints the verification.

#### I3. Create `annual_seasonality_model_ppc.stan` for `y_rep` generation

**Problem**: the current `generated quantities` block in `models/annual_seasonality_model.stan` (line 165) produces `pred_time_effect` and `log_lik` but does **not** sample `y_rep`. Posterior predictive checks for category frequencies need simulated ordinal responses.

**Approach**: create a new Stan file that only contains a `generated quantities` block, to use with `CmdStanModel.generate_quantities()` on existing posterior draws. This avoids refitting the model.

**File to create**: `models/annual_seasonality_model_ppc.stan`

**Contents** (copy the full `functions`, `data`, `transformed data`, `parameters`, and `transformed parameters` blocks from `annual_seasonality_model.stan` unchanged, then replace the `model` block with an empty one and replace `generated quantities` with):

```stan
model {
  // intentionally empty — this file is used only with generate_quantities()
}

generated quantities {
  // keep original outputs
  array[num_entries] vector[pred_N] pred_time_effect;
  matrix[pred_N, 2 * M_f] pred_PHI_f;
  vector[N] log_lik;

  pred_PHI_f = PHI_periodic(pred_N, M_f, 2 * pi() / period, pred_xn);
  for (i in 1:num_entries)
    pred_time_effect[i] = intercept_f[i] + pred_PHI_f * (diagSPD_f .* beta_f[i]);
  for (n in 1:N)
    log_lik[n] = rsm(
      y[n],
      plot_effect[plot_code[n]] + time_effect[entry_code[n]][rating_event_code[n]],
      tau_rater[rater_code[n]]
    );

  // NEW: posterior predictive samples
  array[N] int y_rep;
  for (n in 1:N) {
    real theta_n = plot_effect[plot_code[n]] + time_effect[entry_code[n]][rating_event_code[n]];
    vector[y_max + 1] unsummed_n = append_row(rep_vector(0, 1), theta_n - tau_rater[rater_code[n]]);
    y_rep[n] = categorical_rng(softmax(cumulative_sum(unsummed_n))) - 1;
  }
}
```

**How to run it**:

```python
from cmdstanpy import CmdStanModel
import pickle

# load existing fit
with open("fit_seasonality_nj2_quality.pkl", "rb") as f:
    fit = pickle.load(f)

# compile PPC model
ppc_model = CmdStanModel(
    stan_file="models/annual_seasonality_model_ppc.stan",
    stanc_options={"include-paths": [gptools_include_path]}
)

# generate y_rep from existing posterior draws — no refitting
gq = ppc_model.generate_quantities(
    data=stan_data,          # same stan_data used for the original fit
    previous_fit=fit,
    seed=42
)

# extract y_rep: shape (num_draws, N)
y_rep = gq.stan_variable("y_rep")
```

**Deliverable**: `models/annual_seasonality_model_ppc.stan` file + a script (`scripts/generate_ppc.py` or notebook cell) that produces and saves `y_rep`.

#### I4. Fix or bypass `PosteriorSampleAnalysis` bugs

**Problem**: `PosteriorSampleAnalysis` in `nteprsm/utils.py` has 5 bugs that crash at runtime. However, the existing notebook (`notebooks/manuscript2_visualizations_nj2.ipynb`) already bypasses this class entirely by working directly with CmdStanPy arrays + plotly.

**Bugs at exact locations**:

| Line | Bug | Fix |
|------|-----|-----|
| 425  | `self.datahandler.get_stan_data()["num_entries"]` | change to `self.datahandler.stan_data["num_entries"]` |
| 454  | `self.datahandler.map_name2code("entry_name", "entry_name_code")` | import `map_name2code` from `notebooks/rutils.py` or add as `DataHandler` method |
| 465  | `self.logger.warningn(...)` | change to `self.logger.warning(...)` |
| 504  | `self.datahandler.map_name2code(..., invert=True)` | same as line 454 |
| 614  | `self.datahandler.map_name2code(...)` in `plot_rater_characteristic_curve()` | same as line 454 |

**Recommended decision**: bypass the class. All new figures (V1, V2, R1, I1) should be built the same way the notebook already works — extracting numpy arrays from `fit.stan_variable(...)` and plotting with plotly or matplotlib directly. This avoids a fragile dependency.

**If Henry wants the bugs fixed instead**: apply the 5 fixes listed above, then run a smoke test: instantiate `PosteriorSampleAnalysis` with the NJ2 fit and call `plot_time_effect(entries=4)` to verify it doesn't crash.

**Deliverable**: confirm in writing to Henry which path was taken (bypass or fix) so the notebook coding style is consistent going forward.

## Success criteria for resubmission

The revision should make it easy for the editor to conclude:

- the paper now makes claims proportional to the evidence
- the validation is materially stronger than the original submission
- the model assumptions are justified clearly enough for a methods paper
- the results are systematic rather than anecdotal
- the manuscript is technically and editorially clean

## Suggested execution sequence

### Week 0 (infrastructure sprint, ~3–5 days)

1. Henry: lock scope, decide held-out approach (Stan vs. Python scoring), decide ablation yes/no
2. Olena: verify data access on UMN cluster (task I1)
3. Olena: build train/test split in `DataHandler` (task I2)
4. Olena: add `y_rep` generation path (task I3)
5. Olena: fix or bypass `PosteriorSampleAnalysis` bugs (task I4)
6. Olena: return feasibility note confirming infrastructure is ready

### Week 1

1. launch Analysis 1 (held-out prediction) on cluster
2. launch Analysis 2 (PPCs) using `generate_quantities()` on existing fits
3. define the exact whole-dataset summary metrics for Analysis 3

### Week 2

1. finish held-out prediction outputs
2. finish posterior predictive checks
3. generate whole-dataset summary figure and table
4. rebuild the model-comparison table with differences and uncertainty

### Week 3

1. rewrite Methods and Results around the new evidence
2. rewrite abstract, title, and conclusion
3. add limitations and claim calibration edits

### Week 4

1. optional stretch analyses if still needed
2. copyedit, figure cleanup, and response-letter drafting
3. prepare the marked manuscript and submission files

## Immediate next tasks

1. decide the exact held-out design: event-blocked split or leave-one-event-out
2. decide whether held-out scoring will happen inside Stan (modified generated quantities) or in Python
3. decide whether posterior predictive checks alone are the second validation layer or whether a misspecification stress test is also needed
4. decide whether the ablation ladder (Analysis 5) is in or out — this affects manuscript framing now, not later
5. Olena: complete the pre-execution infrastructure tasks (I1–I4) before launching any analysis jobs
6. draft the seasonality-summary schema for all 89 entries so the whole-dataset figure can be generated early
7. rewrite the current ELPD table specification before any prose revision begins
