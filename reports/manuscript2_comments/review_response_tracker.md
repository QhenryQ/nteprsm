# Review Response Tracker for AJ-2026-01-0038-OA

Use this tracker to convert reviewer comments into concrete manuscript edits, analyses, and response-letter language.

## Status Key

- `todo`: not started
- `in progress`: actively being addressed
- `done`: analysis and manuscript edits complete
- `defer`: not addressed directly; requires justification in the response letter

## Editor and Administrative Items

| ID | Source | Comment summary | Planned action | Evidence needed | Target manuscript area | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | Editor letter | Submit revision by June 1, 2026 | Build internal schedule backward from deadline | revision calendar | project management | todo |
| E2 | Editor letter | Provide point-by-point written responses | Draft structured response letter with one entry per comment | response document | submission package | todo |
| E3 | Editor letter | Show revisions with track changes or highlighted text | Decide final manuscript editing mode early | marked manuscript | final submission files | todo |
| E4 | Editor letter | Provide CRediT author contributions | Collect and confirm contributions from all coauthors | CRediT statement | submission portal | todo |

## Reviewer 1

| ID | Source | Comment summary | Planned action | Evidence needed | Target manuscript area | Status |
| --- | --- | --- | --- | --- | --- | --- |
| R1-1 | Reviewer 1 | Claims are too broad for a single-site demonstration | Reframe as a single-site methods paper unless broader validation is added | revised framing text | title, abstract, introduction, discussion, conclusion | todo |
| R1-2 | Reviewer 1 | Generalizability claim is not earned | Remove or soften multi-location and scalability language | revised claims | abstract, discussion, conclusion | todo |
| R1-3 | Reviewer 1 | Identifiability needs clearer justification | Add explicit identification subsection and latent-scale anchoring explanation | method text, equations, constraints | methods | todo |
| R1-4 | Reviewer 1 | Fixing discrimination parameters to 1 is under-justified | Explain rationale and, if possible, assess sensitivity | justification text and optional sensitivity analysis | methods, discussion, supplement | todo |
| R1-5 | Reviewer 1 | Threshold irregularities suggest instability | Discuss Rater E more critically and consider regularization or ordered-threshold sensitivity | revised threshold discussion and optional sensitivity run | results, discussion, supplement | todo |
| R1-6 | Reviewer 1 | Parameter recovery is too sympathetic to the fitted model | Add harder validation layer beyond same-family recovery | new validation analysis | results, discussion | todo |
| R1-7 | Reviewer 1 | Need held-out prediction, misspecification, reduced-rater, or PPC checks | Choose at least one strong additional validation path | predictive or stress-test outputs | results, figures, tables | todo |
| R1-8 | Reviewer 1 | Current model comparison is confounded | Add ablation comparison or explicitly narrow causal claims | revised comparison table and text | results, discussion | todo |
| R1-9 | Reviewer 1 | Report ELPD differences with uncertainty | Add explicit differences and uncertainty | comparison statistics | results | todo |
| R1-10 | Reviewer 1 | Need a more serious discussion of fit-complexity tradeoffs | Interpret effective parameter increase and practical significance | revised interpretation | results, discussion | todo |
| R1-11 | Reviewer 1 | Results are too selective and narrative | Add full-entry systematic summary | table or figure across all entries | results | todo |
| R1-12 | Reviewer 1 | Need operational interpretation of latent scale | Demonstrate average-rater or z-score translation on real data | worked example or summary figure | results, discussion | todo |
| R1-13 | Reviewer 1 | Limitations are central, not minor caveats | Move limitations forward and state them more forcefully | revised framing | introduction, discussion | todo |
| R1-14 | Reviewer 1 | Writing and typography need cleanup | Full line edit and terminology audit | corrected manuscript | full paper | todo |

## Reviewer 2

| ID | Source | Comment summary | Planned action | Evidence needed | Target manuscript area | Status |
| --- | --- | --- | --- | --- | --- | --- |
| R2-1 | Reviewer 2 | Title and plain-language summary use inconsistent terminology | Choose one consistent description of the contribution | revised text | title, plain-language summary | todo |
| R2-2 | Reviewer 2 | Abstract goal is internally inconsistent | Rewrite abstract around one clear objective | revised abstract | abstract | todo |
| R2-3 | Reviewer 2 | One paragraph belongs in Methodology, not Introduction | Relocate and integrate that paragraph | structural edit | introduction, methods | todo |
| R2-4 | Reviewer 2 | Time complexity improved but model complexity increased | Acknowledge computational gain versus model complexity tradeoff | revised discussion | methods, discussion | todo |
| R2-5 | Reviewer 2 | Synthetic-data parameter choices need more rationale | Document how predefined values were chosen | parameter table or rationale text | methods | todo |
| R2-6 | Reviewer 2 | Calling the synthetic dataset large is misleading | Revise wording for precision | copyedit | results or methods | todo |
| R2-7 | Reviewer 2 | Dynamic rater behavior explanation needs support | Add evidence, qualify claim, or reduce speculation | plot, citation support, or toned-down text | results, discussion | todo |
| R2-8 | Reviewer 2 | Prior description is inconsistent | Reconcile weakly informative versus empirically informed language | revised prior description | methods, discussion | todo |
| R2-9 | Reviewer 2 | SE increased in improved model but paper focuses only on speed | Address standard error increase directly | comparison text and maybe table annotation | results, discussion | todo |
| R2-10 | Reviewer 2 | Seasonality attribution is inconsistent across climate and rater behavior | Distinguish biological seasonality from rater-driven variation | revised interpretation | discussion | todo |
| R2-11 | Reviewer 2 | Intro claim about tackling ordinal-scale subjectivity is not fully reflected in Results | Add explicit results interpretation showing what latent scale resolves | revised results text and example | introduction, results, discussion | todo |
| R2-12 | Reviewer 2 | Minor grammar issue in plain-language summary | Fix typo | copyedit | plain-language summary | todo |
| R2-13 | Reviewer 2 | Figures 5 and 6 have unlabeled axes | Add axis labels and verify all figures | revised figures | figures, captions | todo |

## Associate Editor

| ID | Source | Comment summary | Planned action | Evidence needed | Target manuscript area | Status |
| --- | --- | --- | --- | --- | --- | --- |
| AE1 | Associate Editor | Improve validation part of the study | Prioritize new validation work before lower-priority cosmetic edits | new validation section | results, discussion, response letter | todo |

## Recommended First Pass Sequence

1. Lock scope. Decide whether the revision will remain single-site or add broader validation.
2. Select validation additions. This is the editor-level priority.
3. Run needed analyses. Do not rewrite major claims until the evidence package is fixed.
4. Rewrite framing, methods, results, and discussion.
5. Clean figures, captions, and language.
6. Draft the point-by-point response letter using this tracker.

## Response-Letter Template

For each comment, use a consistent structure:

1. Thank the reviewer briefly and restate the issue precisely.
2. State what was changed, where it was changed, and why.
3. If no direct change was made, justify the decision respectfully and narrowly.
4. Point to the exact revised section, figure, table, or supplement item.

Example skeleton:

```text
Comment R1-X:
[paste or paraphrase reviewer comment]

Response:
We thank the reviewer for raising this point. We have revised the manuscript to [summary of change]. Specifically, we [analysis / text / figure change]. These revisions appear in [section / figure / table].
```

## Draft Responses

Comment R1-5:
The results show threshold irregularities, including overlapping or reversed thresholds for Rater E, where category 2 is effectively unused. This may indicate instability in the threshold specification rather than only an interesting behavioral finding, and it raises the possibility of ordered-threshold constraints, partial pooling, or stronger regularization.

Response:
We thank the reviewer for raising this point. We agree that the Rater E threshold pattern should not be over-interpreted as definitive evidence of a stable behavioral trait. We have revised the manuscript to frame this result more cautiously as an empirical feature of the present unconstrained-threshold specification. Specifically, in Section 3.2 we now state that the apparent nonuse of category 2 for Rater E arises under the current fit and should be interpreted in light of the broader Rasch literature on disordered thresholds, where such patterns are recognized but debated in their interpretation. We cite Adams et al. (2012), who argue that reversed thresholds are not necessarily evidence of category disorder, and Andrich (2013), who argues that they still represent an anomaly requiring substantive explanation. Consistent with that literature, we now present the Rater E result as diagnostically informative but not conclusive, and we identify ordered-threshold or more strongly regularized variants as reasonable future sensitivity targets. These revisions appear in the revised Section 3.2.

Comment R2-10:
In some parts of the paper, the seasonality is attributed to climate effects while in other parts, it is attributed to dynamic rater behavior.

Response:
We thank the reviewer for identifying this ambiguity. We agree that the original wording blurred the distinction between a biological or environmental interpretation of seasonality and a model-based temporal pattern inferred from ordinal ratings. We have revised the manuscript to clarify that the temporal component is a latent seasonality term estimated after adjustment for rater-specific threshold behavior and within-trial spatial heterogeneity. Specifically, in Section 2.5 we now state that this component summarizes systematic temporal structure in turf quality within the NJ2 trial and may be consistent with biological or environmental seasonality, but it is not direct proof of specific climate effects. We also added matching language in the Discussion to state that the inferred temporal signal should be interpreted as a model-based seasonality pattern after adjustment for the rater and spatial terms, rather than as a direct estimate of climate-driven biology. These revisions appear in Section 2.5 and Section 4.