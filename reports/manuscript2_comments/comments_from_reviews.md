None selected

Skip to content
Using Gmail with screen readers
17 of 23,059
Agronomy Journal - Revision Request for Manuscript AJ-2026-01-0038-OA
Inbox

Agronomy Journal <onbehalfof@manuscriptcentral.com>
Attachments
Thu, Apr 2, 7:40 AM (3 days ago)
to me, niphokleung, oboiko, lenkne, ambika.chandra, kmorris, hmrekab

02-Apr-2026

Dear Dr. Yuanshuo Qu,

Thank you for submitting your manuscript, "A Bayesian Approach for Quantifying Turfgrass Seasonality from Turf Quality Ratings in the National Turfgrass Evaluation Program" (AJ-2026-01-0038-OA ), for publication in the Agronomy Journal.

After careful consideration, the expert reviewers and I feel your manuscript has merit but does not yet meet the criteria for publication in Agronomy Journal. Therefore, we will reconsider a new version of the manuscript, provided that the suggestions of the reviewers and myself are addressed. I strongly urge you to revise your manuscript in accordance with the suggestions provided below and submitting a revision. If you choose to revise the manuscript, you must address each revision request or concern with a written response. Please highlight the changes to your manuscript within the document by using the track changes mode in MS Word or by using bold or colored text.

To submit your revision login to https://mc.manuscriptcentral.com/agron and enter your Author Center, where you will find your manuscript title listed under "Manuscripts with Decisions." Under the "Action" column, click on "Create a Revision." Please note, after you create a revision, your manuscript will move to a different queue on your dashboard. If you leave the system and return, you will find the manuscript under "Revised Manuscripts in Draft"—click on "Continue Submission" to access again.

Author Contribution Indication
The contributions of each author to this work must now be indicated when you submit your revised manuscript using CRediT taxonomy (http://credit.niso.org/contributor-roles-defined/). If not provided already you MUST provide this information as part of the revision process. Author Contributions will be published with the accepted article and cannot be edited after article acceptance. Therefore you must ensure the Author Contribution information you provide is accurate prior to final acceptance.

This journal offers a number of license options for published papers; information about this is available here: https://authorservices.wiley.com/author-resources/Journal-Authors/licensing/index.html. The submitting author has confirmed that all co-authors have the necessary rights to grant in the submission, including in light of each co-author’s funder policies. If any author’s funder has a policy that restricts which kinds of license they can sign, for example if the funder is a member of Coalition S, please make sure the submitting author is aware.

Your revised manuscript should be submitted no later than 01-Jun-2026. If you need an extension, please contact me as soon as possible. If it is not possible for you to submit your revision in a reasonable amount of time, we may have to consider your paper as a new submission. Once again, thank you for submitting your manuscript to Agronomy Journal and I look forward to receiving your revision.

Sincerely,

Dr. Hossein Moradi Rekabdarkolaee
Associate Editor, Agronomy Journal

Editor: Dr. Silvia Pampana, silvia.pampana@unipi.it
Technical Editor: Dr. Ramon Leon, rleon@ncsu.edu

Reviewer(s)' Comments to Author:
Reviewer: 1

Comments to the Author
The manuscript has a solid technical idea, but in its current form it overclaims, under-validates, and does not adequately separate methodological novelty from favorable modeling assumptions. The core problem is not whether the model is interesting; it is. The problem is that the evidence presented is too narrow for the strength of the claims. The study is built and demonstrated on a single 2017 NTEP Kentucky bluegrass trial from one location in Adelphia, New Jersey, with 9,612 ratings, 35 rating events, 7 raters, 267 plots, and 89 entries, yet the paper repeatedly frames the framework as robust, scalable, and a foundation for future multi-location analyses. That leap is not earned by the current empirical evidence. A single-site demonstration is fine for a methods paper, but then the claims need to be cut back hard, or the authors need to add external validation across additional locations, years, or species. Right now, the generalizability claim is ahead of the data.
A major revision is needed on model identifiability and justification of assumptions. The paper relaxes the earlier proportional-odds structure by allowing rater-specific category thresholds, while simultaneously modeling seasonality with a periodic GP and space with an RBF GP, and then fixing all discrimination parameters to 1 on the basis that raters are experienced turfgrass researchers. That is not a sufficient statistical justification. In a model with latent quality, rater thresholds, entry-specific seasonality, and spatial structure, identifiability is a first-order issue, not a side note. The manuscript needs a much clearer explanation of what constraints anchor the latent scale, why fixing discrimination is defensible, and how sensitive the inferences are to that assumption. This becomes more pressing because the results themselves show threshold irregularities, including overlapping or reversed thresholds for Rater E, where category 2 is effectively unused. That is not just an interesting behavioral finding; it also raises the possibility of instability in the threshold specification and should push the authors to consider ordered-threshold constraints, partial pooling, or stronger regularization.
The validation strategy is also too sympathetic to the model. The parameter recovery exercise generates synthetic data from the same structural family used in estimation, with the same kernels and favorable event structure, and then shows that the model can recover what it generated. That proves the code is internally coherent. It does not prove robustness in realistic settings where assumptions are wrong, raters drift, kernels are misspecified, event timing is irregular, or rater identities are incomplete. The manuscript itself later admits that rater behavior may shift over time, spatial effects may vary, and inference depends on ratings being well distributed through the year. Once those weaknesses are acknowledged, the current recovery study looks insufficient. The paper needs at least one tougher validation layer: real-data held-out prediction beyond the reported LOO summary, stress tests under misspecified simulation scenarios, sensitivity to fewer rating events or fewer raters, or posterior predictive checks focused on ordinal calibration and category use. Without that, the validation reads like a best-case demonstration.
The model comparison section is not clean enough to support the causal story the paper implies. The new model is not just the old model plus seasonality. It also changes how rater effects are handled, moving from event-level severity adjustments and shared thresholds to rater-specific thresholds applied across events. On top of that, the paper introduces computational approximations for temporal and spatial processes. So when the manuscript reports better ELPD for the new model and faster sampling time, it is not actually isolating the contribution of seasonality. The improvement could come from the revised rater-threshold specification, from approximation choices, or from the combined package. That means the current comparison is confounded. The authors need an ablation analysis: old spatial model, spatial plus revised rater thresholds, spatial plus seasonality, and full model with approximations. They should also report the difference in ELPD explicitly with uncertainty and discuss practical significance, not just say the new model is better because the score is less negative. The jump in effective number of parameters is also substantial, so the tradeoff between fit, complexity, and interpretability needs a more serious treatment.
The results section is too selective and too narrative for a methods paper that claims practical decision support. The seasonality discussion focuses on four illustrative entries and explains their curves qualitatively, but that is not enough when the dataset contains 89 entries and the central claim is that the model yields actionable cultivar-level seasonal insight. The paper should provide a systematic summary across all entries: seasonal amplitude, timing of peak quality, timing of decline, uncertainty measures, and perhaps rankings or clusters of seasonal profiles. The same issue exists for interpretability of the latent scale. The manuscript proposes mapping latent values back to an “average rater” scale and also standardizing them as z-scores, but this remains conceptual rather than operational. The authors need to demonstrate how those translations materially improve decision-making on the actual dataset, not just describe them as possible frameworks. As written, the paper shows that the model produces curves, but not yet that those curves are comprehensively interpretable or decision-ready.
The limitations section is too passive relative to how central those limitations are. The manuscript admits that rater behavior and spatial effects are treated as constant, that seasonality estimation depends on event structure, and that the method requires known rater identities for all observations. These are not minor caveats. They are direct threats to validity and practical adoption. If rater drift is plausible, then a static threshold model may blur temporal changes in rating style with temporal changes in turf quality. If spatial effects change over time, a static spatial field may absorb or distort seasonal signals. If many real-world NTEP datasets do not reliably record rater identity, the method’s applicability is narrower than the paper suggests. These issues should be moved from the back of the paper into the framing of the contribution, and the authors should either quantify their impact or be much more restrained in the claims. Right now the paper acknowledges the landmines after walking past them.
There is also a writing and presentation problem that needs cleanup before publication. Several technical and typographic errors are visible in the proof, including “Seaonality Model,” “Catogory threshold,” “Stochastic processe,” the use of “Radical Basis Function” instead of “Radial Basis Function,” and repeated grammatical issues such as “a entry.” Those are not intellectually fatal, but they weaken confidence in a paper that depends on mathematical precision. The manuscript needs a careful line edit in addition to substantive revision.


Reviewer: 2

Comments to the Author
Reviewer’s Comments to Authors for Manuscript AJ-2026-01-0038-OA, entitled "A Bayesian Approach for Quantifying Turfgrass Seasonality from Turf Quality Ratings in the National Turfgrass Evaluation Program.”

The paper is mostly well written and has some merit if the grey areas are addressed.

Major comments:
1.      While the title implies that a Bayesian Approach is developed for quantifying Turfgrass seasonality, the plain language summary describes it as a statistical model – the consistency would be beneficial to the reader.
2.      The title and part of the abstract implies that the goal is to “quantify turfgrass seasonality from the ratings” while line 6-8 in the Abstract imply that the goal is to improve the ratings from an ordinal scale to a structured analytical framework.
3.      The paragraph in line 67-78 seems misplaced and should be in the Methodology section.
-- We agree. The original paragraph contained methodological detail that was better placed in the Methodology section. We revised the Introduction to retain only a brief motivation for using Gaussian processes and moved the fuller GP description and kernel specification to Section 2.5.

4.      While the time complexity was improved using the new approximation techniques, model complexity was increased by using Gaussian Processes.
5.      In line 157, the authors mention that the synthetic data was generated using predefined values. More details in the choice and rationale would help.
6.      Line 199 describes the dataset as being large – this is contradicting for the synthetic data.
7.      Lines 241-247 assumes that the dynamic rater behavior can be due to a temporal shift. While the authors cite literature on this claim (Huang, 2023), the literature also makes the claim without backing it up with any empirical evidence. Plots or existing literature to support this claim would be beneficial.
8.      The methods discuss the prior as diffuse or weakly informative priors. In the discussion, line 382 implies that the prior used are empirically informed.
9.      The SE increases in the new improved model – Table 1 in line 304. The authors do not address these and instead focus on the improved speed.
10.     In some parts of the paper, the seasonality is attributed to climate effects while in other parts, it is attributed to dynamic rater behavior.
11.     (**) In the introduction, lines 42-47, the approach used in the study” tackles” the challenges including “the subjective nature of the ratings in the ordinal scale…but does not quantify the exact difference between them”. The results from the study does not clearly discuss this.

Minor comments:
1.      The last sentence in the Plain Language Summary Section has a misplaced “I” at the end of the section – minor grammatical error.
2.      Figures 5 and 6 have unlabeled axes.


References
1.      Huang, H.-Y. (2023). “Modeling rating order effects under item response theory models for rater-mediated assessments”. Applied Psychological Measurement, 47(4), 312–327. https://doi.org/10.1177/01466216231174566

Associate Editor's Comments to Author:
Associate Editor: 1
Comments to the Author:
I want to echo the reviewers' comment on improving the validation part of the study.

Note: If this decision letter mentions attachments that did not get delivered, they are in your Author Center in ScholarOne at https://mc.manuscriptcentral.com/agron. Once in your Author Center, click “Manuscripts with Decisions” and click “View Decision Letter.” Attachments will be located at the bottom of the letter.
 One attachment
  •  Scanned by Gmail
