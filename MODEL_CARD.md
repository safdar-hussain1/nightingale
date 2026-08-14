# Model card — Nightingale

Six binary risk models, one per condition, all trained by the same pipeline:
gradient-boosted trees (XGBoost, `tree_method="hist"`) inside a 5-fold outer /
3-fold inner nested cross-validation, with the calibrator cross-fitted across
outer folds so no reported probability was scored by a calibrator that had seen
that row. Every metric is `point [95% CI lo, hi]` from a 2,000-replicate
stratified bootstrap, `seed=42` for training and `seed=0` (offset per metric)
for the intervals. Conformal prediction sets use split conformal at α = 0.1.

These are screening-triage risk models trained on small public research datasets. They are not diagnostic devices and must not be used for medical decisions. The diabetes labels are self-reported survey responses. The cervical-cancer cohort has 55 positive biopsies.

Nothing in this card was tuned to look better. Where a result is weak, close to
chance, or worse after an intervention, it is printed as measured.

---

## breast-cancer — Breast Cancer Wisconsin (Diagnostic)

**Intended use.** Research into screening-triage risk ranking on the WDBC
benchmark, and as a well-separated reference condition for calibration and
parity work. **Not diagnosis.** No clinical decision may be taken from it.

**Training data.** UCI 17 (WDBC), CC BY 4.0. n = 569, positives = 212 (37.26%).
Labels are the dataset's own `Diagnosis` field, B/M, mapped to 0/1 —
histopathological diagnoses recorded alongside the imaging measurements, not
self-report. All 30 features are computed from digitised fine-needle-aspirate
images; there are no demographic fields at all.

**Metrics** (out-of-fold, calibrated):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9864 [0.9759, 0.9948] |
| PR-AUC | 0.9801 [0.9644, 0.9919] |
| Brier | 0.0320 [0.0223, 0.0430] |
| ECE | 0.0240 [0.0160, 0.0402] |

Conformal α = 0.1, q̂ = 0.1111. Verdict mix: 36.56% positive, 63.44% negative,
**0.00% uncertain**. The uncertain region only exists when q̂ > 0.5, so at
q̂ ≈ 0.111 it is structurally empty — this model never abstains, and that is a
property of how well-separated WDBC is, not evidence of reliability.

**Subgroups.** None computed. WDBC carries no sex and no age column — the
features are purely image-derived. That is itself a limitation: this model
cannot be audited for demographic performance gaps at all.

**Calibration method.** Isotonic, cross-fitted across the five outer folds.
Note that its calibrated ECE (0.0240) is slightly *worse* than its uncalibrated
ECE (0.0212) — on a few hundred rows per fold, calibration is genuinely noisy
and does not guarantee improvement. Reported as measured.

**Known failure modes.** Isotonic saturation: several probabilities land exactly
on 0 or 1, which makes the calibrated output insensitive to real changes in the
underlying margin — the fingerprint mechanism had to be redesigned around this
(it compares pre-calibration outputs for exactly this reason). Small-cohort
variance: 569 rows, one institution, one imaging protocol.

**Must not be used for.** Any diagnostic or treatment decision; any imaging
pipeline whose measurements are not produced the way WDBC's were; any claim
about performance on a demographic subgroup, which this dataset cannot support.

---

## cervical-cancer — Cervical Cancer (Risk Factors)

**Intended use.** Research into risk ranking from questionnaire-style risk
factors in a very-low-prevalence screening cohort, and as this project's worked
example of an honestly weak result. **Not diagnosis.**

**Training data.** UCI 383, CC BY 4.0. n = 858, positives = 55 (6.41%). The
target is the `Biopsy` column — a positive cervical-cancer biopsy. **The whole
cohort contains 55 positive biopsies.** Every interval below is wide for that
reason, and no subgroup analysis beyond a single age band is possible.
Three sibling screening outcomes from the same episode (`Hinselmann`,
`Schiller`, `Citology`) were dropped from the feature set as target leakage,
and two columns 91.7% missing were dropped as unusable.

**Metrics** (out-of-fold, calibrated):

| Metric | Value |
|---|---|
| ROC-AUC | 0.6703 [0.5942, 0.7452] |
| PR-AUC | 0.1412 [0.0975, 0.2318] |
| Brier | 0.0586 [0.0544, 0.0626] |
| ECE | 0.0155 [0.0090, 0.0300] |

The PR-AUC upper bound is more than twice its lower bound. This is a weak
model, published because it is the measured result.

Conformal α = 0.1, q̂ = 0.1552. Verdict mix: 0.93% positive, 99.07% negative,
**0.00% uncertain** — again structurally empty at q̂ < 0.5. At 6.41% prevalence
the model almost always says "negative", which is a rational thing for a
low-prevalence model to do and a nearly useless thing for a screening tool to
do.

**Subgroups** (`min_n = 40`; anything smaller is reported as insufficient, never
as a number):

| Age band | n | ROC-AUC |
|---|---:|---|
| <45 | 835 | 0.6742 [0.5872, 0.7591] |
| 45-54 | 18 | insufficient n |
| 55-64 | 1 | insufficient n |
| 65+ | 4 | insufficient n |

The cohort is 97% under 45 by design. The `<45` figure is effectively the
whole-condition figure; **this model has no evidence of any kind about women
over 45.** No sex audit exists — the cohort is single-sex.

**Calibration method.** Isotonic, cross-fitted.

**Known failure modes.** 55 positives is too few to trust any of this to a
decimal place. Age coverage is effectively nil above 45. Two features
(`STDs:cervical condylomatosis`, `STDs:AIDS`) are all-zero in this cohort and
carry no information whatsoever.

**Must not be used for.** Deciding whether anyone should be biopsied, referred,
or reassured; any use on a population older than this cohort; any presentation
of the point estimate without its interval.

---

## heart-disease — Heart Disease (four-hospital cohort)

**Intended use.** Research into risk ranking across four hospitals with very
different prevalence and data availability, and the vehicle for this project's
external-validation study. **Not diagnosis.**

**Training data.** UCI 45, CC BY 4.0. n = 920 pooled across Cleveland (303),
Hungary (294), Switzerland (123) and VA Long Beach (200); positives = 509
(55.33%). The target is the dataset's `num` field binarised at `num > 0`
(presence of heart disease) — clinician-recorded, not self-report. `site` is
recorded in the cleaned data as an audit dimension and is **excluded from the
feature set**; the model never sees which hospital a row came from.

**Metrics** (out-of-fold, calibrated, pooled model):

| Metric | Value |
|---|---|
| ROC-AUC | 0.8892 [0.8663, 0.9101] |
| PR-AUC | 0.8892 [0.8616, 0.9167] |
| Brier | 0.1280 [0.1145, 0.1425] |
| ECE | 0.0236 [0.0202, 0.0536] |

Conformal α = 0.1, q̂ = 0.6842. Verdict mix: 45.76% positive, 34.78% negative,
**19.46% uncertain** — this model does abstain, on roughly one row in five.

**Subgroups** (`min_n = 40`):

*By site — the headline result:*

| Site | n | ROC-AUC | Mean calibration error |
|---|---:|---|---:|
| Cleveland | 303 | 0.8900 [0.8513, 0.9266] | 0.0062 |
| Hungary | 294 | 0.8760 [0.8295, 0.9170] | 0.0358 |
| **Switzerland** | 123 | **0.7359 [0.5190, 0.9103]** | **0.0712** |
| **VA Long Beach** | 200 | **0.7099 [0.6282, 0.7920]** | 0.0031 |

Switzerland's lower bound is 0.519 — at n = 123 its performance is **not
reliably distinguishable from a coin flip** at the low end of the interval,
even though the point estimate looks respectable. This model must not be
presented as equally reliable everywhere it was validated.

*By sex:* group 0 (n = 194) 0.8844 [0.8244, 0.9353]; group 1 (n = 726) 0.8709
[0.8428, 0.8977]. Intervals overlap heavily — no material gap.

*By age band:* <45 (n = 178) 0.9360 [0.8822, 0.9751]; 45-54 (n = 294) 0.8889
[0.8492, 0.9250]; 55-64 (n = 345) 0.8186 [0.7682, 0.8661]; 65+ (n = 103) 0.8799
[0.7892, 0.9518]. Overlapping intervals throughout — no material gap.

**Calibration method.** Isotonic, cross-fitted.

**Known failure modes.**

- **Cross-site calibration shatters.** A Cleveland-only model deployed at the
  other three sites keeps some discrimination (0.69–0.87 ROC-AUC) and loses
  calibration entirely (ECE 0.078–0.207 naive). Intercept-only recalibration
  fixes Hungary (ECE 0.1585 → 0.0466) and Switzerland (0.2007 → 0.0305) and
  **does not fix VA** (0.0697 → 0.0918, intervals overlapping): VA's naive
  calibration **slope is 0.478**, so its problem is a too-steep risk gradient,
  not a prevalence shift, and an intercept correction cannot address it.
- **Feature availability is site-dependent.** After cleaning, the share of rows
  the model cannot use: `chol` 0.0% at Cleveland but 100.0% at Switzerland and
  28.0% at VA; `ca` 1.3% at Cleveland but 95.9–99.0% elsewhere; `thal` 0.7% at
  Cleveland but 42.3–90.5% elsewhere. A model that looks feature-rich in
  training is running on a much thinner input at three of its four sites.
- Small-cohort variance: 920 rows total, 123 at the smallest site.

**Must not be used for.** Diagnosis, referral, or triage of any actual patient;
deployment at a site whose prevalence or data availability differs from
Cleveland's without local recalibration *and* a check that the miscalibration is
a shift rather than a slope.

---

## kidney-disease — Chronic Kidney Disease

**Intended use.** Research on a small, cleanly-split CKD benchmark, and as this
project's worked example of a number that is too good to take at face value.
**Not diagnosis.**

**Training data.** UCI 336, CC BY 4.0. n = 400, positives = 250 (62.50%). The
target is the dataset's `class` column (`ckd` / `notckd`) — a clinical
diagnosis, though the raw labels are whitespace-dirty (`'ckd\t'` appears twice
and is stripped during cleaning, or it would silently become a third class).

**Metrics** (out-of-fold, calibrated):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9979 [0.9937, 1.0000] |
| PR-AUC | 0.9985 [0.9955, 1.0000] |
| Brier | 0.0059 [0.0011, 0.0129] |
| ECE | 0.0091 [0.0028, 0.0171] |

**This number was investigated before it was published**, on the principle that
a too-good number is a bug report. It is not target leakage — no feature or
transform reproduces the target — but it is not a clinical result either:

1. All ten categorical clinical-finding columns split the classes perfectly
   across all 400 rows. A trivial rule ("any abnormal finding ⇒ CKD") scores
   93.0% accuracy with **zero** false positives.
2. Individual labs are extremely discriminative on complete cases alone —
   haemoglobin 0.969, packed cell volume 0.952, RBC count 0.924, serum
   creatinine 0.923. That part is real physiology.
3. **Missingness is class-correlated and is itself a shortcut.** Healthy rows
   have 3–6% missing labs; CKD rows are missing a third to half of several.
   The presence/absence indicator alone reaches ROC-AUC 0.7247 on `rbcc`,
   0.6747 on `wbcc`, 0.6493 on `pot`. XGBoost learns a default direction per
   split, so it can read the label off which tests happen to be present.

**Conformal disclosure — q̂ = 0.0, so this model can never abstain.** 98.25% of
its calibrated out-of-fold probabilities are exactly 0.0 or 1.0 (isotonic
saturating on near-separable data), so the 90th-percentile nonconformity score
is zero and the conformal set is empty for every probability strictly inside
(0, 1). Every verdict therefore comes from the documented argmax fallback:
63.25% positive, 36.75% negative, **0.00% uncertain — and "uncertain" is
unreachable, not merely unobserved.** Anywhere this model's verdicts are shown,
that must be stated, or a reader will infer it is uniformly confident because it
is uniformly correct.

**Subgroups** (`min_n = 40`):

| Age band | n | ROC-AUC |
|---|---:|---|
| <45 | 114 | 1.0000 [1.0000, 1.0000] |
| 45-54 | 81 | 0.9901 [0.9691, 1.0000] |
| 55-64 | 98 | 1.0000 [1.0000, 1.0000] |
| 65+ | 98 | 1.0000 [1.0000, 1.0000] |
| (age missing) | 9 | insufficient n |

Separability holds in every age band, not just in aggregate. No sex column
exists in this dataset, so no sex audit is possible.

**Calibration method.** Isotonic, cross-fitted.

**Known failure modes.** The headline number is a property of this 400-row
cohort's construction (a hospital's confirmed CKD caseload versus a comparison
set with unremarkable labs), not of CKD screening. It would not survive contact
with an undifferentiated primary-care population given a complete panel. The
conformal machinery is inert here.

**Must not be used for.** Any claim that CKD is 99.8%-detectable; any deployment
where a missing lab result means "not ordered yet" rather than what it means in
this cohort; any use of its verdicts without the q̂ = 0 disclosure.

---

## liver-disease — ILPD (Indian Liver Patient Dataset)

**Intended use.** Research into risk ranking from a routine liver-function
panel. **Not diagnosis.**

**Training data.** UCI 225 (ILPD), CC BY 4.0. n = 583, positives = 416 (71.36%).
The raw target is `Selector`, coded **1 = liver patient, 2 = non-patient** —
inverted relative to the usual convention and remapped during cleaning; getting
this backwards would produce a model that is confidently wrong. Note what the
label means: "patient" is *referred to and recorded by a liver clinic*, not a
specific disease or severity. The cohort is 71% positive, which is a clinic
population, not a screening population.

**Metrics** (out-of-fold, calibrated):

| Metric | Value |
|---|---|
| ROC-AUC | 0.7051 [0.6626, 0.7451] |
| PR-AUC | 0.8622 [0.8359, 0.8872] |
| Brier | 0.1805 [0.1699, 0.1907] |
| ECE | 0.0412 [0.0241, 0.0726] |

PR-AUC looks strong only because prevalence is 71.36% — always predicting
positive already scores about 0.714. ROC-AUC (0.7051) is the honest
discrimination figure.

Conformal α = 0.1, q̂ = 0.5846. Verdict mix: 51.97% positive, 0.86% negative,
**47.17% uncertain**. This model abstains on nearly half its rows, which is the
correct behaviour for a model this weak and is the most useful thing it does.

**Subgroups** (`min_n = 40`):

*By sex — a real gap:*

| Group | n | ROC-AUC |
|---|---:|---|
| Female (`sex_male` = 0) | 142 | **0.6278 [0.5295, 0.7155]** |
| Male (`sex_male` = 1) | 441 | 0.7230 [0.6723, 0.7694] |

A gap of roughly 0.10, and the female group's lower bound (0.5295) is close to
chance. **This model discriminates meaningfully worse for women**, and must not
be presented as condition-wide.

*By age band:* <45 (n = 275) 0.6724 [0.6105, 0.7311]; 45-54 (n = 132) 0.6437
[0.5437, 0.7395]; 55-64 (n = 99) 0.7840 [0.6918, 0.8659]; 65+ (n = 77) 0.7608
[0.6365, 0.8715]. Wide, overlapping intervals — no band reliably worse.

**Calibration method.** Isotonic, cross-fitted.

**Known failure modes.** The sex gap above. A label that means "attended a liver
clinic" rather than a defined disease state. Severe class imbalance in the
non-obvious direction (71% positive), which makes every threshold-free summary
flattering. Small-cohort variance at 583 rows, 142 of them female.

**Must not be used for.** Any decision about a woman's liver health, given the
subgroup result; any screening context, given the cohort is a clinic
population; any inference about which liver disease a person has.

---

## diabetes — CDC Diabetes Health Indicators (BRFSS 2015)

**Intended use.** Research into population-scale risk stratification from
survey-reported health indicators. **Not diagnosis.**

**Training data.** UCI 891, derived from the CDC Behavioral Risk Factor
Surveillance System 2015 telephone survey; public domain. n = 253,680,
positives = 35,346 (13.93%).

**Label provenance — read this before quoting any number here.** The target is
`Diabetes_binary`: **a self-reported survey answer**, not a clinical diagnosis
and not a laboratory result. It records whether a respondent told a telephone
interviewer they had been told they have diabetes or prediabetes. Undiagnosed
diabetes is invisible to this label by construction, and so is anyone who
declined to say. All 21 features are likewise integer-coded self-report.

**Metrics** (out-of-fold, calibrated):

| Metric | Value |
|---|---|
| ROC-AUC | 0.8288 [0.8268, 0.8309] |
| PR-AUC | 0.4273 [0.4227, 0.4321] |
| Brier | 0.0969 [0.0964, 0.0973] |
| ECE | 0.0008 [0.0007, 0.0024] |

The intervals are tight because n is large, not because the model is certain
about any individual. ECE near 0.001 says the *average* predicted risk matches
the *average* observed rate — it says nothing about an individual prediction.

Conformal α = 0.1, q̂ = 0.6007. Verdict mix: 1.16% positive, 91.24% negative,
**7.61% uncertain**.

**Subgroups** (`min_n = 40`; with n this large, non-overlapping intervals are
easy to obtain and small gaps are statistically real):

*By sex:* group 0 (n = 141,974) 0.8401 [0.8373, 0.8428]; group 1
(n = 111,706) 0.8139 [0.8107, 0.8170]. A ~0.026 gap with non-overlapping
intervals — small in absolute size, but real.

*By age band:*

| Age band | n | ROC-AUC |
|---|---:|---|
| <45 | 54,401 | 0.8548 [0.8468, 0.8626] |
| 45-54 | 46,133 | 0.8367 [0.8315, 0.8421] |
| 55-64 | 64,076 | 0.8194 [0.8152, 0.8235] |
| **65+** | 89,070 | **0.7631 [0.7592, 0.7667]** |

A clear monotonic decline with age, with non-overlapping intervals throughout.
The 65+ band is the largest single group (35% of the dataset) and the worst
served — worth stating plainly to anyone considering an older population.

**Calibration method.** Isotonic, cross-fitted. The inner grid search was run on
a 20,000-row subsample per inner fold for tractability; the outer folds and the
final fit used all rows.

**Known failure modes.** The self-report label ceiling above all else. BRFSS
2015 is a decade-old snapshot of one country's telephone-survey respondents;
nothing in this project detects or corrects drift since. Age-related decline in
discrimination. Landline/mobile survey coverage bias is inherited whole and
unmeasured here.

**Must not be used for.** Diagnosing diabetes; screening an individual; any
population outside US adults in 2015 without revalidation; any claim that the
label represents clinical diabetes status.

---

## Global limitations

**External validity has been tested for exactly one of the six conditions.**
The heart-disease model was deployed at three hospitals it never saw during
training, and the result was mixed by design: discrimination partly transferred,
calibration did not transfer at all, and one site (VA Long Beach) could not be
repaired by the recalibration that worked at the other two.

**The other five conditions have zero external validation.** Breast cancer,
cervical cancer, chronic kidney disease, liver disease and diabetes were each
trained and evaluated within a single cohort. Every number reported for them is
an out-of-fold estimate on rows drawn from the same source, cleaned the same
way, collected under the same protocol. Nothing here establishes that any of
those five models would retain either its discrimination or its calibration on
a different hospital, a different country, a different decade, or a different
patient mix — and the one case where that *was* tested says calibration is the
first thing to break.

**Cohort sizes are small.** Five of the six conditions have between 400 and 920
rows, with as few as 55 positives (cervical cancer) and 142 rows in an audited
subgroup (liver disease, female). Bootstrap intervals are reported everywhere
precisely because point estimates at this scale are unstable; the intervals are
the result, not decoration on it.

**Subgroup audits are limited to what the data carries.** Breast cancer has no
demographic columns at all. Cervical cancer and chronic kidney disease have no
sex column. Cervical cancer's cohort is 97% under 45, so three of its four age
bands are reported as insufficient n rather than as numbers. An absent audit is
not a passed audit.

**Calibration is the fragile part.** Every calibrated figure here is
cross-fitted, and an earlier in-sample version of the same code produced ECE
near 1e-19 — a value that only looked like a triumph. Calibration on a few
hundred rows per fold is noisy enough that it does not always improve on the
uncalibrated model (breast cancer, above, is slightly worse after calibration).

**What the signature proves is narrow.** The provenance chain proves the
artifacts on disk are the bytes that were signed, and that a given bundle
belongs to a named condition. It proves nothing about clinical validity.

**None of these are medical devices.** Repeating the banner at the top, because
it is the only sentence in this document that matters more than the numbers:
these are screening-triage research models on small public datasets, they are
not diagnostic devices, and they must not be used for medical decisions.
