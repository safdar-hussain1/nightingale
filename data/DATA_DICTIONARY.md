# Data dictionary

Column reference for the six cleaned datasets under `data/cleaned/<slug>.csv.gz`
— the files every later stage reads and the ones covered by the signed
manifest. Ranges are the observed min–max in the cleaned data (the same values
each `models/<slug>/model.json` publishes for its input widgets); missingness is
the share of rows that are NaN in the cleaned file, computed directly from it.

**Cleaned-data contract.** Every file is feature columns plus a `target` column
(`int64`, exactly `{0, 1}`). `heart-disease` additionally carries a `site`
column, which is **not a feature** — see the note at the end. Files are written
with `compression={"method": "gzip", "mtime": 0}` so a re-run is byte-identical
and the signature still verifies.

**Missing means missing.** Nothing here is mean-imputed or zero-filled. XGBoost
learns a default branch direction per split, so NaN is a first-class input, and
a blank field in the CLI or the dashboard becomes NaN rather than 0. On the real
heart-disease bundle, treating blanks as 0 instead moved a prediction from
0.8462 to 0.1350 and flipped its verdict.

---

## breast-cancer

**Source.** UCI 17, Breast Cancer Wisconsin (Diagnostic), CC BY 4.0.
**Shape.** 569 × 31 (30 features + `target`). **No column has any missing
value.**

**Target.** `target` = 1 when the original `Diagnosis` field is `M`
(malignant), 0 when `B` (benign). 212 / 569 = 37.26% positive.

Ten measurements of cell nuclei from a digitised fine-needle aspirate, each
reported three ways: suffix `1` = mean, `2` = standard error, `3` = "worst"
(mean of the three largest values). All are dimensionless image-derived
quantities; the dataset defines no units.

| Feature | Label | Type | Unit | Range | Missing |
|---|---|---|---|---|---|
| `radius1` | Nucleus radius (mean) | numeric | — | 6.981–28.11 | 0.0% |
| `texture1` | Nucleus texture (mean) | numeric | — | 9.71–39.28 | 0.0% |
| `perimeter1` | Nucleus perimeter (mean) | numeric | — | 43.79–188.5 | 0.0% |
| `area1` | Nucleus area (mean) | numeric | — | 143.5–2501 | 0.0% |
| `smoothness1` | Nucleus smoothness (mean) | numeric | — | 0.05263–0.1634 | 0.0% |
| `compactness1` | Nucleus compactness (mean) | numeric | — | 0.01938–0.3454 | 0.0% |
| `concavity1` | Nucleus concavity (mean) | numeric | — | 0–0.4268 | 0.0% |
| `concave_points1` | Nucleus concave points (mean) | numeric | — | 0–0.2012 | 0.0% |
| `symmetry1` | Nucleus symmetry (mean) | numeric | — | 0.106–0.304 | 0.0% |
| `fractal_dimension1` | Nucleus fractal dimension (mean) | numeric | — | 0.04996–0.09744 | 0.0% |
| `radius2` | Nucleus radius (std. error) | numeric | — | 0.1115–2.873 | 0.0% |
| `texture2` | Nucleus texture (std. error) | numeric | — | 0.3602–4.885 | 0.0% |
| `perimeter2` | Nucleus perimeter (std. error) | numeric | — | 0.757–21.98 | 0.0% |
| `area2` | Nucleus area (std. error) | numeric | — | 6.802–542.2 | 0.0% |
| `smoothness2` | Nucleus smoothness (std. error) | numeric | — | 0.001713–0.03113 | 0.0% |
| `compactness2` | Nucleus compactness (std. error) | numeric | — | 0.002252–0.1354 | 0.0% |
| `concavity2` | Nucleus concavity (std. error) | numeric | — | 0–0.396 | 0.0% |
| `concave_points2` | Nucleus concave points (std. error) | numeric | — | 0–0.05279 | 0.0% |
| `symmetry2` | Nucleus symmetry (std. error) | numeric | — | 0.007882–0.07895 | 0.0% |
| `fractal_dimension2` | Nucleus fractal dimension (std. error) | numeric | — | 0.0008948–0.02984 | 0.0% |
| `radius3` | Nucleus radius (worst) | numeric | — | 7.93–36.04 | 0.0% |
| `texture3` | Nucleus texture (worst) | numeric | — | 12.02–49.54 | 0.0% |
| `perimeter3` | Nucleus perimeter (worst) | numeric | — | 50.41–251.2 | 0.0% |
| `area3` | Nucleus area (worst) | numeric | — | 185.2–4254 | 0.0% |
| `smoothness3` | Nucleus smoothness (worst) | numeric | — | 0.07117–0.2226 | 0.0% |
| `compactness3` | Nucleus compactness (worst) | numeric | — | 0.02729–1.058 | 0.0% |
| `concavity3` | Nucleus concavity (worst) | numeric | — | 0–1.252 | 0.0% |
| `concave_points3` | Nucleus concave points (worst) | numeric | — | 0–0.291 | 0.0% |
| `symmetry3` | Nucleus symmetry (worst) | numeric | — | 0.1565–0.6638 | 0.0% |
| `fractal_dimension3` | Nucleus fractal dimension (worst) | numeric | — | 0.05504–0.2075 | 0.0% |

**Cleaning decisions.** Drop the `id` column if present (a row identifier is
never a feature). Map `Diagnosis` B/M → `target` 0/1 and drop `Diagnosis`.
Nothing else — this dataset arrives clean.

---

## cervical-cancer

**Source.** UCI 383, Cervical Cancer (Risk Factors), CC BY 4.0.
**Shape.** 858 × 31 (30 features + `target`). 24 columns carry at least one
missing value. Missing values arrive as the literal string `?` in the raw CSV
and become NaN.

**Target.** `target` = 1 for a positive cervical-cancer biopsy (the raw
`Biopsy` column). 55 / 858 = 6.41% positive.

| Feature | Label | Type | Unit | Range | Missing |
|---|---|---|---|---|---|
| `Age` | Age | numeric | years | 13–84 | 0.0% |
| `Number of sexual partners` | Number of sexual partners | numeric | count | 1–28 | 3.0% |
| `First sexual intercourse` | Age at first sexual intercourse | numeric | years | 10–32 | 0.8% |
| `Num of pregnancies` | Number of pregnancies | numeric | count | 0–11 | 6.5% |
| `Smokes` | Smokes | binary | 0/1 | 0–1 | 1.5% |
| `Smokes (years)` | Years smoking | numeric | years | 0–37 | 1.5% |
| `Smokes (packs/year)` | Packs per year | numeric | packs/year | 0–37 | 1.5% |
| `Hormonal Contraceptives` | Uses hormonal contraceptives | binary | 0/1 | 0–1 | 12.6% |
| `Hormonal Contraceptives (years)` | Years on hormonal contraceptives | numeric | years | 0–30 | 12.6% |
| `IUD` | Has an IUD | binary | 0/1 | 0–1 | 13.6% |
| `IUD (years)` | Years with an IUD | numeric | years | 0–19 | 13.6% |
| `STDs` | Any STD reported | binary | 0/1 | 0–1 | 12.2% |
| `STDs (number)` | Number of STDs reported | numeric | count | 0–4 | 12.2% |
| `STDs:condylomatosis` | Condylomatosis | binary | 0/1 | 0–1 | 12.2% |
| `STDs:cervical condylomatosis` | Cervical condylomatosis | binary | 0/1 | 0–0 (all zero) | 12.2% |
| `STDs:vaginal condylomatosis` | Vaginal condylomatosis | binary | 0/1 | 0–1 | 12.2% |
| `STDs:vulvo-perineal condylomatosis` | Vulvo-perineal condylomatosis | binary | 0/1 | 0–1 | 12.2% |
| `STDs:syphilis` | Syphilis | binary | 0/1 | 0–1 | 12.2% |
| `STDs:pelvic inflammatory disease` | Pelvic inflammatory disease | binary | 0/1 | 0–1 | 12.2% |
| `STDs:genital herpes` | Genital herpes | binary | 0/1 | 0–1 | 12.2% |
| `STDs:molluscum contagiosum` | Molluscum contagiosum | binary | 0/1 | 0–1 | 12.2% |
| `STDs:AIDS` | AIDS | binary | 0/1 | 0–0 (all zero) | 12.2% |
| `STDs:HIV` | HIV | binary | 0/1 | 0–1 | 12.2% |
| `STDs:Hepatitis B` | Hepatitis B | binary | 0/1 | 0–1 | 12.2% |
| `STDs:HPV` | HPV | binary | 0/1 | 0–1 | 12.2% |
| `STDs: Number of diagnosis` | Number of prior STD diagnoses | numeric | count | 0–3 | 0.0% |
| `Dx:Cancer` | Prior cancer diagnosis | binary | 0/1 | 0–1 | 0.0% |
| `Dx:CIN` | Prior CIN diagnosis | binary | 0/1 | 0–1 | 0.0% |
| `Dx:HPV` | Prior HPV diagnosis | binary | 0/1 | 0–1 | 0.0% |
| `Dx` | Any prior diagnosis | binary | 0/1 | 0–1 | 0.0% |

`STDs:cervical condylomatosis` and `STDs:AIDS` are 0 for every patient in this
858-row cohort. They are kept for schema fidelity and carry no information.

**Cleaning decisions.**

- **Dropped as too sparse:** `STDs: Time since first diagnosis` and
  `STDs: Time since last diagnosis` — both 787 / 858 = **91.7% missing**. No
  imputation recovers usable signal from 71 observations spread across a
  continuous scale, and keeping them would mostly encode "was this ever
  recorded", which is a data-collection artefact rather than a risk factor.
- **Dropped as target leakage:** `Hinselmann`, `Schiller` and `Citology`. All
  three are alternative screening tests performed in the *same screening
  episode* as `Biopsy`, the chosen target. A model given them would be
  predicting a biopsy result from three other readings of the same event —
  excellent on paper, useless in the clinic, where the point is to decide
  whether to do the screening at all. `Biopsy` itself is dropped from the
  features after being mapped to `target`.
- Net: 36 raw columns → 30 features + `target`.

---

## heart-disease

**Source.** UCI 45, Heart Disease, CC BY 4.0 — all four processed site files
(`processed.cleveland.data`, `.hungarian`, `.switzerland`, `.va`), not just the
Cleveland file that the `ucimlrepo` package serves.
**Shape.** 920 × 15 (13 features + `site` + `target`). 10 columns carry missing
values.

**Target.** `target` = 1 when the raw `num` field is > 0 (presence of heart
disease at any of its four graded levels), 0 when `num == 0`. 509 / 920 =
55.33% positive. `num` is dropped after mapping.

| Feature | Label | Type | Unit | Range | Missing |
|---|---|---|---|---|---|
| `age` | Age | numeric | years | 28–77 | 0.0% |
| `sex` | Sex (1 = male, 0 = female) | binary | 0/1 | 0–1 | 0.0% |
| `cp` | Chest pain type (1 typical angina … 4 asymptomatic) | categorical | code | 1–4 | 0.0% |
| `trestbps` | Resting blood pressure | numeric | mm Hg | 80–200 | 6.5% |
| `chol` | Serum cholesterol | numeric | mg/dL | 85–603 | 22.0% |
| `fbs` | Fasting blood sugar > 120 mg/dL | binary | 0/1 | 0–1 | 9.8% |
| `restecg` | Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy) | categorical | code | 0–2 | 0.2% |
| `thalach` | Maximum heart rate achieved | numeric | bpm | 60–202 | 6.0% |
| `exang` | Exercise-induced angina | binary | 0/1 | 0–1 | 6.0% |
| `oldpeak` | ST depression, exercise vs rest | numeric | mm ST | −2.6–6.2 | 6.7% |
| `slope` | Slope of the peak exercise ST segment (1 up, 2 flat, 3 down) | categorical | code | 1–3 | 33.6% |
| `ca` | Major vessels coloured by fluoroscopy | numeric | count | 0–3 | 66.4% |
| `thal` | Thalassemia test result (3 normal, 6 fixed defect, 7 reversible defect) | categorical | code | 3–7 | 52.8% |

**Cleaning decisions — the sentinel rules.**

- **`chol == 0` → NaN, scoped to Switzerland and VA only.** A living patient
  cannot have zero serum cholesterol; at these two sites, `0` is how missing
  cholesterol was encoded. Raw zero-fractions: Cleveland 0.0%, Hungary 0.0%,
  **Switzerland 100.0% (all 123 rows)**, VA 24.5% (49 of 200). 172 values are
  converted in total. **The rule is site-scoped on purpose:** applying it
  globally would destroy any genuine low-cholesterol reading at Cleveland or
  Hungary, which never emit a real `chol == 0`. A test asserts that Cleveland's
  zeros survive, and it was verified by deliberately removing the site scoping
  and confirming the test goes red.
- **`trestbps == 0` → NaN, globally.** One value, at VA. Zero resting blood
  pressure is impossible everywhere, so no site scoping is needed.
- After cleaning, no row anywhere has `chol == 0` or `trestbps == 0`, and
  Switzerland's `chol` is 100% NaN. The independent sentinel detector flags
  `chol` on raw Switzerland at a zero-fraction of 1.0000 and stops flagging it
  after cleaning — the cleaner and the detector agree.

**The `site` column.** Values `cleveland` (303), `hungarian` (294),
`switzerland` (123), `va` (200). It records which of the four contributing
hospitals a row came from and exists for two purposes: the subgroup audit in
`models/heart-disease/metrics.json`, and the Cleveland-only external-validation
study in `models/heart-disease/external.json`. **It is never a model feature** —
`nightingale.model` excludes it before training, and the model therefore cannot
learn "rows from Switzerland are usually positive". Feature availability by
site, as the share of rows the model cannot use after cleaning:

| Site | n | `chol` | `ca` | `thal` |
|---|---:|---:|---:|---:|
| Cleveland | 303 | 0.0% | 1.3% | 0.7% |
| Hungary | 294 | 7.8% | 99.0% | 90.5% |
| Switzerland | 123 | 100.0% | 95.9% | 42.3% |
| VA Long Beach | 200 | 28.0% | 99.0% | 83.0% |

VA's 28.0% is the union of its 24.5% sentinel zeros and its genuine `?` values,
which is why it exceeds the sentinel count alone.

---

## kidney-disease

**Source.** UCI 336, Chronic Kidney Disease, CC BY 4.0.
**Shape.** 400 × 25 (24 features + `target`). All 24 feature columns carry at
least one missing value.

**Target.** `target` = 1 for `ckd`, 0 for `notckd`. 250 / 400 = 62.50%
positive.

| Feature | Label | Type | Unit | Range / values | Missing |
|---|---|---|---|---|---|
| `age` | Age | numeric | years | 2–90 | 2.2% |
| `bp` | Blood pressure | numeric | mm Hg | 50–180 | 3.0% |
| `sg` | Urine specific gravity | categorical | — | 1.005 / 1.010 / 1.015 / 1.020 / 1.025 | 11.8% |
| `al` | Albumin (urine) | categorical | grade | 0–5 | 11.5% |
| `su` | Sugar (urine) | categorical | grade | 0–5 | 12.2% |
| `rbc` | Red blood cells (urine) | categorical | — | normal / abnormal | 38.0% |
| `pc` | Pus cells | categorical | — | normal / abnormal | 16.2% |
| `pcc` | Pus cell clumps | categorical | — | notpresent / present | 1.0% |
| `ba` | Bacteria | categorical | — | notpresent / present | 1.0% |
| `bgr` | Blood glucose, random | numeric | mg/dL | 22–490 | 11.0% |
| `bu` | Blood urea | numeric | mg/dL | 1.5–391 | 4.8% |
| `sc` | Serum creatinine | numeric | mg/dL | 0.4–76 | 4.2% |
| `sod` | Sodium | numeric | mEq/L | 4.5–163 | 21.8% |
| `pot` | Potassium | numeric | mEq/L | 2.5–47 | 22.0% |
| `hemo` | Haemoglobin | numeric | g/dL | 3.1–17.8 | 13.0% |
| `pcv` | Packed cell volume | numeric | % | 9–54 | 17.8% |
| `wbcc` | White blood cell count | numeric | cells/cmm | 2,200–26,400 | 26.5% |
| `rbcc` | Red blood cell count | numeric | millions/cmm | 2.1–8 | 32.8% |
| `htn` | Hypertension | categorical | — | no / yes | 0.5% |
| `dm` | Diabetes mellitus | categorical | — | no / yes | 0.5% |
| `cad` | Coronary artery disease | categorical | — | no / yes | 0.5% |
| `appet` | Appetite | categorical | — | good / poor | 0.2% |
| `pe` | Pedal oedema | categorical | — | no / yes | 0.2% |
| `ane` | Anaemia | categorical | — | no / yes | 0.2% |

The ten categorical columns are one-hot encoded inside each training fold, which
is why the model sees 34 features (14 numeric + 20 dummies) while the cleaned
file has 24 columns. This is the only condition with any one-hot expansion.

**Cleaning decisions.**

- **Strip whitespace from string columns, including the label.** The raw
  `class` column contains `'ckd\t'` (2 rows) alongside `'ckd'` (248) and
  `'notckd'` (150). Without the strip, the trailing tab silently creates a
  third class and two CKD patients are quietly mislabelled.
- Map the stripped `class` to `target` (ckd → 1) and drop it.
- **Missingness in this dataset is class-correlated**, and that is documented in
  `MODEL_CARD.md` rather than corrected here: healthy rows have 3–6% missing
  labs while CKD rows are missing a third to half of several. The
  presence/absence indicator for `rbcc` alone reaches ROC-AUC 0.7247. The
  cleaner deliberately does not impute this away — hiding it would make the
  model's headline number harder to interrogate, not more honest.

---

## liver-disease

**Source.** UCI 225, ILPD (Indian Liver Patient Dataset), CC BY 4.0.
**Shape.** 583 × 11 (10 features + `target`). One column (`A/G Ratio`) has
missing values.

**Target.** `target` = 1 for a liver patient, 0 for a non-patient. 416 / 583 =
71.36% positive. "Patient" means the person was recorded as a liver-clinic
patient — not a specific disease, stage, or severity.

| Feature | Label | Type | Unit | Range | Missing |
|---|---|---|---|---|---|
| `Age` | Age | numeric | years | 4–90 | 0.0% |
| `TB` | Total bilirubin | numeric | mg/dL | 0.4–75 | 0.0% |
| `DB` | Direct bilirubin | numeric | mg/dL | 0.1–19.7 | 0.0% |
| `Alkphos` | Alkaline phosphatase | numeric | IU/L | 63–2,110 | 0.0% |
| `Sgpt` | Alanine aminotransferase (SGPT/ALT) | numeric | IU/L | 10–2,000 | 0.0% |
| `Sgot` | Aspartate aminotransferase (SGOT/AST) | numeric | IU/L | 10–4,929 | 0.0% |
| `TP` | Total proteins | numeric | g/dL | 2.7–9.6 | 0.0% |
| `ALB` | Albumin | numeric | g/dL | 0.9–5.5 | 0.0% |
| `A/G Ratio` | Albumin / globulin ratio | numeric | ratio | 0.3–2.8 | 0.7% |
| `sex_male` | Sex is male | binary | 0/1 | 0–1 | 0.0% |

**Cleaning decisions.**

- **`Selector` inversion.** ILPD's raw target is `Selector`, coded
  **1 = liver patient, 2 = non-patient** — inverted relative to the usual
  convention where the larger code is the positive class. The cleaner maps
  `{1: 1, 2: 0}` explicitly and drops `Selector`. Taking the raw values at face
  value, or applying a `> 1` rule, produces a model that is confidently
  backwards, with a plausible-looking AUC.
- `Gender` (string `Male`/`Female`) is replaced by the binary `sex_male` and
  dropped.
- `A/G Ratio`'s 4 nulls (0.7%) are left as NaN, not imputed.

---

## diabetes

**Source.** UCI 891, CDC Diabetes Health Indicators, derived from the CDC
Behavioral Risk Factor Surveillance System 2015 survey. Public domain.
**Shape.** 253,680 × 22 (21 features + `target`). **No column has any missing
value** — this is a completed survey extract.

**Target.** `target` = 1 when the respondent reported having been told they have
diabetes or prediabetes (`Diabetes_binary`). 35,346 / 253,680 = 13.93%
positive. **Every value in this table, target and features alike, is a
self-reported telephone-survey response**, not a clinical record or a laboratory
result.

| Feature | Label | Type | Unit | Range | Missing |
|---|---|---|---|---|---|
| `HighBP` | Told they have high blood pressure | binary | 0/1 | 0–1 | 0.0% |
| `HighChol` | Told they have high cholesterol | binary | 0/1 | 0–1 | 0.0% |
| `CholCheck` | Cholesterol checked in the last 5 years | binary | 0/1 | 0–1 | 0.0% |
| `BMI` | Body mass index | numeric | kg/m² | 12–98 | 0.0% |
| `Smoker` | Smoked ≥ 100 cigarettes in their life | binary | 0/1 | 0–1 | 0.0% |
| `Stroke` | Told they have had a stroke | binary | 0/1 | 0–1 | 0.0% |
| `HeartDiseaseorAttack` | Coronary heart disease or myocardial infarction | binary | 0/1 | 0–1 | 0.0% |
| `PhysActivity` | Physical activity in the past 30 days | binary | 0/1 | 0–1 | 0.0% |
| `Fruits` | Eats fruit at least once a day | binary | 0/1 | 0–1 | 0.0% |
| `Veggies` | Eats vegetables at least once a day | binary | 0/1 | 0–1 | 0.0% |
| `HvyAlcoholConsump` | Heavy alcohol consumption | binary | 0/1 | 0–1 | 0.0% |
| `AnyHealthcare` | Has any health coverage | binary | 0/1 | 0–1 | 0.0% |
| `NoDocbcCost` | Could not see a doctor because of cost | binary | 0/1 | 0–1 | 0.0% |
| `GenHlth` | Self-rated general health (1 excellent … 5 poor) | ordinal | code | 1–5 | 0.0% |
| `MentHlth` | Days of poor mental health in the last 30 | numeric | days | 0–30 | 0.0% |
| `PhysHlth` | Days of poor physical health in the last 30 | numeric | days | 0–30 | 0.0% |
| `DiffWalk` | Serious difficulty walking or climbing stairs | binary | 0/1 | 0–1 | 0.0% |
| `Sex` | Sex (1 = male, 0 = female) | binary | 0/1 | 0–1 | 0.0% |
| `Age` | BRFSS age bracket (1 = 18–24 … 13 = 80+) | ordinal | code | 1–13 | 0.0% |
| `Education` | Education level (1 lowest … 6 highest) | ordinal | code | 1–6 | 0.0% |
| `Income` | Income bracket (1 lowest … 8 highest) | ordinal | code | 1–8 | 0.0% |

`Age` is a bracket code, **not a number of years** — reading `Age = 9` as nine
years old is the obvious way to misuse this file.

**Cleaning decisions.** Map `Diabetes_binary` to `target` and drop it. Nothing
else: the extract arrives complete and integer-coded, so there is no missingness
to handle and no sentinel to convert. The self-report caveat is a labelling
limitation, not something cleaning can fix, and it is carried into
`MODEL_CARD.md` and the honesty banner instead.

---

## Cross-cutting notes

**Screening-outcome columns are dropped as leakage, not kept as features.** The
clearest case is cervical cancer's `Hinselmann` / `Schiller` / `Citology`, which
are alternative readings of the same screening episode that produced the
`Biopsy` target. The general rule this project applies: if a column could only
have been recorded *because* the outcome was already being determined, it is not
available at the moment a screening decision is made, and including it inflates
every metric while making the model useless for its stated purpose.

**Sentinel-coded missingness is converted, not modelled.** Heart disease's
`chol == 0` at Switzerland and VA is the case this project was built around: a
value that is in-range for the data type, impossible for the quantity, and
therefore invisible to any check that only looks for NaN. `nightingale.sentinel`
scans for it statistically and the cleaner converts it, site by site.

**Provenance.** All six cleaned files are listed in `provenance/manifest.json`
with their SHA-256 hashes, covered by the Ed25519 signature. `nightingale
verify` recomputes them from disk. Re-running `nightingale clean` reproduces
each file byte for byte, so a regenerated checkout still verifies.
