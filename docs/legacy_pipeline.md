# Historical pipeline audit

## Audit scope and inventory

> Update, 10 August 2026: the historical Python scripts have now been supplied
> under `examples/synthetic/`. The inventory below records the initial audit
> only. Integration started with `01_mise_en_forme_donnees.py`; its workbook
> parsing and raw preview now live in `luxplate.varioskan` and the Streamlit UI.

The repository was inspected recursively on 10 August 2026. At audit time it
contained only Git administrative files and `.gitkeep`: **no Python script,
Excel workbook, CSV export, image, or historical result was present**. There is
therefore no scientific implementation to reverse-engineer yet and no claim of
legacy equivalence can responsibly be made.

| Asset class | Observed | Auditable behavior |
|---|---:|---|
| Python scripts | 0 | None |
| Excel workbooks (`.xlsx`, `.xls`, `.xlsm`) | 0 | None |
| CSV datasets | 0 | None |
| Plate layouts | 0 | None |
| Historical expected outputs | 0 | None |

This absence is a blocking input for a *precise* historical reconstruction,
but not a reason to guess. Private files should remain untracked and can be
placed in a locally ignored location for a later audit. An anonymized synthetic
fixture may be committed under `tests/data/synthetic/`.

## Provisional pipeline reconstructed from the specification

This is a requirements map, **not an assertion about legacy code**:

1. Read Varioskan kinetic workbooks or endpoint plates.
2. Detect absorbance, luminescence, and plate-plan sheets; require user choice
   when more than one luminescence sheet is plausible.
3. Convert observations to immutable long-form raw data.
4. detect binary, scientific, partial, and internal duplicates without merging.
5. Retain original metadata, create machine proposals, and require validation.
6. Assign experiment, plate, biological-replicate, and technical-replicate IDs.
7. Record QC/outlier decisions without silently deleting observations.
8. Associate validated blanks within the experimental design and subtract them.
9. Apply an effective OD threshold and calculate `Lum_corr / DO_corr` only in
   the valid observed interval.
10. Extract kinetic metrics per technical curve, then summarize technical wells
    within each biological replicate.
11. Render figures and perform inference using biological replicates as N.
12. Export data, decisions, configurations, hashes, figures, and reports.

## Legacy-script review template

When scripts are supplied, each receives a section containing: name; purpose;
inputs; outputs; calculations and numerical conventions; important functions;
assumptions; duplicated code; hardcoded groups, strains, colors, paths, or time
grids; behavior differences; and generalization candidates. Each formula will
be paired with a fixture and `numpy.testing.assert_allclose` regression test.

## Risks to investigate explicitly

- file count or plate count used as biological N;
- global meanings assigned to experiment-specific group labels;
- silent choice among luminescence sheets;
- missing values converted to zero or time grids interpolated implicitly;
- pooled technical wells used as independent observations;
- conflicting measurements averaged or duplicates removed silently;
- blanks selected by position or group without validated experimental context;
- hardcoded strains, media, colors, paths, endpoints, or baseline windows;
- inconsistent trapezoid, peak-tie, final-value, and doubling-time definitions.

## Scientific behavior-change policy

If a suspected defect is found, preserve and document the current result, add a
test reproducing it, explain the scientific concern, propose a correction, and
offer explicit `legacy` and `corrected` modes where compatibility demands it.
Historical behavior is never changed silently.
