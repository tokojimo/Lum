# LuxPlate Analyzer

LuxPlate Analyzer is a local Python/Streamlit application for processing,
quality-controlling, visualizing and statistically analyzing bacterial growth
and luminescence plate-reader experiments. It is initially designed around
Varioskan exports while keeping import and scientific processing extensible.

> **Status:** audit-first scaffold. No legacy scripts or example workbooks were
> present in the initial repository, so format parsing and historical numerical
> equivalence are deliberately not claimed yet. See the [audit](docs/legacy_pipeline.md).

## Features and roadmap

The planned workflow includes multi-Excel/CSV import, Varioskan parsing,
duplicate detection, experimental-design validation, multiple biological
replicates per file, metadata harmonization, QC/outlier review, explicit blank
correction, luminescence/OD normalization, kinetic extraction, a Figure
Builder, biological-replicate-aware statistics, PNG/SVG/PDF export, and
reloadable analysis configurations. The current scaffold establishes the data
model, module boundaries, replicate-safe primitives, tests, and CI.

## Installation

```bash
git clone <repository>
cd luxplate-analyzer
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS, then:

```bash
pip install -r requirements.txt
```

## Launch

```bash
streamlit run app.py
```

The present UI is intentionally a status screen; the full interface follows
the format and legacy-behavior audit rather than guessing scientific rules.

## Workflow

1. Import
2. Validation
3. Experimental design
4. QC
5. Processing
6. Figure Builder
7. Statistics
8. Export

## Experimental design

**One file is NOT necessarily one biological replicate.** A workbook can hold
Rep1 and Rep2 while another holds Rep3 (N=3). Conversely, three files may all
belong to Rep1 (N=1). File, plate, well, group, and technical-well counts never
define biological N.

## Biological vs technical replicates

**Three biological replicates with three technical wells each correspond to
N=3, not N=9.** Technical wells are summarized inside each biological replicate
before biological summaries or inference. N is reported per comparison.

## Duplicate detection

The planned layers are SHA-256 binary identity, a canonical scientific
fingerprint, component-wise partial similarity, and internal measurement-key
checks. Findings are reported for review: files are not merged, conflicting
values are not averaged, and duplicates are not silently deleted.

## Metadata harmonization

Every label moves through `original label → normalized proposal → user
validation`. The original is immutable. Alias confidence is advisory, and
experiment-specific groups never receive universal meanings.

## Data processing and normalization

```text
Raw → QC → Blank correction → DO threshold
    → Luminescence normalization → Kinetic parameters
```

After an explicitly validated blank association, normalization is
`Lum_norm = Lum_corr / DO_corr`. The default effective OD threshold is
`max(mean(blank DO_corr) + 3 × SD(blank DO_corr), 0.05)`, with three consecutive
valid points. These parameters will remain visible and configurable. Missing
values are never replaced by zero.

## Statistical analysis

The independent unit is `biological_replicate_id`, not a technical well.
Analyses target per-biological-replicate kinetic metrics by default. Paired vs
independent design must be declared; the test, assumptions, raw/adjusted p,
effect size, confidence interval where available, correction (Holm by default
for suitable pairwise families), and actual N are reported. N<2 remains
descriptive, never rescued through pseudoreplication.

## Figure Builder

The planned builder filters biological replicates, strains, media, conditions,
and metrics without rerunning processing. It supports individual curves or
biological mean ± SD/SEM, faceting, dual OD/luminescence axes, stable editable
colors, and parameter plots exposing each biological replicate.

## Export formats

Figures are saved directly from matplotlib as PNG (150/300/600 dpi), vector
SVG, and vector PDF. Figure points/summaries use CSV/XLSX; complete traceable
archives use ZIP; configurations use JSON.

## Reproducibility

Raw data remain immutable, exclusions and metadata corrections are logged, and
configs are reloadable. Exports link source hashes through wells, technical and
biological replicates, transformations, figures, and statistics.

## Testing

```bash
pytest
```

Initial synthetic tests pin kinetic primitives and the critical biological-N
rules. Real regression fixtures must be anonymized before commit.

## Project structure

```text
app.py                    Streamlit entry point
luxplate/                 UI-independent scientific library
docs/                     audit and data-model decisions
tests/                    public synthetic regression tests
legacy_scripts/           historical sources when supplied
pages/                    future Streamlit workflow pages
.github/workflows/        Python 3.12 pytest CI
```

## Privacy

LuxPlate Analyzer runs locally. Experimental datasets are not uploaded to an
external service. Spreadsheet and CSV patterns are ignored by default; only
explicit synthetic fixture directories may be committed.

## Limitations

- No representative Varioskan workbook was available in the initial checkout,
  so supported sheet variants remain to be established from fixtures.
- The scaffold does not yet expose the processing/statistics UI.
- Ambiguous sheets, metadata, aliases, groups, duplicates, and blanks always
  require review: **warn, don't guess**.

## Contributing

Create a focused branch, add synthetic tests for every scientific behavior,
run `pytest`, and document numerical or schema changes. Never commit private
experimental data or silently change legacy behavior.

## License

LuxPlate Analyzer is available under the [MIT License](LICENSE).

