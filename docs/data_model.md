# Traceable data model

## Principles

Raw observations are immutable. Original, machine-normalized, and
user-validated metadata are separate fields. A source file is provenance—not a
biological replicate. Ambiguity remains an explicit review state.

## Hierarchy

```text
experiment_id
  └─ plate_id (one or more source files may contribute)
      └─ biological_replicate_id (may span files/plates)
          └─ condition_id
              └─ technical_replicate_id / well
                  └─ observed time point and measurement
```

Neither direction implies a one-to-one relationship: one file may contain
multiple biological replicates, and one biological replicate may span files.

## Observation table

The canonical long table has one row per observed well/time/reading. Required
provenance and identity fields include `source_file`, `source_sha256`,
`plate_id`, `experiment_id`, `biological_replicate_id`,
`technical_replicate_id`, `condition_id`, `strain_id`, `medium_id`, `group_id`,
`puits`, `temps_h`, and `lecture`. Measurement fields include `DO_brute`,
`Lum_brute`, `DO_corr`, `Lum_corr`, and `Lum_norm`.

The internal uniqueness candidate is `(plate_id, biological_replicate_id,
puits, temps_h, lecture)`. Repeated equal values are flagged as duplicates;
repeated unequal values are conflicts. Neither is averaged automatically.

## Metadata lifecycle

Each semantic dimension follows the same three-stage pattern:

| Dimension | Immutable source | Machine proposal | User-approved value |
|---|---|---|---|
| strain | `strain_original` | `strain_normalized` | `strain_display_name`, `strain_id` |
| medium | `medium_original` | `medium_normalized` | `medium_display_name`, `medium_id` |
| condition | `condition_original` | `condition_normalized` | `condition_display_name`, `condition_id` |
| group | `group_original` | `group_normalized` | `group_validated`, `group_id` |

Mappings have `type`, `original`, `normalized`, `decision`, user/time provenance,
and configuration version. Group meaning is experiment-scoped, never global.

## Replicate semantics

`biological_replicate_id` is the independent unit and the sole source of
biological N. `technical_replicate_id` identifies repeated wells/series within
that unit. For multi-biological summaries, technical observations are first
averaged per biological replicate at each observed time; biological values are
then summarized. N is computed per condition/comparison, so missing conditions
reduce only the relevant N.

## Immutable processing stages

```text
RAW_DATA → VALIDATED_METADATA → FILTERED_DATA → BLANK_CORRECTED_DATA
         → NORMALIZED_DATA → KINETIC_PARAMETERS → FIGURE_DATA
```

Each stage is a new dataset with parent identifier, configuration hash,
software version, timestamp, and decision-log references. Missing values remain
missing. Interpolation is off by default and, when requested, produces marked
derived rows rather than overwriting observations.

## Review and configuration entities

- import issues: severity, file, code, human-readable message, decision;
- duplicate comparisons: hash/similarity components and explicit resolution;
- exclusions: point/series scope, reason, proposal, final decision;
- blank associations: condition, biological replicate, group, blank series,
  validation state;
- processing settings: OD minimum, blank SD multiplier, consecutive points;
- figure/statistics configurations: exact selections, units, N, styles, tests,
  pairing, corrections, version, and source stage.

## Validation invariants

- Wells conform to A1–H12 (canonicalized without losing the source spelling).
- Time is finite, nonnegative, and monotone within a series; differing grids are
  warned about but preserved.
- No inferred blank, alias, exclusion, merge, interpolation, or conflict average
  is applied without an explicit recorded decision.
- Inferential statistics are unavailable below two independent biological IDs.

