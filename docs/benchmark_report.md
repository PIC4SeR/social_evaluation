# Benchmark Reports

`benchmark_report` takes the per-run metrics produced by `metrics_evaluation`
and turns them into benchmark artifacts for one or more scenarios: LaTeX
summary tables, success-rate tables and plots, box/violin distribution plots,
proxemic-intrusion plots, statistical tests (omnibus + post-hoc + success-rate
permutation test), pairwise heatmaps, rliable plots, and LaTeX report driver
files that pull everything together.

```bash
benchmark_report -h
benchmark_report --config config/report_config.yaml --base-output-dir results/report
```

Contents:

- [Command line](#command-line)
- [Input data](#input-data)
- [Configuration file](#configuration-file)
  - [`benchmark` section](#benchmark-section)
  - [Scenarios, episodes and episode groups](#scenarios-episodes-and-episode-groups)
  - [Metric groups](#metric-groups)
  - [Aggregate metrics](#aggregate-metrics)
  - [Intrusion labels](#intrusion-labels)
  - [Failed runs](#failed-runs)
- [Outputs](#outputs)
  - [Tables](#tables)
  - [Statistics](#statistics)
  - [Plots](#plots)
  - [rliable plots](#rliable-plots)
  - [LaTeX report drivers](#latex-report-drivers)
- [Output layout](#output-layout)
- [How the numbers are computed](#how-the-numbers-are-computed)
- [Python API](#python-api)
- [Pitfalls](#pitfalls)

## Command line

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | `report_config.yaml` (current directory) | YAML configuration file. |
| `--base-output-dir` | directory containing the config file | Where the per-scenario report directories are written. |

For each scenario the command runs, in this order:

1. table exports (success-rate tables, metric tables, statistics CSVs);
2. plots (success rate, distributions, proxemics, heatmaps, rliable);
3. LaTeX report drivers.

Progress is printed with a `[benchmark-report]` prefix. Plot failures that are
expected on small datasets (no significant pairs for a heatmap, a missing
baseline for probability of improvement) are logged and skipped. Other errors
stop the run.

## Input data

`root_dir` is scanned **recursively** for `*.pkl`, `*.pickle` and `*.csv`
files, and all of them are concatenated into one run-level table (one row per
run). These are the files written by `metrics_evaluation` or by the experiment
harnesses.

Columns the loader understands:

| Column | Required | Notes |
| --- | --- | --- |
| `experiment_tag` or `experiment` | yes | Episode key. `experiment_tag` wins when both exist. |
| `algorithm` or `navigator` | yes | Navigator key. For CSV files the key is taken from the file name (`<algo>_outcomes.csv` → `<algo>`). |
| `run` or `run_id` | no | Run identifier. Without it the row index is used. rliable plots pair runs by this id. |
| `result` | no | Run outcome. Without it, `completed` is mapped to `Success`/`Failure`; without either, every run counts as `Success`. |
| metric columns | – | Any numeric column referenced in `metric_groups` or `intrusion_labels`. |

`result` values are classified as follows: exactly `Success` counts as a
success, and strings containing `Timeout`, `Collision` or `Navigation failure`
count as those failure types. Anything else (for example `Failure`) is a
failure that matches none of the stacked categories in the success-rate plot.

**Per-environment episode keys.** When files live in a
`<root_dir>/<algorithm>/<environment>/...` layout and the episode tag has no
`/`, the loader rewrites the episode key to `<environment>/<episode>`. This lets
the same episode name in different environments be told apart, so in that
layout the YAML must refer to episodes as `<environment>/<episode>`. Flat
layouts are left unchanged.

**Derived columns.** If the social, personal and intimate space intrusion
columns are all present, `public_space_intrusions = 100 − (social + personal +
intimate)` is added. The same is done for the `group_*` variants.

Only rows whose episode is in the scenario's episodes **and** whose navigator is
in `navigators` are kept. If none match, the run stops with an error listing the
episode keys and navigators that do exist in `root_dir`. Configured episodes or
navigators without any run are reported as warnings. The rows are then
relabelled with `episode_names` and `navigator_names`, and every later step
works with these display labels.

## Configuration file

The YAML file has two top-level mappings, `benchmark` (what to evaluate) and
`outputs` (what to produce). `config/report_config.yaml` is a working example.

### `benchmark` section

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `root_dir` | yes | – | Metrics directory. A relative path is resolved against the base output directory (see [Pitfalls](#pitfalls)). |
| `scenario` | yes | – | One scenario entry or a list of them. Each scenario gets its own report directory. |
| `navigators` | yes | – | Navigator keys to include, in display order. Can be overridden per scenario. |
| `navigator_names` | yes | – | `key: display label`. A missing entry falls back to the raw key (a warning is logged). Labels may contain LaTeX. |
| `metric_groups` | yes | – | See [Metric groups](#metric-groups). |
| `intrusion_labels` | yes | – | See [Intrusion labels](#intrusion-labels). Can be `{}`. |
| `episodes` | no | `[]` | Episode keys shown as individual rows/panels. Can be overridden per scenario. |
| `episode_names` | no | derived | `key: display label`. A missing entry becomes the key title-cased with `_` → space (`episode_18` → `Episode 18`). |
| `episode_groups` | no | `{}` | `group label: [episode keys]`. `experiment_groups` is accepted as an alias. |
| `metric_captions` | no | `{}` | `group key: caption` for the LaTeX metric tables. |
| `discard_failed_metrics` | no | `true` | Drop failed runs from metric summaries. See [Failed runs](#failed-runs). |
| `alpha` | no | `0.8` | Fill transparency for bars and boxes. |
| `use_tex` | no | `true` | Render plot text with LaTeX (needs a TeX installation). With `false`, LaTeX markup is stripped from labels. |
| `intrusion_cmap` | no | `Paired` | Colormap for the outcome and intrusion categories. |

### Scenarios, episodes and episode groups

A scenario entry is either a plain name or a mapping that overrides the global
settings:

```yaml
benchmark:
  scenario:
    - simple_scenario               # uses the global episodes/navigators
    - name: mixed
      episodes: [episode_1, episode_2]
      navigators: [ORCA, SARL]
      episode_groups:
        Overall: [episode_1, episode_2, episode_13]
      discard_failed_metrics: false
```

Scenario names must be unique: they become directory names. Duplicate episodes
or navigators inside a scenario are rejected.

An **episode group** pools several episodes into one extra row (tables) or
panel (plots), placed after the individual episodes. Members can be given as
episode keys or display names. A key that is not in `episodes` is still loaded
but gets no row of its own.

What is displayed depends on which of the two keys you set:

| `episodes` | `episode_groups` | Rows / panels |
| --- | --- | --- |
| set | not set | one per episode |
| set | set | one per episode, then one per group |
| not set | set | one per group only (groups-only mode) |

The success-rate LaTeX tables (`export_success_rate_tables`) and the rliable
plots always work per episode and ignore groups.

### Metric groups

Metric groups organise the metrics into tables and figures. Every group
produces its own summary table and its own distribution figure.

```yaml
benchmark:
  metric_groups:
    metrics_nav:
      time_to_reach_goal:
        name: '\\textbf{TTG (s)}'
        direction: DOWN
        discard_failed: true
      avg_robot_linear_speed:
        name: '\\textbf{ARV (m/s)}'
        direction: UP
    metrics_social:
      social_work_per_second:
        name: '$\\mathbf{SW_{step}}$'
        direction: DOWN
        discard_failed: false
      avg_distance_to_closest_person:
        name: '\\textbf{AMD (m)}'
        direction: UP
      plot_metrics:                 # optional: subset shown in distribution plots
        - social_work_per_second
```

Per-metric keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | metric key | Display label. An arrow (↑ for `UP`, ↓ for `DOWN`) is appended automatically. A `_` not followed by `{` becomes a space, so keep LaTeX subscripts braced (`SW_{step}`). |
| `direction` | `UP` | Which way is better. Used for table colouring, aggregate scores, rliable normalisation and success-weighted penalties. |
| `discard_failed` | global setting | Per-metric override for dropping failed runs. |

Reserved keys inside a group:

- `plot_metrics`: list of metric keys from the group to show in the
  distribution plots. Tables still list every metric.
- `aggregate_metric`: a derived score, see below.

### Aggregate metrics

`aggregate_metric` adds one more column to the group: a per-run weighted score
in `[0, 1]` built from the group's metrics.

```yaml
metrics_social:
  # ... metrics ...
  aggregate_metric:
    enabled: true
    key: social_score               # default: <group>_aggregate
    name: '\\textbf{Social score}'  # default: "<Group> Aggregate" in bold
    direction: UP                   # UP: 1 is best; DOWN reports 1 - score
    metrics: [social_work_per_second, avg_distance_to_closest_person]  # default: all metrics in the group
    weights: {social_work_per_second: 2.0}  # default weight 1.0
    normalize: minmax               # minmax (default) or none
    limits: [0.0, 1.0]              # or {metric_key: [lo, hi], default: [lo, hi]}
    require_complete: true          # NaN score if any source metric is missing
    exclude_metrics_by_experiment:  # leave metrics out for some episodes/groups
      Open Crowded: [avg_distance_to_closest_person]
    plot_only: false                # true: plot only the aggregate for this group
```

For each source metric, a `DOWN` metric is first negated so that larger is
better. With `minmax`, values are then scaled into `[0, 1]` using the
per-episode min/max over all navigators (or the fixed `limits`) and clipped.
The aggregate is the weighted mean of these normalised values. Failed-run
filtering follows each source metric's own setting. A warning is logged when
source metrics are missing or the aggregate ends up with no rows.

### Intrusion labels

`intrusion_labels` lists the columns stacked in the proxemic-intrusion plot,
from the closest zone to the farthest. Two forms are accepted:

```yaml
intrusion_labels:
  intimate_space_intrusions: Intimate Space Intrusions    # simple form
  personal_space_intrusions:
    name: Personal Space Intrusions                       # extended form
    discard_failed: false
```

### Failed runs

A run is *failed* when `result != "Success"`. Whether failed runs count in a
metric's statistics is decided by the first of these that is set:

1. `discard_failed` on the metric (in `metric_groups` or `intrusion_labels`);
2. `discard_failed_metrics` on the scenario entry;
3. global `benchmark.discard_failed_metrics` (default `true`).

Keep failed runs for metrics that are meaningful on a partial trajectory
(clearance, social work, intrusions). Drop them for metrics that assume the
goal was reached (time to goal, path length).

Success rates and the success-rate test always use every run.

## Outputs

Every key under `outputs` is optional and defaults to off. A key is turned on
with `true`, or with a mapping of options. A mapping is enabled unless it
contains `enabled: false`.

`clean` (default `true`) is the exception: before anything is generated, it
deletes the scenario's `tables/` and `plots/` directories, so the report
contains only what this run produced. Deletion happens after the data has loaded,
so a configuration error leaves the previous report in place. Set
`clean: false` to keep earlier files, for example when you build a report over
several runs that each enable different outputs.

### Tables

#### `export_success_rate_tables`

`tables/success_rate_by_navigator1.tex` and `...2.tex`: goal-reached
percentage per episode × navigator. The episodes are split into two tables of
equal size. This output takes no options and ignores episode groups (use
`export_metric_tables`, which has a success-rate column, if you need groups).

#### `export_metric_tables`

```yaml
export_metric_tables:
  enabled: true
  groups: [metrics_social]  # default: every metric group
  scale: 0.71               # \scalebox factor of the table
  show_std: true            # print "mean ± std" (false: mean only)
  cmap_name: viridis        # colour cells by rank within each episode; omit for no colouring
  weighted: false           # also write the success-weighted variant
```

For each group this writes `tables/<group>_summary_by_navigator.tex`: one
multirow block per episode (then per episode group), one row per navigator, a
success-rate column, then one column per metric. With `cmap_name`, cells are
shaded according to `direction` and a colour bar is saved to
`tables/<group>_colormap.pdf`.

With `weighted: true` the tool also writes
`tables/<group>_weighted_summary_by_navigator.tex`, in which failures are
penalised:

```
weighted_mean = SR · mean + (1 − SR) · worst
weighted_std  = SR · std
```

Here `SR` is the success fraction of that navigator on that episode, and
`worst` is the worst value of the metric on the episode across all navigators
and all runs (the maximum for `DOWN`, the minimum for `UP`).

### Statistics

The statistical tests are configured once, in `outputs.statistics`. The same
settings are used by the CSV export, the stars on the distribution plots, the
pairwise heatmaps and the LaTeX statistics section. `export_anova_analysis` is
an older name for this block and is still accepted.

```yaml
statistics:
  enabled: true
  export: true            # false: configure the tests without writing the CSVs
  output_subdir: null     # CSV folder under tables/ (default: tables/stats)
  test: auto              # auto | anova | welch | kruskal | permutation
  posthoc: null           # tukey | tamhane | dunn | conover | mannwhitney | ttest
  p_adjust: holm          # correction of the pairwise p-values (statsmodels name)
  correct_omnibus: holm   # correction across all omnibus tests; "none" to skip
  alpha: 0.05
  min_group_size: 2       # navigators with fewer runs are left out of a test
  metrics: null           # default: every configured metric
  pairs: null             # [[A, B], ...] pairs listed in the LaTeX post-hoc table
  permutations: 9999      # label shuffles for test: permutation
  success_test: true      # false, or {n_resamples: 9999, seed: 0}
  seed: 0
```

Even when `enabled: false`, the settings in this block still apply to the
plots and heatmaps that use the tests. Without the block, these defaults are
used.

**Procedure.** The tests are run for each pair of (episode or episode group,
metric):

1. Keep the navigators that have at least `min_group_size` valid runs. If fewer
   than two remain, the cell is marked untestable and a `note` is recorded.
2. Run the **omnibus test**:
   - `auto`: a median-centred Levene test chooses `welch` (p ≤ `alpha`, unequal
     variances) or `anova`;
   - `anova` / `welch`: one-way F-test; effect size η²;
   - `kruskal`: Kruskal–Wallis; effect size η²_H = (H − k + 1)/(n − k);
   - `permutation`: Kruskal–Wallis H with its p-value from shuffling the
     navigator labels. Use this when each navigator has few runs (about 5),
     because the χ² approximation is unreliable there.
3. Correct the omnibus p-values across **all** tested cells (`correct_omnibus`).
4. Run the **post-hoc** test only where the corrected omnibus p ≤ `alpha`.
   The default post-hoc depends on the omnibus test: anova → Tukey, welch →
   Tamhane, kruskal/permutation → Dunn. Tukey and Tamhane are adjusted by
   construction; the others use `p_adjust`. Each pair also gets the mean
   difference (a − b), Hedges' g, and both medians.
5. **Success-rate test** (per episode/group): permutation χ² test of
   independence between navigator and success, with p-values corrected
   with `correct_omnibus`.

Files written to `tables/stats/`:

| File | Content |
| --- | --- |
| `stats_summary.csv` | One row per (episode_group, metric): test, statistic, df1, df2, p_value, p_adjusted, effect_size, effect_name, n, k, significant, note. |
| `stats_pairwise.csv` | One row per pair compared post hoc: posthoc, p_adjusted, reject, mean_diff, hedges_g, median_a, median_b, and `listed` (whether the pair appears in the LaTeX table: the configured `pairs`, otherwise the significant pairs). |
| `stats_success.csv` | Per episode/group: n, successes, chi2, p_value, p_adjusted. |
| `stats_runs.csv` | The run-level data that was tested, for re-analysis. |

### Plots

All plots are written as PNG files under `plots/`. Several plots accept
`font_sizes` with the keys `title`, `axis_label`, `tick_label`, `legend`,
`legend_ncol`, and (where relevant) `sup_label` and `xtick_label`.

#### `plot_success_rate`

`plots/success_rate_by_navigator.png`: one stacked bar chart per episode and
group, splitting the runs into success / timeout / collision / navigation
failure.

| Option | Default |
| --- | --- |
| `n_cols` | `3` panels per row |
| `font_sizes` | 37 everywhere, `legend_ncol: 4` |

#### `plot_metric_distributions`

Takes a **list** of entries, one figure series per entry (for example one box
and one violin). `true` alone draws box plots with the default options.

```yaml
plot_metric_distributions:
  - kind: box                     # box | violin
    n_rows: 1                     # episodes per figure; more → _page<N> files (0: all)
    n_cols: 2                     # metric panels per row (0: all in one row)
    show_episode_ylabel: true     # episode name as y-label of the first column
    connect_medians: false        # join navigator medians (box only)
    font_sizes: {title: 40, axis_label: 40, tick_label: 40, legend: 40, legend_ncol: 4}
    # statistical annotations
    annotate_stats: true
    stat_annotation_mode: baseline_stars   # baseline_stars | brackets
    stat_baseline: 'SFM-NMPC'              # display label of the reference navigator
    highlight_baseline: true               # bold legend entry + label above it
    highlight_baseline_label: 'Ours'
    stat_use_statistics: true              # use the tests of the `statistics` block
    stat_alpha: 0.05
    stat_min_group_size: 2
    stat_font_size: 30
    stat_line_height: 0.04
```

Output: `plots/<group>_<kind>plots_by_navigator[_page<N>].png`, one figure
series per metric group (filtered by `plot_metrics` when set).

Annotation modes:

- `baseline_stars`: draws stars above every navigator that differs
  significantly from `stat_baseline` (`*` ≤ 0.05, `**` ≤ 0.01, `***` ≤ 0.001,
  `****` ≤ 0.0001). A `<figure>.caption.txt` with the test details is saved
  next to each figure.
- `brackets`: draws pairwise brackets with `statannotations`. This package is
  not installed with this one (`pip install statannotations`).

With `stat_use_statistics: true` (the default), the stars and brackets come
from the [statistics](#statistics) block, whatever its configuration: the omnibus
test set by `statistics.test` (`auto`, `anova`, `welch`, `kruskal` or
`permutation`), then the matching post-hoc (Tukey, Tamhane, Dunn, or
`statistics.posthoc`). Only pairs in cells where the corrected omnibus test is
significant get a star or bracket, and the post-hoc's adjusted p-value is the
one shown. `stat_alpha` and `stat_min_group_size` take precedence over the
block's `alpha` and `min_group_size`. With `false`,
`statannotations` runs its own test, controlled by `stat_test` (`t-test_ind`,
`Mann-Whitney`, `Kruskal`, ...), `stat_correction` (default `bonferroni`,
`null` for none), `stat_comparisons` (`[[A, B], ...]`, default all pairs) and
`stat_display_format` (`star` or `simple`).

`stat_use_anova` is the older name of `stat_use_statistics`. Configs that use
it still work.

#### `plot_proxemic_intrusions`

`plots/proxemic_distances_intrusions.png`: stacked bars of the mean
`intrusion_labels` percentages per navigator, one panel per episode and group.

| Option | Default |
| --- | --- |
| `n_rows` | `2` |
| `font_sizes` | 60 everywhere, `xtick_label: 40`, `legend_ncol: 4` |

#### Pairwise heatmaps

```yaml
plot_pairwise_pvalue_heatmap:      # adjusted p-values, shown as stars
  enabled: true
  metrics: [time_to_reach_goal]    # default: statistics.metrics, else all
plot_pairwise_mean_diff_heatmap: true   # mean(row) − mean(column)
plot_pairwise_hedges_heatmap: true      # Hedges' g (row vs column)
```

Each heatmap type produces one PNG per (episode/group, metric) whose omnibus
test was significant:
`plots/stats/<group>/<metric>_{pairwise_pvalues|mean_diff|hedges_g}.png`.

Options: `metrics`, `alpha`, `min_group_size`, `output_subdir` (under
`plots/`), `cmap` (defaults `mako` / `coolwarm` / `coolwarm`), `annotate`
(`true`), `fmt` (`.2f`), `label_size`, `title_size`, `colorbar_label_size`,
`annot_font_size`. If no cell is significant, the heatmap is skipped and a
message is logged.

### rliable plots

These plots use [rliable](https://github.com/google-research/rliable). Each
metric is turned into one score matrix per navigator, with runs as rows and
episodes as columns. Runs are matched across episodes by `run` id, and a run
missing from any episode is dropped. With `normalize: true`, each episode is
min-max scaled over all navigators, and `DOWN` metrics are flipped so that 1 is
always best. Episode groups are not used here.

#### `plot_rliable_interval_estimates`

`plots/rliable/<metric>_interval_estimates.png`: median, IQM, mean and
optimality gap with stratified-bootstrap confidence intervals.

| Option | Default |
| --- | --- |
| `metrics` | every configured metric |
| `normalize` | `true` |
| `reps` | `5000` |

#### `plot_metric_performance_profiles`

Score-distribution performance profiles (fraction of runs with score > τ).

| Option | Default |
| --- | --- |
| `groups` | every metric group (when `metrics` is not given either) |
| `metrics` | extra custom list → `selected_metrics_performance_profiles.png` |
| `normalize` | `true` |
| `reps` | `2000` |
| `tau` | 201 points spanning the observed scores ± 5 % |
| `use_average_distribution` | `false` |
| `legend_columns` | `4` |
| `figsize_scale` | `6.5` |
| `output_subdir` | `plots/rliable` |

Output: `<output_subdir>/<group>_performance_profiles.png`.

#### `plot_probability_of_improvement`

`plots/rliable/<metric>_probability_of_improvement.png`: P(navigator > baseline)
with confidence intervals, for each navigator against the baseline.

| Option | Default |
| --- | --- |
| `baseline_algorithm` | **required**; raw key or display label |
| `metrics` | **required**, non-empty list |
| `normalize` | `true` |
| `reps` | `2000` |

### LaTeX report drivers

```yaml
generate_latex_reports:
  enabled: true
  prefix: paper_report   # default: name of the base output directory
  overwrite: true        # default false: an existing file makes the scenario be skipped
  weighted: false        # also write the success-weighted document
```

This step writes `<scenario_dir>/<prefix>_<scenario>.tex`, a standalone
document assembled from whatever is in the scenario's `tables/` and `plots/`
directories:

1. per metric group: the summary table, followed by its box and violin figures
   (all pages);
2. the success-rate and proxemic figures;
3. a *Statistical tests* section built from `stats_summary.csv`: a summary
   sentence, then longtables of the significant omnibus tests, the listed
   post-hoc pairs and the success-rate test, followed by the p-value heatmaps.

With `weighted: true` (and the weighted tables exported), the step also writes
`<prefix>_<scenario>_weighted.tex`. It contains the weighted tables, the same
figures, and `\input`s for `rliable_interval_estimates_nav.tex`,
`rliable_performance_profiles.tex` and `rliable_prob_of_improvement.tex` when
those files exist in the scenario directory. This tool does not write them;
you add them by hand.

Compile the document from inside the scenario directory, because all asset
paths are relative to it:

```bash
cd results/report/episode18_annotated
pdflatex paper_report_episode18_annotated.tex
```

## Output layout

```
<base-output-dir>/
└── <scenario>/
    ├── <prefix>_<scenario>.tex
    ├── <prefix>_<scenario>_weighted.tex          (weighted: true)
    ├── tables/
    │   ├── success_rate_by_navigator{1,2}.tex
    │   ├── <group>_summary_by_navigator.tex
    │   ├── <group>_weighted_summary_by_navigator.tex
    │   ├── <group>_colormap.pdf
    │   └── stats/
    │       └── stats_{summary,pairwise,success,runs}.csv
    └── plots/
        ├── success_rate_by_navigator.png
        ├── <group>_{box,violin}plots_by_navigator[_page<N>].png
        ├── <group>_{box,violin}plots_by_navigator[_page<N>].caption.txt
        ├── proxemic_distances_intrusions.png
        ├── stats/<episode_group>/<metric>_{pairwise_pvalues,mean_diff,hedges_g}.png
        └── rliable/
            ├── <metric>_interval_estimates.png
            ├── <metric>_probability_of_improvement.png
            └── <group>_performance_profiles.png
```

## How the numbers are computed

| Quantity | Per episode | Per episode group |
| --- | --- | --- |
| Success rate | successful runs / all runs × 100 | successes / runs, pooled over every member episode |
| Metric mean | mean over the kept runs | mean of the per-episode means (each episode weighs the same) |
| Metric std | std over the kept runs | √(mean_std² + std_of_means²): mean of the per-episode stds (within-episode) combined with the std of the per-episode means (between-episode) |
| Statistics / distribution plots | runs of that episode | all runs of the member episodes, pooled |

"Kept runs" means after the [failed-run filtering](#failed-runs) for that
metric.

## Python API

The CLI is a thin wrapper around the classes in `social_evaluation.benchmark_report`:

```python
from pathlib import Path
from social_evaluation.benchmark_report import BenchmarkReport, StatsSettings
from social_evaluation.benchmark_report_ import _build_benchmark_config, _load_yaml_config

raw = _load_yaml_config(Path("config/report_config.yaml"))
cfg = _build_benchmark_config(raw["benchmark"], Path("results/report"),
                              raw["benchmark"]["scenario"][0])
report = BenchmarkReport(cfg)
report.stats_settings = StatsSettings(test="permutation")

report.metrics_results          # run-level DataFrame after loading/relabelling
report.success_rate             # success-rate table (episodes + groups)
summary, pairwise = report.analyze_algorithm_effects(["time_to_reach_goal"])
report.export_metric_tables(cmap_name="viridis")
report.plot_metric_distributions(kind="box")
```

Tests covering this module: `test/stats_test.py` (statistics) and
`test/benchmark_report_aggregate_test.py` (metric-group parsing and aggregate metrics).

## Pitfalls

- **Relative `root_dir`** is resolved against the *base output directory*: the
  config file's directory by default, but `--base-output-dir` when that is
  given. Use an absolute path when you change the output directory.
- **`clean: false`.** LaTeX drivers include every matching table and figure
  in `tables/` and `plots/`. With `clean: false`, that includes files left
  over from earlier runs with different settings. Hand-written files in those
  two folders are deleted when `clean` is on, so keep them elsewhere.
- **`quality_result_cmap` is ignored.** The key the code reads is
  `intrusion_cmap`.
- **Per-environment layout.** In a `<root>/<algo>/<env>/...` tree, the episode
  keys become `<env>/<episode>` (see [Input data](#input-data)), so
  `episode_groups` members must use that form.
- **`use_tex: true`** needs a working LaTeX installation (`latex`, `dvipng`).
  Without one, rendering fails; set `use_tex: false` instead.
- **Brackets mode** needs the optional `statannotations` package.
- **Few runs.** With about 5 runs per navigator, use `test: permutation` (or
  `kruskal`) instead of relying on the F or χ² approximations. Remember that
  `min_group_size` silently leaves out navigators below the threshold.
