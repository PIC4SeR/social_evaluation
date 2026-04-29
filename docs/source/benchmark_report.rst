Benchmark Reports
=================

``benchmark_report`` turns a metrics dataset into benchmark artifacts for a set
of scenarios, episodes, and navigation algorithms. It can export summary
tables, success-rate tables, distribution plots, proxemic intrusion plots,
ANOVA/Tukey analyses, pairwise heatmaps, rliable plots, probability of
improvement plots, and LaTeX report driver files.

.. code-block:: bash

   benchmark_report -h

Inputs
------

The command reads a YAML file with two top-level sections:

- ``benchmark``: dataset paths, scenario definitions, algorithm labels, metric
  grouping, display labels, and statistical/report settings.
- ``outputs``: the artifacts to generate.

The default example is ``config/report_config.yaml``.

.. code-block:: bash

   benchmark_report --config config/report_config.yaml

Use ``--base-output-dir`` to override the directory where scenario report
directories are written:

.. code-block:: bash

   benchmark_report \
     --config config/report_config.yaml \
     --base-output-dir results/paper_report

Benchmark Configuration
-----------------------

The required benchmark keys are:

- ``root_dir``: directory containing metrics/results for the configured
  scenarios.
- ``scenario``: one scenario entry or a list of scenario entries.
- ``navigators``: algorithm keys to include.
- ``navigator_names``: display names for algorithm keys.
- ``metric_groups``: metric groups, labels, optimization direction, and optional
  failed-run handling.
- ``intrusion_labels``: labels and failed-run handling for proxemic intrusion
  metrics.

.. code-block:: yaml

   benchmark:
     root_dir: /path/to/results/experiments
     scenario:
       - name: episode18_annotated
         episode_groups:
           Open Crowded:
             - episode_18
     navigators:
       - DWB_social_costmap
       - MPPI_social_costmap
       - ORCA
     navigator_names:
       DWB_social_costmap: "DWB-GSC"
       MPPI_social_costmap: "MPPI-GSC"
       ORCA: "ORCA"
     metric_groups:
       navigation:
         time_to_reach_goal:
           name: "TTG (s)"
           direction: DOWN
           discard_failed: true
         avg_robot_linear_speed:
           name: "ARV (m/s)"
           direction: UP
     intrusion_labels:
       personal_space_intrusions:
         name: Personal Space Intrusions
         discard_failed: false

Metric ``direction`` tells the report whether larger (``UP``) or smaller
(``DOWN``) values are better. This is used by ranking, rliable normalization,
and performance-profile plots.

Scenarios and Episodes
----------------------

A scenario can be a string or a mapping. Mapping entries can override global
settings for that scenario:

.. code-block:: yaml

   benchmark:
     scenario:
       - name: mixed
         episodes:
           - episode_1
           - episode_2
         navigators:
           - ORCA
           - SARL
         episode_groups:
           Overall:
             - episode_1
             - episode_2
         discard_failed_metrics: false

``episode_groups`` controls grouped report views. If an episode label is
missing from ``episode_names``, the command derives one from the episode key.

Failed Runs
-----------

By default, failed runs are discarded for metric summaries. You can keep failed
runs globally, per scenario, or per metric:

.. code-block:: yaml

   benchmark:
     discard_failed_metrics: true
     scenario:
       - name: mixed
         discard_failed_metrics: false
     metric_groups:
       social:
         social_work_per_second:
           name: "SW step"
           direction: DOWN
           discard_failed: false

Metric-level ``discard_failed`` values take precedence for that metric. The same
field is available under ``intrusion_labels``.

Outputs
-------

Each output can be a boolean or a mapping with options. The most common outputs
are:

.. code-block:: yaml

   outputs:
     export_success_rate_tables: true
     export_metric_tables:
       enabled: true
       show_std: false
       groups:
         - navigation
     plot_success_rate:
       enabled: true
       n_cols: 2
     plot_metric_distributions:
       - kind: box
         cmap_name: Paired
         n_rows: 1
         n_cols: 2
         show_episode_ylabel: false
         annotate_stats: true
         stat_annotation_mode: baseline_stars
         stat_baseline: "SFM-NMPC"
         highlight_baseline: true
     plot_proxemic_intrusions:
       enabled: true
       n_rows: 1
     generate_latex_reports:
       enabled: true
       prefix: paper_report
       overwrite: true

Additional supported outputs include:

- ``export_anova_analysis``
- ``plot_pairwise_pvalue_heatmap``
- ``plot_pairwise_mean_diff_heatmap``
- ``plot_pairwise_hedges_heatmap``
- ``plot_metric_performance_profiles``
- ``plot_rliable_interval_estimates``
- ``plot_probability_of_improvement``

Statistical annotations on metric distribution plots can use pairwise brackets
or baseline stars. Configure them with ``annotate_stats``, ``stat_test``,
``stat_correction``, ``stat_alpha``, ``stat_use_anova``,
``stat_annotation_mode``, ``stat_baseline``, and ``stat_comparisons``.

Output Layout
-------------

The command writes one directory per scenario under the selected base output
directory. Generated artifacts are grouped into scenario-local subdirectories
such as figures, tables, statistics, rliable outputs, and LaTeX report files.
