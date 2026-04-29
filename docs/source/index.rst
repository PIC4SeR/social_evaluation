.. social_evaluation documentation master file, created by
   sphinx-quickstart on Tue Mar 25 23:23:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Social Evaluation
===============================

Installation
============

.. code-block:: bash

   git clone ...
   cd social_evaluation
   pip install -e . # to install the package in editable mode

Usage of the package
====================

The package is divided into different parts:

- Extract ROS 2 bag topics into pandas dataframes stored in pickle files.
- Filter robot and agent trajectories with EMA smoothing and Hampel outlier removal.
- Plot single-run and aggregated trajectory images or animations.
- Compute social-navigation metrics from extracted data.
- Generate benchmark tables, plots, statistical analyses, and LaTeX report drivers.

Bag extraction and data conversion
----------------------------------

:doc:`/bag_extraction`

Filtering of the data
---------------------

:doc:`/data_filtering`

Plotting of the animations of the trajectories
----------------------------------------------

:doc:`/trajectory_plots`

Computation of the social evaluation metrics
--------------------------------------------

:doc:`/metrics_evaluation`

Benchmark report generation
---------------------------

:doc:`/benchmark_report`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   bag_extraction
   data_filtering
   trajectory_plots
   metrics_evaluation
   benchmark_report
   
.. toctree::
   :maxdepth: 2  
   :caption: Reference:
   
   social_evaluation/social_evaluation
