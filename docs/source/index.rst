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

Usage
=====

The package is divided into different parts:

- The extraction of the data from the rosbags and the conversion of the data into pandas dataframes stored in pickle files
- The filtering of the data to remove the outliers
- The plotting of animations of the trajectories of the agents
- The computation of the social evaluation metrics
- The plotting of the social evaluation metrics

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

Plotting and elaboration of the social evaluation metrics
---------------------------------------------------------

:doc:`/metrics_visualization`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   bag_extraction
   data_filtering
   trajectory_plots
   metrics_evaluation
   metrics_visualization
   
.. toctree::
   :maxdepth: 2  
   :caption: Reference:
   
   modules

