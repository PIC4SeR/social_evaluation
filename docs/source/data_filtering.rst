Data filtering
==============

This document describes the process of filtering the data for the social evaluation after the extraction of the data from the bags.
The data filtering is done in order to remove the data that is too noisy or not relevant for the social evaluation.

Filtering the data
------------------

The data filtering is done using the `data_filtering` module of the `social_evaluation_graphs` package. This module provides functions to filter the data using a exponential moving average (EMA) and to remove outliers with an hampel filter.

Usage
-----


To filter the data, you need to provide the path to the pickle file containing the dataframes, the path to the output file where the filtered data will be saved, and the parameters for the EMA and hampel filter.
You can use the following command to visualize the help message and see the available options:

.. code-block:: bash

   data_filtering --pickle-file --help
   
Parameters
----------

- `--pickle-file`: The path to the input pickle file containing the dataframes.
- `--output-file`: The path to the output pickle file where the filtered data will be saved.
- `--alpha`: The smoothing factor for the exponential moving average (EMA). A value between 0 and 1, where a higher value means less smoothing.
- `--hampel-window-size`: The size of the window for the hampel filter. It should be an odd integer.
- `--hampel-n-sigmas`: The threshold for the hampel filter. It is used to determine if a value is an outlier. A higher value means that fewer points will be considered as outliers. This is a multiplier for the standard deviation of the data in the window.

Examples of the data filtering
------------------------------


.. code-block:: bash

   data_filtering --pickle-file dataframes.pickle --output-file filtered_data.pickle --alpha 0.1 --hampel-window-size 5 --hampel-n-sigmas 3

This command will filter the data in `dataframes.pickle` using an EMA with a smoothing factor of 0.1 and a hampel filter with a window size of 5 and a threshold of 3. The filtered data will be saved in `filtered_data.pickle`.

Output
------

The output of the filtering process is a pickle file containing the filtered dataframes. The filtered data will have the same structure as the input data, but with the noisy and outlier data removed.
The filtered data can then be used for further analysis, such as plotting the trajectories or computing the social evaluation metrics.


.. toctree::
   :maxdepth: 2