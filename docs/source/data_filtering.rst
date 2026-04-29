Data Filtering
==============

This document describes the process of filtering the data for the social evaluation after the extraction of the data from the bags.
The data filtering is done in order to remove the data that is too noisy or not relevant for the social evaluation.

Filtering the data
------------------

The ``data_filtering`` command applies exponential moving average (EMA)
smoothing and Hampel outlier removal to the robot and agent trajectories in
extracted pickle files. If a ``map_to_base_link`` transform is available, the
robot trajectory is filtered from that transform; otherwise the odometry
trajectory is filtered.

Usage
-----


Inspect the available options with:

.. code-block:: bash

   data_filtering -h

Filter one pickle file:

.. code-block:: bash

   data_filtering \
     --pickle-file <path_to_pickle_file.pickle> \
     --output-file <path_to_output_file.pickle> \
     --alpha <alpha_value> \
     --hampel-window-size <hampel_window_size> \
     --hampel-n-sigmas <hampel_threshold>

Filter every ``.pickle`` file in a directory:

.. code-block:: bash

   data_filtering \
     --base-path <path_to_pickle_directory> \
     --base-output-dir <path_to_filtered_output_directory>
   
Parameters
----------

- ``--pickle-file``: The path to the input pickle file containing the dataframes.
- ``--base-path``: A directory of ``.pickle`` files to filter.
- ``--alpha``: The smoothing factor for the exponential moving average (EMA). A value between 0 and 1, where a higher value means less smoothing.
- ``--hampel-window-size``: The size of the window for the Hampel filter. It should be an odd integer.
- ``--hampel-n-sigmas``: The threshold for the Hampel filter. It is used to determine if a value is an outlier. A higher value means that fewer points will be considered as outliers. This is a multiplier for the standard deviation of the data in the window.
- ``--output-file``: The output pickle file for single-file mode.
- ``--base-output-dir``: Output directory for single-file or directory mode.

Use either ``--pickle-file`` or ``--base-path``, not both. Use either
``--output-file`` or ``--base-output-dir``, not both. If no output is provided,
the command writes next to the input file with ``_filtered`` appended to the
filename.

Examples of the data filtering
------------------------------


.. code-block:: bash

   data_filtering \
     --pickle-file dataframes.pickle \
     --output-file filtered_data.pickle \
     --alpha 0.1 \
     --hampel-window-size 5 \
     --hampel-n-sigmas 3

This command will filter the data in ``dataframes.pickle`` using an EMA with a smoothing factor of 0.1 and a Hampel filter with a window size of 5 and a threshold of 3. The filtered data will be saved in ``filtered_data.pickle``.

Output
------

The output of the filtering process is a pickle file containing the filtered dataframes. The filtered data will have the same structure as the input data, but with the noisy and outlier data removed.
The filtered data can then be used for further analysis, such as plotting the trajectories or computing the social evaluation metrics.


.. toctree::
   :maxdepth: 2
