Trajectory Plots
================

``plot_trajectories`` generates static trajectory images and optional
animations from extracted pickle files. It can plot each run independently and
can also plot aggregated trajectories for each experiment/algorithm pair.

Usage of the plot_trajectories code
-----------------------------------

After converting ROS 2 bags to pickle files, inspect the available options with:

.. code-block:: bash

    plot_trajectories -h # to see the help message

To run the code, you need to provide:

- A trajectory plotting config file.
- Either one extracted pickle file or a base directory containing per-experiment pickle files.
- The output directory where images and animations will be saved.

.. code-block:: bash

    plot_trajectories \
      --pickle-file <path_to_pickle_file> \
      --config-file <path_to_config_file> \
      --output-dir <path_to_output_dir>

For per-experiment extraction outputs, load all pickles under a directory:

.. code-block:: bash

    plot_trajectories \
      --pickle-base-path <path_to_pickle_directory> \
      --config-file config/trajectory_plots.yaml \
      --output-dir plotted_trajectories

The config file should contain the following information:

.. code-block:: yaml

    # trajectory_plots.yaml
    plot_single: true
    plot_aggregated: false
    plot_colorbar: false
    time_freq: 0.1
    use_odometry: false
    save_fig: true
    save_gif: true
    save_webm: false
    save_animation_formats:
      - gif
    animation_writer: pillow
    fontsize: 16
    image_width_multi: 1.0
    bbox_pad_inches: 0.05

    experiments:
      experiment_1:
        name: "Experiment 1"
        path: "<path_to_experiment_1_pickle_file>"
        ax_limits: [-5, 5, -5, 5]

If ``save_animation_formats`` is set, supported values are ``gif`` and
``webm``. GIF output uses Pillow. WebM output requires an ffmpeg writer. Set
``use_odometry`` to ``true`` to plot robot odometry instead of the
``map_to_base_link`` transform.

Output
------

The command writes outputs under:

.. code-block:: bash

   <output_dir>/<experiment_name>/<algorithm_name>/

For single-run plots, files are named ``experiment_<id>.png``, ``.gif``, or
``.webm``. For aggregate plots, files are named ``aggregated_trajectories``.

Example
-------

The following is an example of the output of the code.

The following is the trajectory of the robot in the social navigation task.

.. image:: images/experiment.png
   :alt: trajectory

The following is the animation of the trajectory of the robot in the social navigation task.

.. image:: images/experiment.gif
   :alt: animation

.. toctree::
    :maxdepth: 2
