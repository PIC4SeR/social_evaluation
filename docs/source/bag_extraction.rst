Bag Extraction
==============

Extract dataframes from ROS 2 bags and save them as pickle files
----------------------------------------------------------------

The first step is to extract dataframes from ROS 2 bags and save them as pickle
files.

.. code-block:: bash

    bag_extraction -h # to see the help

The script ``bag_extraction`` extracts the dataframes from ROS 2 bags and saves
them as pickle files. It reads the bag root, experiment names, algorithm names,
run ids, time limits, and topic names from the extraction config.

The supported msg types are:

- ``geometry_msgs/msg/PoseStamped``
- ``nav_msgs/msg/Odometry``
- ``nav_msgs/msg/Path``
- ``people_msgs/msg/People``
- ``std_msgs/msg/String``
- ``tf2_msgs/msg/TFMessage``

The script will save the dataframes as pickle files in the output directory. The name of the pickle files can be specified by the user.

The script needs 2 configuration files:

- ``config.yaml`` is used to process different bag files in a structured way, but requires that a certain file system structure is used. The config file contains the names of the experiments, the algorithms, the experiment ids, the time limits for each experiment and algorithm, and the base path of the bag files. The time limits are used to extract the dataframes only for the time interval that we are interested in. The config file also contains the names of the topics that will be extracted from the rosbag file.

.. code-block:: yaml

    # config.yaml
    experiment_names: 
      - "experiment_1" # name of the first experiment
      - "experiment_2" # name of the second experiment 
    algorithms:
      - "first_algorithm" # name of the first algorithm
      - "second_algorithm" # name of the second algorithm
    experiment_ids:
      - 1 # id of the first experiment
      - 2 # id of the second experiment
    time_limits:
      experiment_1:
        first_algorithm:
          1: 
            start_time: 18.765
            end_time: 29.664
          2:
            start_time: 19.035
            end_time: 31.659

    base_path: <path_to_bag_files_here>

The bag files should be placed in a structured way in the file system:

.. code-block:: bash

    <bags_root_directory>
    ├── experiment_1
    │   ├── first_algorithm
    │   │   ├── experiment_1
    │   │   │   ├── experiment_1_0.db3
    │   │   │   └── metadata.yaml
    │   │   └── experiment_2
    │   │       ├── experiment_2_0.db3
    │   │       └── metadata.yaml
    │   └── second_algorithm
    │       ├── experiment_1
    │       │   ├── experiment_1_0.db3
    │       │   └── metadata.yaml
    │       └── experiment_2
    │           ├── experiment_2_0.db3
    │           └── metadata.yaml
    └── experiment_2
        ├── first_algorithm
        │   ├── experiment_1
        │   │   ├── experiment_1_0.db3
        │   │   └── metadata.yaml
        │   └── experiment_2
        │       ├── experiment_2_0.db3
        │       └── metadata.yaml
        └── second_algorithm
            ├── experiment_1
            │   ├── experiment_1_0.db3
            │   └── metadata.yaml
            └── experiment_2
                ├── experiment_2_0.db3
                └── metadata.yaml

- ``experiments_config.yaml`` contains per-experiment metadata. Each experiment entry defines the transforms to collect from the ROS 2 bag, the map file to use, robot dynamics parameters, and the people identifiers to evaluate.

.. code-block:: yaml

    # experiments_config.yaml
    experiment_name:
      transforms:
        - map_to_base_link:
            child_frame_id: "base_link"
            frame_id: "map"

      map_file: maps/<experiment_map.yaml>

      robot_dynamics:
        period_for_diff: 5
        max_velocity: 0.5
        max_angular_velocity: 0.5
        alpha: 0.1

      examined_people:
        - U_1.U_1

Example usage
-------------

To extract the data using the ``config.yaml`` and ``experiments_config.yaml`` files, run the following command:

.. code-block:: bash

    bag_extraction \
      --config-file config.yaml \
      --exp-config-file experiments_config.yaml \
      --output-file <path_to_output_file.pickle>

The script will extract the dataframes from the rosbag file and save them as pickle files in the output pickle file specified by the user.

For large benchmark datasets, write one pickle per experiment instead of one
combined output file:

.. code-block:: bash

    bag_extraction \
      --config-file config.yaml \
      --exp-config-file experiments_config.yaml \
      --base-output-path extracted_bags

When ``--base-output-path`` is used, the command stores per-experiment pickles
under that directory and skips scenarios that are already present in an
existing per-experiment pickle. Without ``--base-output-path``, an existing
``--output-file`` is loaded and reused so interrupted extractions can continue.
