Bag Extraction
==============

Extract dataframes from rosbags and save them as pickle files
-------------------------------------------------------------

The first step is to extract the dataframes from the rosbags and save them as pickle files.

.. code-block:: bash

    bag_extraction -h # to see the help

The script `bag_extraction` extracts the dataframes from the rosbags and saves them as pickle files. The script takes as input the path to the rosbag file and the path to the output directory where the pickle files will be saved. The script also takes as input the name of the topics that will be extracted from the rosbag file.

The supported msg types are:

- `sensor_msgs/msg/LaserScan`
- `geometry_msgs/msg/PoseStamped`
- `nav_msgs/msg/Odometry`
- `nav_msgs/msg/Path`
- `people_msgs/msg/People`
- `tf2_msgs/msg/TFMessage`

The script will save the dataframes as pickle files in the output directory. The name of the pickle files can be specified by the user.

The script needs 2 configuration files:

- `config.yaml` is used to process different bag files in a structured way, but requires that a certain file system structure is used. The config file contains the names of the experiments, the algorithms, the experiment ids, the time limits for each experiment and algorithm, and the base path of the bag files. The time limits are used to extract the dataframes only for the time interval that we are interested in. The config file also contains the names of the topics that will be extracted from the rosbag file.

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

- `experiment_config.yaml` which contains information about the experiment that we want to extract the dataframes for. The file contains the transforms that will be collected from the rosbag file, the map file that will be used for the experiment, the robot dynamics, and the examined people identifiers. The file also contains the name of the experiment that we want to extract the dataframes for.

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

To extract the data using the `config.yaml` and `experiment_config.yaml` files, run the following command:

.. code-block:: bash

    bag_extraction --config-file config.yaml --exp-config-file experiment_config.yaml --output-file *<path_to_output_file.pickle>*

The script will extract the dataframes from the rosbag file and save them as pickle files in the output pickle file specified by the user.
