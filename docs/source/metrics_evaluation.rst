Metrics Evaluation
==================

Evaluate the social metrics from the pickle files
-------------------------------------------------

Use the script ``metrics_evaluation`` to evaluate social metrics from extracted
pickle files. The command reads experiment and algorithm names from
``--config-file``, loads extracted run data from ``--data-file``, reads metric
parameters and the selected metric list from ``--metrics-file``, and writes one
metrics dataframe to ``--output-file``.

.. code-block:: bash

  metrics_evaluation -h # to see the help

The supported social metrics are:

- **time_to_reach_goal** : the time that the robot needs to reach the goal
- **path_length** : the length of the path that the robot follows to reach the goal
- **cumulative_heading_changes** : the number of heading changes that the robot makes
- **avg_distance_to_closest_person** : the average distance of the robot to the closest person
- **minimum_distance_to_people** : the minimum distance of the robot to the people
- **maximum_distance_to_people** : the maximum distance of the robot to the people
- **intimate_space_intrusions** : the percentage of the time that the robot intrudes the intimate space of the people
- **personal_space_intrusions** : the percentage of the time that the robot intrudes the personal space of the people
- **social_space_intrusions** : the percentage of the time that the robot intrudes the social space of the people
- **public_space_intrusions** : the percentage of the time that the robot intrudes the public space of the people
- **group_intimate_space_intrusions** : the percentage of the time that the robot intrudes the intimate space of the group of people
- **group_personal_space_intrusions** : the percentage of the time that the robot intrudes the personal space of the group of people
- **group_social_space_intrusions** : the percentage of the time that the robot intrudes the social space of the group of people
- **group_public_space_intrusions** : the percentage of the time that the robot intrudes the public space of the group of people
- **robot_on_person_collision** : the number of collisions between the robot and the people caused by the robot
- **person_on_robot_collision** : the number of collisions between the robot and the people caused by the people
- **time_not_moving** : the cumulative time that the robot is not moving
- **completed** : if the robot reached the goal in the experiment
- **final_goal_distance** : the distance of the robot to the goal at the end of the experiment
- **minimum_goal_distance** : the minimum distance of the robot to the goal
- **avg_robot_linear_speed** : the average linear speed of the robot
- **avg_robot_angular_speed** : the average angular speed of the robot
- **avg_acceleration** : the average acceleration of the robot
- **avg_overacceleration** : the average overacceleration of the robot
- **avg_pedestrian_velocity** : the average velocity of the people
- **avg_closest_pedestrian_velocity** : the average velocity of the closest person to the robot
- **social_force_on_agents** : the social force that the robot applies to the people in modulus
- **social_force_on_robot** : the social force that the people apply to the robot in modulus
- **social_work** : the social work that the robot applies to the environment and that is caused by the robot
- **social_work_per_second** : the instantaneous social work per second that the robot applies to the environment and that is caused by the robot
- **obstacle_force_on_agents** : the obstacle force that the robot applies to the people in modulus
- **obstacle_force_on_robot** : the obstacle force that the people apply to the robot in modulus

The script saves the results as a single pandas dataframe in the output pickle
file specified by the user. Each row represents one evaluated run and includes
the selected metric values plus ``experiment``, ``algorithm``, and ``run``
columns.

The script needs two configuration files:

- ``config.yaml`` provides the ``experiment_names`` and ``algorithms`` used to
  group runs from the extracted data.

.. code-block:: yaml

  # config.yaml
  experiment_names:
    - group
    - queue
  algorithms:
    - first_planner
    - second_planner

- ``metrics.yaml`` contains robot/agent parameters and the social metrics to
  evaluate.

.. code-block:: yaml

  # metrics.yaml
  metrics_config:
    robot_radius: 0.27
    agent_radius: 0.17
    period: 5
    min_velocity: 0.08
    min_angular_velocity: 0.01
    max_accel: 1.0
    max_jerk: 0.5
    goal_tolerance: 0.3
    alpha: 0.1
    distance_threshold: 0.2
  metrics:
    - time_to_reach_goal
    - path_length
    - cumulative_heading_changes
    - avg_distance_to_closest_person
    - minimum_distance_to_people
    - maximum_distance_to_people
    - intimate_space_intrusions
    - personal_space_intrusions
    - social_space_intrusions
    ...

Example usage of the script
---------------------------

To evaluate metrics from an already extracted pickle file, run:

.. code-block:: bash

  metrics_evaluation \
    --config-file config/config.yaml \
    --metrics-file config/metrics.yaml \
    --data-file <path_to_data_pickle_file>.pickle \
    --output-file <path_to_output_pickle_file>.pickle

The command loads already extracted data from ``--data-file``, evaluates the
metrics listed in ``--metrics-file``, and writes the metrics dataframe to
``--output-file``.


.. toctree::
   :maxdepth: 2
