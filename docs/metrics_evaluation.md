# Metrics Evaluation

## Evaluate the social metrics from the pickle files

Use the script `metrics_evaluation` to evaluate social metrics from extracted
pickle files. The command reads experiment and algorithm names from
`--config-file`, loads extracted run data from `--data-file`, reads metric
parameters and the selected metric list from `--metrics-file`, and writes one
metrics dataframe to `--output-file`.

```bash
metrics_evaluation -h # to see the help
```

The supported social metrics are:

- __*time_to_reach_goal*__ : the time that the robot needs to reach the goal
- __*path_length*__ : the length of the path that the robot follows to reach the goal
- __*cumulative_heading_changes*__ : the number of heading changes that the robot makes
- __*avg_distance_to_closest_person*__ : the average distance of the robot to the closest person
- __*minimum_distance_to_people*__ : the minimum distance of the robot to the people
- __*maximum_distance_to_people*__ : the maximum distance of the robot to the people
- __*intimate_space_intrusions*__ : the percentage of the time that the robot intrudes the intimate space of the people
- __*personal_space_intrusions*__ : the percentage of the time that the robot intrudes the personal space of the people
- __*social_space_intrusions*__ : the percentage of the time that the robot intrudes the social space of the people
- __*public_space_intrusions*__ : the percentage of the time that the robot intrudes the public space of the people
- __*group_intimate_space_intrusions*__ : the percentage of the time that the robot intrudes the intimate space of the group of people
- __*group_personal_space_intrusions*__ : the percentage of the time that the robot intrudes the personal space of the group of people
- __*group_social_space_intrusions*__ : the percentage of the time that the robot intrudes the social space of the group of people
- __*group_public_space_intrusions*__ : the percentage of the time that the robot intrudes the public space of the group of people
- __*robot_on_person_collision*__ : the number of collisions between the robot and the people caused by the robot
- __*person_on_robot_collision*__ : the number of collisions between the robot and the people caused by the people
- __*time_not_moving*__ : the cumulative time that the robot is not moving
- __*completed*__ : if the robot reached the goal in the experiment
- __*final_distance_to_target*__ : the distance of the robot to the goal at the end of the experiment
- __*minimum_distance_to_target*__ : the minimum distance of the robot to the goal
- __*avg_robot_linear_speed*__ : the average linear speed of the robot
- __*avg_robot_angular_speed*__ : the average angular speed of the robot
- __*avg_acceleration*__ : the average acceleration of the robot
- __*avg_overacceleration*__ : the average overacceleration of the robot
- __*avg_pedestrian_velocity*__ : the average velocity of the people
- __*avg_closest_pedestrian_velocity*__ : the average velocity of the closest person to the robot
- __*social_force_on_agents*__ : the social force that the robot applies to the people in modulus
- __*social_force_on_robot*__ : the social force that the people apply to the robot in modulus
- __*social_work*__ : the social work that the robot applies to the environment and that is caused by the robot
- __*social_work_per_second*__ : the instantaneous social work per second that the robot applies to the environment and that is caused by the robot
- __*obstacle_force_on_agents*__ : the obstacle force that the robot applies to the people in modulus
- __*obstacle_force_on_robot*__ : the obstacle force that the people apply to the robot in modulus

The script saves the results as a single pandas dataframe in the output pickle
file specified by the user. Each row represents one evaluated run and includes
the selected metric values plus `experiment`, `algorithm`, and `run` columns.

The script needs two configuration files:

- `config.yaml` provides the `experiment_names` and `algorithms` used to group
  runs from the extracted data.

```yaml
# config.yaml
experiment_names:
  - group
  - queue
algorithms:
  - first_planner
  - second_planner
```

- `metrics.yaml` contains robot/agent parameters and the social metrics to
  evaluate.

```yaml
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
```

## Example usage

To evaluate metrics from an already extracted pickle file, run:

```bash
metrics_evaluation \
  --config-file config/config.yaml \
  --metrics-file config/metrics.yaml \
  --data-file <path_to_data_pickle_file>.pickle \
  --output-file <path_to_output_pickle_file>.pickle
```

The command loads already extracted data from `--data-file`, evaluates the
metrics listed in `--metrics-file`, and writes the metrics dataframe to
`--output-file`.
