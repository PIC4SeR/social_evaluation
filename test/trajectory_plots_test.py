import matplotlib.pyplot as plt
import pandas as pd

import social_evaluation.trajectory_plots as plots


def test_robot_images_use_indexed_fallbacks():
    images, _, _ = plots.get_images_for_experiment(0, ["jackal", "not_added_yet"])
    assert len(images) == 2
    assert not (images[0] == images[1]).all()


def test_plot_trajectories_draws_every_fleet_member():
    combined = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "position_x_agent0": [0.0, 1.0],
            "position_y_agent0": [1.0, 1.0],
            "yaw_agent0": [0.0, 0.0],
        }
    )
    robots = {
        "r1": pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "position_x": [0.0, 1.0],
                "position_y": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            }
        ),
        "r2": pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "position_x": [1.0, 2.0],
                "position_y": [2.0, 2.0],
                "yaw": [0.0, 0.0],
            }
        ),
    }
    fig, ax = plt.subplots()
    plots.plot_trajectories(combined, ax, fig=fig, animate=False, df_robots=robots)
    labels = {collection.get_label() for collection in ax.collections}
    assert {"r1", "r2"} <= labels
    r1 = next(collection for collection in ax.collections if collection.get_label() == "r1")
    assert max(r1.cmap(0.0)[:3]) - min(r1.cmap(0.0)[:3]) > 0.3
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["r1", "r2"]
    assert len(ax.artists) > 2
    plt.close(fig)
