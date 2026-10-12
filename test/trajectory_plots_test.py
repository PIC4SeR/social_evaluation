import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import AbstractMovieWriter

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


class _CountingWriter(AbstractMovieWriter):
    """Records how many trails the axes hold when each frame is grabbed."""

    def __init__(self, ax):
        super().__init__(fps=10)
        self.ax, self.trails = ax, []

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi)

    def grab_frame(self, **savefig_kwargs):
        self.trails.append(len(self.ax.collections))

    def finish(self):
        pass


def _fleet(samples):
    t = np.arange(samples) * 0.1
    line = lambda y: pd.DataFrame(
        {"time": t, "position_x": t, "position_y": np.full(samples, y), "yaw": 0.0}
    )
    combined = pd.DataFrame(
        {"time": t, "position_x_agent0": -t, "position_y_agent0": 1.0, "yaw_agent0": 0.0}
    )
    return combined, {"r1": line(0.0), "r2": line(2.0)}


def test_animation_holds_one_frame_and_leaves_the_last_for_the_png():
    """ArtistAnimation kept every frame's sprites alive (60-130 MB a frame for
    four TB2s) and a 592-frame run was OOM-killed; frames are now built on draw."""
    samples = 30
    combined, robots = _fleet(samples)
    fig, ax = plt.subplots()
    ani = plots.plot_trajectories(combined, ax, fig=fig, animate=True, df_robots=robots)
    assert len(ax.collections) == 0  # nothing built up front
    writer = _CountingWriter(ax)
    ani.save("unused.gif", writer=writer)
    assert writer.trails == [3] * samples  # two robots and a pedestrian, one frame each
    plots.get_bbox_for_figure(fig, ax)  # the plain draw main() does before the PNG
    assert [len(trail.get_segments()) for trail in ax.collections] == [samples - 1] * 3
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["r1", "r2"]
    first, second = (tuple(h.get_color()) for h in legend.legend_handles)
    assert first != second  # each robot's own colour, not the default blue
    plt.close(fig)


def test_exact_scale_markers_are_drawn_at_their_size_in_metres():
    """marker_sizes: a marker asked for 1 m is 1 m on the axes; the legacy path
    keeps OffsetImage's dpi/72 enlargement, so existing plots are unchanged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from social_evaluation.trajectory_plots import plot_utils as pu

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    fig.canvas.draw()
    image = np.ones((50, 50, 4))
    one_metre_px = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    widths = {}
    for exact in (True, False):
        box = pu.create_annotation_box(image, 5, 5, desired_width_px=pu.meters_to_pixels(1.0, ax),
                                       exact_scale=exact)
        ax.add_artist(box)
        fig.canvas.draw()
        # the image itself, not the AnnotationBbox (which adds its own padding)
        widths[exact] = box.offsetbox.get_window_extent(fig.canvas.get_renderer()).width
        box.remove()
    plt.close(fig)
    assert abs(widths[True] - one_metre_px) / one_metre_px < 0.03
    assert abs(widths[False] / widths[True] - 100 / 72) < 0.03


def test_marker_size_resolves_robot_models_and_agents():
    from social_evaluation.trajectory_plots.animate_agents import marker_size
    sizes = {"robots": {"turtlebot2": 0.356, "jackal": 0.49, "default": 0.4}, "agent": 0.5}
    assert marker_size(sizes, "turtlebot2_1") == 0.356
    assert marker_size(sizes, "jackal") == 0.49
    assert marker_size(sizes, "tb3_0") == 0.4
    assert marker_size(sizes, None) == 0.5
    assert marker_size(None, "jackal") == 0.15 and marker_size(None, None) == 0.13
