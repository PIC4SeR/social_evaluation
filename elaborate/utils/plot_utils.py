import pandas as pd
from pathlib import Path
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, List, Tuple
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

from social_evaluation.utils.metrics_utils import combine_df_robot_agents
from scipy.ndimage import rotate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, ArtistAnimation
from tqdm import trange


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image data to ensure valid RGB ranges.
    Floats are clipped to [0, 1], integers to [0, 255].
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Clip float data to [0, 1]
        image = np.clip(image, 0, 1)
    elif image.dtype == np.int32 or image.dtype == np.int64 or np.any(image > 255):
        # Clip integer data to [0, 255] and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def meters_to_pixels(meters: float, ax: plt.Axes) -> float:
    """
    Convert meters to pixels using the current axis limits.
    """
    x0, x1 = ax.get_xbound()
    width_px = ax.get_window_extent().width
    width_m = x1 - x0
    return meters * width_px / width_m


def create_line_collection(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    norm: mcolors.Normalize,
    cmap: str = "viridis",
) -> mcoll.LineCollection:
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(velocity)
    return lc


def create_image_marker(
    image: np.ndarray, desired_width_px: float = 20, angle: float = 0, alpha: float = 1
) -> OffsetImage:
    rotated_image = normalize_image(rotate(image, np.degrees(angle), reshape=True))
    return OffsetImage(
        rotated_image,
        zoom=desired_width_px / image.shape[0],
        alpha=alpha,
    )


def create_annotation_box(
    image: np.ndarray,
    x: float,
    y: float,
    desired_width_px: float = 20,
    angle: float = 0,
    alpha: float = 1,
) -> AnnotationBbox:
    image_rot = create_image_marker(
        image, desired_width_px=desired_width_px, angle=angle, alpha=alpha
    )
    return AnnotationBbox(image_rot, (x, y), frameon=False)
