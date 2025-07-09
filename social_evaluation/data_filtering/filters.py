import numpy as np
import pandas as pd
from social_evaluation.common import align_df_robot_agents, align_df_agents_on_time
from typing import List, Dict, Union, Optional, Tuple
from enum import Enum, auto


def filtered_robot_agents(
    df_robot: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    alpha: float = 0.1,
    hampel_params: Optional[Dict[str, Union[int, float]]] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Create a filtered version of the original data structure by applying
    smoothing techniques to the 'agents' and 'transforms' series.
    This function supports both simple moving average (SMA) and exponential moving average (EMA)
    smoothing methods.

    Args:
        df_robot: DataFrame containing robot data with a 'time' column.
        df_agents_list: List of DataFrames, each containing agent data with a 'time' column.
        alpha: Smoothing factor for EMA (default: 0.1). For SMA, this is not used.
        hampel_params: Optional dictionary with parameters for the Hampel filter,
    Returns:
        DataFrame with processed robot data and a list of DataFrames with processed agent data.
    """

    return align_df_robot_agents(
        compute_ema(df_robot, alpha=alpha, hampel_params=hampel_params),
        filter_agents_list(df_agents_list, alpha=alpha, hampel_params=hampel_params),
    )


def filter_agents_list(
    df_agents_list: List[pd.DataFrame],
    alpha: float = 0.1,
    hampel_params: Optional[Dict[str, Union[int, float]]] = None,
) -> pd.DataFrame:
    """
    Filter agent data using specified smoothing method.

    Args:
        df_agents: DataFrame containing agent data with a 'time' column.
        alpha: Smoothing factor for EMA (default: 0.1). For SMA, this is not used.
        hampel_params: Optional dictionary with parameters for the Hampel filter,
    1. 'window': Size of the rolling window (default: 21).
    2. 'n_sigma': Multiplier for the scaled MAD (default: 1.5).

    Returns:
        DataFrame with processed agent data.
    """
    df_agents_list = align_df_agents_on_time(df_agents_list)
    return [
        compute_ema(df_agent, alpha=alpha, hampel_params=hampel_params)
        for df_agent in df_agents_list
    ]


def compute_ema(
    df: pd.DataFrame, alpha: float = 0.1, hampel_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Compute exponential moving average for numeric columns while preserving non-numeric columns.

    Args:
        df: Input DataFrame
        alpha: Smoothing factor for EMA (default: 0.1)

    Returns:
        DataFrame with processed numeric columns and original non-numeric columns
    """

    # Identify non-numeric columns and their metadata
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    col_metadata = {
        col: {"data": df[col], "index": df.columns.get_loc(col)}
        for col in non_numeric_cols
    }

    # Default Hampel parameters if not provided
    if hampel_params is None:
        hampel_params = {"window": 21, "n_sigma": 1.5}
    # Process numeric columns with EMA
    numeric_df = df.drop(columns=non_numeric_cols)
    numeric_df_filtered = hampel_filter_with_mirror_padding(
        numeric_df, window=hampel_params["window"], n_sigma=hampel_params["n_sigma"]
    )

    # Compute EMA using the provided span
    ema_numeric = numeric_df_filtered.ewm(alpha=alpha, adjust=False).mean()

    # Reinsert non-numeric columns at their original positions
    for col, meta in col_metadata.items():
        ema_numeric.insert(meta["index"], col, meta["data"])

    return ema_numeric


def hampel_filter_with_mirror_padding(
    series: pd.Series, window: int = 5, n_sigma: float = 3.0
) -> pd.Series:
    """
    Applies the Hampel filter using mirror padding at the boundaries.

    Args:
        series: Input pandas Series to filter.
        window: Size of the rolling window (should be odd).
        n_sigma: Multiplier for the scaled MAD (Median Absolute Deviation) to determine outliers.

    Returns:
        A pandas Series with outliers replaced.
    """
    # Calculate half the window size
    half_w = window // 2

    # Ensure the series is long enough for padding
    if len(series) < window:
        raise ValueError(
            "Series length must be at least equal to the window size for mirror padding."
        )

    # Mirror padding:
    # For the left pad, reflect the values immediately after the first element.
    left_pad = series.iloc[:half_w][::-1]
    # For the right pad, reflect the values immediately before the last element.
    right_pad = series.iloc[-half_w:][::-1]
    original_index = series["time"]
    first_time = original_index.iloc[0]
    second_time = original_index.iloc[1]
    # Determine time step: if the index is not uniformly spaced,
    # you might choose the median of differences.
    if isinstance(original_index, pd.DatetimeIndex):
        dt_left = original_index[1] - original_index[0]
        dt_right = original_index[-1] - original_index[-2]
        left_index = [
            original_index[0] - dt_left * (i + 1) for i in reversed(range(half_w))
        ]
        right_index = [original_index[-1] + dt_right * (i + 1) for i in range(half_w)]
    else:
        # Assume a numeric index with uniform spacing.
        dt = second_time - first_time
        left_index = [
            original_index.iloc[0] - dt * (i + 1) for i in reversed(range(half_w))
        ]
        right_index = [original_index.iloc[-1] + dt * (i + 1) for i in range(half_w)]
    left_pad["time"] = left_index
    right_pad["time"] = right_index
    # Concatenate the left pad, original series, and right pad.

    s_padded = pd.concat([left_pad, series, right_pad]).reset_index(drop=True)

    # Define the Hampel function to apply on each window.
    # Fill missing values in s_padded using forward fill, then backward fill as fallback
    s_padded = s_padded.ffill().bfill()
    # Apply the rolling Hampel filter on the padded series.

    filtered_padded = hampel_filter(s_padded, window=window, n_sigma=n_sigma)

    # Trim the padded parts, returning only the filtered values corresponding to the original series.
    filtered = filtered_padded.iloc[half_w : half_w + len(series)].reset_index(
        drop=True
    )
    return filtered


def hampel_filter(
    series: pd.Series, window: int = 5, n_sigma: float = 1.5
) -> pd.Series:
    """
    Applies the Hampel filter to a pandas Series.
    Args:
        series: Input pandas Series to filter.
        window: Size of the rolling window (should be odd).
        n_sigma: Multiplier for the scaled MAD (Median Absolute Deviation) to determine outliers,
            it is the number of standard deviations away from the median to consider a point as an outlier.
    Returns:
        A pandas Series with outliers replaced.
    """

    def hampel(x: np.ndarray) -> float:
        """
        Hampel filter function to detect and replace outliers in a numpy array.
        Args:
            x: Numpy array of values in the current rolling window.
        Returns:
            The median of the array if the center value is an outlier, otherwise the center value.
        """
        if np.isnan(x).any():
            return np.nan  # Handle NaN values gracefully
        median = np.median(x)  # Compute the median of the window
        threshold = (
            n_sigma * 1.4826 * np.median(np.abs(x - median))
        )  # MAD scaled to match STD
        center = x[len(x) // 2]
        if np.abs(center - median) > threshold:
            return median  # Replace outlier with median
        return center

    return series.rolling(window, center=True, min_periods=1).apply(hampel, raw=True)
