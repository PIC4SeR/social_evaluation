import pandas as pd
import numpy as np


def create_filtered_data(original_data, window=5, method="ema"):
    """
    Create a filtered version of data_1 with processed series.

    Args:
        original_data: The original data_1 structure
        window: Smoothing window size
        method: 'sma' (simple moving average) or 'ema' (exponential)

    Returns:
        Dictionary mirroring original_data structure with processed series
    """
    filtered_data_1 = {}

    if "agents" in original_data and "transforms" in original_data:

        # Process transforms: apply EMA smoothing
        filtered_data_1["transforms"] = {
            key: compute_ema(df, alpha=2 / (window + 1))
            for key, df in original_data["transforms"].items()
        }

        # Process agents: choose EMA or rolling average based on method
        filtered_data_1["agents"] = [
            (
                compute_ema(agent, alpha=2 / (window + 1))
                if method == "ema"
                else compute_rolling_average(agent, window=window)
            )
            for agent in original_data["agents"]
        ]
        filtered_data_1["agents"], filtered_data_1["transforms"] = (
            # TODO: align_agents_on_time USE combine_agents_on_time
            align_agents_transforms_on_time(
                filtered_data_1["agents"], filtered_data_1["transforms"]
            )
        )
    else:
        # Process transforms series
        if "transforms" in original_data:
            filtered_data_1["transforms"] = {
                transform_key: compute_ema(transform_df, alpha=2 / (window + 1))
                for transform_key, transform_df in original_data["transforms"].items()
            }

        # Process agents series
        if "agents" in original_data:
            # TODO: align_agents_on_time USE combine_agents_on_time
            agents_used = align_agents_on_time(original_data["agents"])
            filtered_data_1["agents"] = [
                (
                    compute_ema(agent, alpha=2 / (window + 1))
                    if method == "ema"
                    else compute_rolling_average(agent, window=window)
                )
                for agent in agents_used
            ]

    # Copy non-processed keys
    processed_keys = {"transforms", "agents"}
    filtered_data_1.update(
        {k: v for k, v in original_data.items() if k not in processed_keys}
    )

    return filtered_data_1


def compute_ema(df, alpha):
    """
    Compute exponential moving average for numeric columns while preserving non-numeric columns.

    Args:
        df: Input DataFrame
        span: Effective window size for EMA (default: 5)

    Returns:
        DataFrame with processed numeric columns and original non-numeric columns
    """
    # Identify non-numeric columns and their metadata
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    col_metadata = {
        col: {"data": df[col], "index": df.columns.get_loc(col)}
        for col in non_numeric_cols
    }

    # Process numeric columns with EMA
    numeric_df = df.drop(columns=non_numeric_cols)
    numeric_df = numeric_df.ffill().bfill()
    numeric_df_filtered = hampel_filter_with_mirror_padding(
        numeric_df, window=20, threshold=1.5
    )

    # Compute EMA using the provided span
    ema_numeric = numeric_df_filtered.ewm(alpha=alpha, adjust=False).mean()

    # Reinsert non-numeric columns at their original positions
    for col, meta in col_metadata.items():
        ema_numeric.insert(meta["index"], col, meta["data"])

    return ema_numeric


def hampel_filter_with_mirror_padding(s, window=5, threshold=3.0):
    """
    Applies the Hampel filter using mirror padding at the boundaries.

    Args:
        s: A pandas Series containing numeric data.
        window: Size of the rolling window (should be odd).
        threshold: Multiplier for the scaled MAD to determine outlier threshold.

    Returns:
        A pandas Series with outliers replaced.
    """
    # Calculate half the window size
    half_w = window // 2

    # Ensure the series is long enough for padding
    if len(s) < window:
        raise ValueError(
            "Series length must be at least equal to the window size for mirror padding."
        )

    # Mirror padding:
    # For the left pad, reflect the values immediately after the first element.
    left_pad = s.iloc[:half_w][::-1]
    # For the right pad, reflect the values immediately before the last element.
    right_pad = s.iloc[-half_w:][::-1]
    original_index = s["time"]
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

    s_padded = pd.concat([left_pad, s, right_pad]).reset_index(drop=True)

    # Define the Hampel function to apply on each window.

    # Apply the rolling Hampel filter on the padded series.

    filtered_padded = hampel_filter(s_padded, window=window, threshold=threshold)

    # Trim the padded parts, returning only the filtered values corresponding to the original series.
    filtered = filtered_padded.iloc[half_w : half_w + len(s)].reset_index(drop=True)
    return filtered


def hampel_filter(s, window=5, threshold=3.0):
    def hampel(x):
        if np.isnan(x).any():
            return np.nan
        median = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - median))  # MAD scaled to match STD
        center = x[len(x) // 2]
        if np.abs(center - median) > threshold * mad:
            return median  # Replace outlier with median
        return center

    return s.rolling(window, center=True, min_periods=1).apply(hampel, raw=True)


def compute_rolling_average(df, window=5):
    """
    Compute rolling average for numeric columns while preserving non-numeric columns.

    Args:
        df: Input DataFrame
        window: Rolling window size (default: 5)

    Returns:
        DataFrame with processed numeric columns and original non-numeric columns
    """
    # Identify non-numeric columns and their metadata
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    col_metadata = {
        col: {"data": df[col], "index": df.columns.get_loc(col)}
        for col in non_numeric_cols
    }

    # Process numeric columns
    numeric_df = df.drop(columns=non_numeric_cols)
    avg_numeric = (
        numeric_df.rolling(window=window, min_periods=1).mean().ffill().bfill()
    )

    # Reinsert non-numeric columns at their original positions
    for col, meta in col_metadata.items():
        avg_numeric.insert(meta["index"], col, meta["data"])

    return avg_numeric
