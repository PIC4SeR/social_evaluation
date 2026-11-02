#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Convert the package's pickle files (extracted/filtered bags, metrics) to CSV.

A DataFrame becomes ``<name>.csv``. A dict becomes a folder: DataFrames and
2-D arrays inside it become CSV files, lists of DataFrames (the agents) are
stacked into one CSV with an ``item`` column, and scalars/tuples/small dicts
(goal, map_metadata) go to ``values.yaml``. Bag-path keys (the top level of
extracted data) become folders relative to the bags' common root.
"""
import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict

import argcomplete
import numpy as np
import pandas as pd
import yaml


def _plain(value: Any) -> Any:
    """Make numpy/tuple values YAML-safe."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _is_table_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, pd.DataFrame) for v in value)


def _has_tables(value: Any) -> bool:
    if isinstance(value, (pd.DataFrame, pd.Series)) or _is_table_list(value):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 2
    if isinstance(value, dict):
        return any(_has_tables(v) for v in value.values())
    return False


def _folder_names(keys: list) -> Dict[Any, str]:
    """Bag-path keys -> the bag folder relative to the common root, the recorder's
    ``scenario_<k>`` run folders renamed ``run_<k>``; other keys unchanged."""
    if len(keys) > 1 and all(isinstance(k, str) and k.endswith(".db3") for k in keys):
        parents = [os.path.dirname(k) for k in keys]
        root = os.path.commonpath(parents)
        return {
            k: re.sub(r"(^|/)scenario_(\d+)$", r"\1run_\2", os.path.relpath(p, root))
            for k, p in zip(keys, parents)
        }
    return {k: str(k).strip("/").replace("/", "_") for k in keys}


def write_csv(obj: Any, out: Path) -> int:
    """Write ``obj`` under ``out`` (``out.csv`` or folder ``out/``). Returns the CSV count."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if obj.empty:
            return 0
        out.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(out.with_suffix(".csv"), index=not isinstance(obj.index, pd.RangeIndex))
        return 1
    if _is_table_list(obj):
        stacked = pd.concat([df.assign(item=i) for i, df in enumerate(obj)], ignore_index=True)
        return write_csv(stacked, out)
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out.with_suffix(".csv"), obj, delimiter=",", fmt="%g")
        return 1
    if isinstance(obj, dict):
        names = _folder_names(list(obj))
        values = {}
        count = 0
        for key, value in obj.items():
            if _has_tables(value):
                count += write_csv(value, out / names[key])
            elif value is not None and not (isinstance(value, dict) and not value):
                values[str(key)] = _plain(value)
        if values:
            out.mkdir(parents=True, exist_ok=True)
            (out / "values.yaml").write_text(yaml.safe_dump(values, sort_keys=False))
        return count
    raise TypeError(f"{out}: cannot convert {type(obj).__name__} to CSV")


def main():
    parser = argparse.ArgumentParser(description="Convert social_evaluation pickle files to CSV")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pickle-file",
        type=Path,
        help="Path to a single pickle file to convert",
    )
    input_group.add_argument(
        "--base-path",
        type=Path,
        help="Directory whose pickle files (recursively) are converted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the CSVs (default: next to each pickle)",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.pickle_file is not None:
        if not args.pickle_file.is_file():
            parser.error(f"Pickle file {args.pickle_file} does not exist.")
        files, root = [args.pickle_file], args.pickle_file.parent
    else:
        if not args.base_path.is_dir():
            parser.error(f"Base path {args.base_path} does not exist.")
        files, root = sorted(args.base_path.rglob("*.pickle")), args.base_path

    for pickle_file in files:
        out_dir = args.output_dir / pickle_file.parent.relative_to(root) if args.output_dir else pickle_file.parent
        # pickle runs code from the file: only convert files you produced or trust.
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        n = write_csv(data, out_dir / pickle_file.stem)
        print(f"{pickle_file} -> {out_dir / pickle_file.stem}[.csv] ({n} CSV files)")


if __name__ == "__main__":
    main()
