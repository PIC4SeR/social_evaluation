import pickle
from pathlib import Path
from social_evaluation.data_filtering.filters import create_filtered_data
import argcomplete, argparse
import pickle


def main():
    package_path = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Evaluate social metrics")
    parser.add_argument(
        "--pickle-file",
        type=Path,
        help="Path to the file to be filtered",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    file_to_use = args.pickle_file
    print(f"Filtering file {file_to_use}")
    with open(file_to_use, "rb") as file:
        data = pickle.load(file)
    filtered_data = {
        key: create_filtered_data(original, window=5, method="ema")
        for key, original in data.items()
    }
    new_file = file_to_use.with_name(
        file_to_use.name.replace(".pickle", "_filtered.pickle")
    )
    print(f"Saving filtered data to {new_file}")
    with open(new_file, "wb") as f_out:
        pickle.dump(filtered_data, f_out)


if __name__ == "__main__":
    main()
