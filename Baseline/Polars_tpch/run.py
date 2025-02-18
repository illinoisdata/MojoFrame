import argparse
import importlib
import os
import shutil
import signal
import sys
from datetime import datetime
from types import FrameType


def clear_cache():
    """Clear __pycache__ files for fair benchmarking"""

    print("> Clearing caches...")
    if os.path.exists("./src/queries/__pycache__"):
        shutil.rmtree("./src/queries/__pycache__")
    if os.path.exists("./src/utils/__pycache__"):
        shutil.rmtree("./src/utils/__pycache__")


def keyboard_interrupt_handler(signal_num: int, stack_frame: FrameType):
    """A handler function used to nicely handle a
    Ctrl-C input

    Args:
        signal_num (int): the system signal int
        stack_frame (FrameType): the current execution frame
    """
    # TODO: Implement file cleanup
    print(
        f">> Interrupted << {datetime.now().strftime('(%d-%m-%Y : %H:%M)')} Performing file cleanup..."
    )
    clear_cache()
    sys.exit(130)


signal.signal(signal.SIGINT, keyboard_interrupt_handler)  # type: ignore


class DefaultsAndTypesFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
):
    """Custom formatter inheriting from both the ArgumentDefaultsHelpFormatter
    and the MetavarTypeHelpFormatter of the argparse module to display both type
    and default information in the help message
    """


def main(args):
    clear_cache()
    os.environ["CWD"] = os.path.dirname(os.path.realpath(__file__))
    for key, value in vars(args).items():
        unpacked = value if not isinstance(value, list) else value[0]
        os.environ[key.upper()] = (
            str(unpacked) if not isinstance(unpacked, str) else unpacked
        )

    # Import here so that the local environment variables are
    # made AFTER we set them here.
    from src.utils.utils import generate_query_plot, generate_ram_plot

    for i in range(args.start_query, args.end_query + 1):
        for trial in range(args.num_trials):
            print(f">> Starting query {i} Trial {trial}...")
            # Dynamically import module
            query = importlib.import_module(f"src.queries.query{i}")
            query.q()
            # Unimport module so all trials are fair
            del query
            print(f">> Query {i} Trial {trial} finished.")
            clear_cache()
        print()

    if args.write_plot:
        if args.include_ram:
            generate_ram_plot()
        else:
            generate_query_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python ./run.py",
        formatter_class=DefaultsAndTypesFormatter,
    )
    parser.add_argument(
        "start_query",
        help="The TPC-H query to start with (inclusive)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "end_query",
        help="The TPC-H query to end with (inclusive)",
        type=int,
        default=22,
    )
    parser.add_argument(
        "-n",
        "--num_trials",
        help="The number of times to run each individual query",
        type=int,
        default=5,
        choices=[i for i in range(1, 21)],
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        help="the directory (relative to the project root) with the input data",
        nargs=1,
        type=str,
        default="data/",
    )
    parser.add_argument(
        "-f",
        "--data_file",
        help="Subdirectory within data_dir which hold the actual data files",
        nargs=1,
        type=str,
        default="tiny_tpch/",
    )
    parser.add_argument(
        "-ft",
        "--file_type",
        help="The input data filetype",
        choices=["csv", "parquet", "feather"],
        nargs=1,
        type=str,
        default="csv",
    )
    parser.add_argument(
        "-i",
        "--include_io",
        help="Whether to include data fetching time in the query duration result",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--test_results",
        help="Whether to test the query results against the correct answers",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--include_ram",
        help="Whether to record ram usage during queries",
        action="store_true",
    )
    parser.add_argument(
        "-sr",
        "--show_results",
        help="Whether to print the query results to standard output",
        action="store_true",
    )
    parser.add_argument(
        "-svr",
        "--save_results",
        help="Whether to save the results to output_dir",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--log_timings",
        help="Whether to save the timing results to output_dir",
        action="store_true",
    )
    parser.add_argument(
        "-pl",
        "--write_plot",
        help="Whether to save a timings plot to output_dir",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
