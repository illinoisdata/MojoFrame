import argparse
import importlib
import os
import signal
import sys
from types import FrameType


def keyboard_interrupt_handler(signal_num: int, stack_frame: FrameType):
    """A handler function used to nicely handle a
    Ctrl-C input

    Args:
        signal_num (int): the system signal int
        stack_frame (FrameType): the current execution frame
    """
    # TODO: Implement file cleanup
    print("Interrupted. Performing file cleanup...")
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
    os.environ["CWD"] = os.path.dirname(os.path.realpath(__file__))
    for key, value in vars(args).items():
        unpacked = value if not isinstance(value, list) else value[0]
        os.environ[key.upper()] = (
            str(unpacked) if not isinstance(unpacked, str) else unpacked
        )

    # Import here so that the local environment variables are
    # made AFTER we set them here.
    from src.utils.utils import generate_query_plot

    for i in range(args.start_query, args.end_query + 1):
        print(f"Starting query {i}...")
        query = importlib.import_module(f"src.queries.query{i}")
        query.q()
        print(f"Query {i} finished.")

    if args.write_plot:
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
        "-i",
        "--include_io",
        help="Whether to include data fetching time in the query duration result",
        nargs=1,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-t",
        "--test_results",
        help="Whether to test the query results against the correct answers",
        nargs=1,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-r",
        "--include_ram",
        help="Whether to record ram usage during queries",
        nargs=1,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-sr",
        "--show_results",
        help="Whether to print the query results to standard output",
        nargs=1,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-svr",
        "--save_results",
        help="Whether to save the results to output_dir",
        nargs=1,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-l",
        "--log_timings",
        help="Whether to save the timing results to output_dir",
        nargs=1,
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-pl",
        "--write_plot",
        help="Whether to save a timings plot to output_dir",
        nargs=1,
        type=bool,
        default=False,
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

    args = parser.parse_args()

    main(args)
