import argparse
import signal
import sys
from subprocess import run
from types import FrameType

from src.utils.timerutil import TPCHTimer


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


signal.signal(signal.SIGINT, keyboard_interrupt_handler)


def main(args):
    for i in range(args.start_query, args.end_query + 1):
        run([sys.executable, "-m", f"src.queries.query{i}"])
    print(TPCHTimer.times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("polars_tpch")
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
        nargs=1,
        help="the directory (relative to the project root) with the input data",
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        nargs=1,
        help="the directory (relative to the project root) for file output",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--scale",
        nargs=1,
        help="The scale factor of the data, used for locating input data.",
        type=int,
    )
    parser.add_argument(
        "-p",
        "--partition",
        nargs=1,
        help="The partition of the data, used for locating input data.",
        type=int,
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
        "--record_ram",
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
    parser.add_argument(
        "-f",
        "--file_type",
        help="The input data filetype",
        choices=["parquet", "feather"],
        default="parquet",
    )

    args = parser.parse_args()

    main(args)
