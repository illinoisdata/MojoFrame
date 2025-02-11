import os
import tracemalloc
from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl

from src.utils.timerutil import TPCHTimer

# The TPC-H queries to execute, in range [START_QUERY, END_QUERY]
START_QUERY: int = int(os.environ.get("START_QUERY", 1))
END_QUERY: int = int(os.environ.get("END_QUERY", 22))
# Whether to include data fetching time in the query duration result
INCLUDE_IO: bool = os.environ.get("INCLUDE_IO", False) == "True"
# Whether to record the RAM usage as well
INCLUDE_RAM: bool = os.environ.get("INCLUDE_RAM", False) == "True"
if INCLUDE_RAM:
    # The dictionary with the RAM values
    RAM_USAGE: dict[str, int] = {}
# The filetype of the input data
FILE_TYPE: str = os.environ.get("FILE_TYPE", "csv")
# Dataset directory
DATA_DIR: str = os.environ.get("DATA_DIR", "data/")
# The subdirectory of the data file holding the parquet/feather
DATA_FILE: str = os.environ.get("DATA_FILE", "tiny_tpch/")
# Current directory
CWD: str = os.environ.get("CWD", os.path.dirname(os.path.realpath(__file__)))
# Whether to print the query results while running
SHOW_RESULTS: bool = os.environ.get("SHOW_RESULTS", False) == "True"
# Whether to save the query results to a file
SAVE_RESULTS: bool = os.environ.get("SAVE_RESULTS", False) == "True"
# Whether to log the query timings
LOG_TIMINGS: bool = os.environ.get("LOG_TIMINGS", False) == "True"
# Whether to test the results of the queries for accuracy
TEST_RESULTS: bool = os.environ.get("TEST_RESULTS", False) == "True"
# Absolute path to the data directory
DATASET_BASE_DIR: str = os.path.join(CWD, DATA_DIR, DATA_FILE)
# Absolute path to the expected TPC-H query answers
ANSWERS_BASE_DIR: str = os.path.join(CWD, DATASET_BASE_DIR, "answers/")
# Output directory
OUTPUT_BASE_DIR: str = os.path.join(CWD, "outputs/", DATA_FILE)
if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
# Timings CSV output directory
TIMINGS_FILE: str = os.path.join(CWD, f"{OUTPUT_BASE_DIR}/timings.csv")
# Plots directory
DEFAULT_PLOTS_DIR: str = os.path.join(CWD, f"{OUTPUT_BASE_DIR}/plots")
if not os.path.exists(DEFAULT_PLOTS_DIR):
    os.makedirs(DEFAULT_PLOTS_DIR, exist_ok=True)


def fetch_dataset(path: str) -> pl.LazyFrame:
    """Lazily fetch TPC-H dataset found at path

    Args:
        path (str): the path to the dataset

    Returns:
        pl.LazyFrame: the dataset placed in a polars
        dataframe
    """
    path = f"{path}.{FILE_TYPE}*"
    match FILE_TYPE:
        case "csv":
            scan: pl.LazyFrame = pl.scan_csv(path, cache=False)
        case "parquet":
            scan: pl.LazyFrame = pl.scan_parquet(path, cache=False)
        case "feather":
            scan: pl.LazyFrame = pl.scan_ipc(path, cache=False)
        case _:
            raise IOError(f"File type: {FILE_TYPE} not expected")

    return scan.collect().lazy()


def get_query_answer(query_num: int, base_dir: str = ANSWERS_BASE_DIR) -> pl.LazyFrame:
    """Retrieve the answer to TPC-H query number query_num
    in the form of a polars dataframe

    Args:
        query (int): the TPC-H query number
        base_dir (str, optional): the answers directory. Defaults to ANSWERS_BASE_DIR.

    Returns:
        pl.LazyFrame: the answer to query in a polars
        dataframe
    """
    answer_ldf: pl.LazyFrame = pl.scan_csv(
        source=os.path.join(base_dir, f"{query_num}.csv"),
        separator=",",
        has_header=True,
        try_parse_dates=True,
    )
    answer_schema: pl.Schema = answer_ldf.collect_schema()
    columns: list[str] = answer_schema.names()
    answer_ldf = answer_ldf.select(
        [pl.col(c).alias(c.strip()) for c in columns]
    ).with_columns([pl.col(pl.datatypes.Utf8).str.strip_chars().name.keep()])

    return answer_ldf


def test_results(query_num: int, result_df: pl.DataFrame) -> bool:
    """Test the results of TPC-H query number query_num

    Args:
        query_num (int): the TPC-H query number to test
        result_df (pl.DataFrame): the results from running
        the query

    Returns:
        bool: whether the passed dataframe is the correct answer
        or not
    """
    with TPCHTimer(f"Query {query_num} result testing"):
        answer: pl.DataFrame = get_query_answer(query_num).collect()
        result_df = (
            result_df.lazy()
            .with_columns([pl.col(pl.datatypes.Utf8).str.strip_chars().name.keep()])
            .collect()
        )
        correct = result_df.equals(answer)

        if correct:
            print(f"QUERY {query_num} FAILED")

        return correct


def get_line_item_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the lineitem data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "lineitem"))


def get_orders_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the orders data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "orders"))


def get_customer_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the customer data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "customer"))


def get_region_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the region data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "region"))


def get_nation_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the nation data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "nation"))


def get_supplier_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the supplier data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "supplier"))


def get_part_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the part data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "part"))


def get_part_supp_ds(base_dir: str = DATASET_BASE_DIR) -> pl.LazyFrame:
    """Fetch the partsupp data items

    Args:
        base_dir (str, optional): base dataset directory. Defaults to DATASET_BASE_DIR.

    Returns:
        pl.LazyFrame: polars dataframe containing the fetched data
    """
    return fetch_dataset(os.path.join(base_dir, "partsupp"))


def write_row(
    query_num: str,
    load_time: float,
    exec_time: float,
    version: str,
    success: bool = True,
) -> None:
    """Write the timings results for TPC-H query number query_num
    to the TIMINGS_FILE in a CSV format.

    Args:
        query_num (str): the TPC-H query number
        load_time (float): The data loading time for the query
        exec_time (float): The query execution time
        version (str): The polars version
        success (bool, optional): Whether the query was a success or not.
        Defaults to True.
    """
    with open(TIMINGS_FILE, "a") as f:
        if f.tell() == 0:
            f.write("version,query_number,load_time,exec_time,include_io,success\n")
        f.write(f"{version},{query_num},{load_time},{exec_time},{INCLUDE_IO},{success}")


def run_query(query_num: int, lp: pl.LazyFrame):
    """Execute TPC-H query number query_num

    Args:
        query_num (int): query number (1-22)
        lp (pl.LazyFrame): polars lazyframe for processing
    """
    if INCLUDE_RAM:
        tracemalloc.start()

    with TPCHTimer(name=f"Query {query_num} execution", logging=False):
        result: pl.DataFrame = lp.collect()

    if INCLUDE_RAM:
        _, peak = tracemalloc.get_traced_memory()
        peak -= tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()
        RAM_USAGE[f"Query {query_num} peak RAM"] = peak

    load_time: float = TPCHTimer.times[f"Data load time for Query {query_num}"]
    exec_time: float = TPCHTimer.times[f"Query {query_num} execution"]
    if INCLUDE_IO:
        exec_time += load_time
        load_time = 0.0

    success: bool = test_results(query_num, result) if TEST_RESULTS else True

    if LOG_TIMINGS:
        write_row(
            query_num=str(query_num),
            load_time=load_time,
            exec_time=exec_time,
            version=pl.__version__,
            success=success,
        )

    if SHOW_RESULTS:
        print(result)

    if SAVE_RESULTS:
        result.write_csv(f"{OUTPUT_BASE_DIR}/query{query_num}.csv")


def generate_query_plot():
    """Generate a bar plot of the query timings
    and output it to the plots directory inside
    the output directory
    """
    data_load_time: list[float] = []
    execution_time: list[float] = []
    for query_num in range(START_QUERY, END_QUERY + 1):
        if f"Query {query_num} execution" in TPCHTimer.times:
            execution_time.append(TPCHTimer.times[f"Query {query_num} execution"])
            data_load_time.append(
                TPCHTimer.times[f"Data load time for Query {query_num}"]
            )
        else:
            execution_time.append(0)
            data_load_time.append(0)

    fig, ax = plt.subplots()

    # If INCLUDE_IO is set, concatenate the data load times and query
    # execution times into one, otherwise have the bar graphs stacked
    concatenated_times: list[float] = [
        load + execute for load, execute in zip(data_load_time, execution_time)
    ]
    if INCLUDE_IO:
        ax.bar(range(START_QUERY, END_QUERY + 1), concatenated_times)
        for i, height in zip(range(START_QUERY, END_QUERY + 1), concatenated_times):
            ax.text(i, height * 1.05, f"{height:.2f}", ha="center", fontsize=10)
    else:
        ax.bar(range(START_QUERY, END_QUERY + 1), data_load_time)
        ax.bar(
            range(START_QUERY, END_QUERY + 1),
            execution_time,
            bottom=data_load_time,
        )
        ax.legend(["Data Loading Time", "Query Execution Time"])
        for i, height_load, height_exec in zip(
            range(START_QUERY, END_QUERY + 1), data_load_time, execution_time
        ):
            ax.text(i, height_load / 2, f"{height_load:.2f}", ha="center", fontsize=10)
            ax.text(
                i,
                height_load + height_exec / 2,
                f"{height_exec:.2f}",
                ha="center",
                fontsize=10,
            )

    ax.set_ylim(
        0,
        max(concatenated_times) * 1.1,
    )

    ax.set_xlabel("TPC-H Query")
    ax.set_ylabel("Execution Time (s)")
    ax.set_xticks(range(START_QUERY, END_QUERY + 1))
    ax.tick_params(bottom=False, left=False)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("lightgray")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="lightgray")
    ax.xaxis.grid(False)
    ax.set_title("TPC-H Query Execution Time for Polars")
    fig.savefig(
        os.path.join(
            DEFAULT_PLOTS_DIR,
            datetime.now().strftime("%m-%d-%y_%H:%M"),
        ),
        dpi=200,
    )
