import os

import polars as pl

# from timerutil import TPCHTimer

# The filetype of the input data
FILE_TYPE = os.environ.get("FILE_TYPE", "parquet")
# TODO : ADD DATA PATH
# Dataset directory``
DATA_DIR = os.environ.get("DATA_DIR", "../../../Data")
# Current directory
CWD = os.path.dirname(os.path.realpath(__file__))
# Whether to print the query results while running
SHOW_RESULTS = bool(os.environ.get("SHOW_RESULTS", False))
# Whether to log the query timings
LOG_TIMINGS = bool(os.environ.get("LOG_TIMINGS", False))
# Whether to write the TPC-H query timings graph
WRITE_PLOT = bool(os.environ.get("WRITE_PLOT", False))
# Whether to test the results of the queries for accuracy
TEST_RESULTS = bool(os.environ.get("TEST_RESULTS", False))
# Data scale
SCALE_FACTOR = os.environ.get("SCALE_FACTOR", "1")
# Data partition
PARTITION = os.environ.get("PARTITION", "1")
# Directory to the specific scale/partition dataset
DATASET_BASE_DIR = os.path.join(
    DATA_DIR, f"scale={SCALE_FACTOR}/partition={PARTITION}/parquet/"
)
# Directory for the specific scale/partition answers
ANSWERS_BASE_DIR = os.path.join(DATA_DIR, f"scale={SCALE_FACTOR}/original/")
# Output directory
OUTPUT_BASE_DIR = os.path.join(
    CWD, f"outputs/scale={SCALE_FACTOR}/partition={PARTITION}"
)
if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
# Timings CSV output directory
TIMINGS_FILE = os.path.join(CWD, f"{OUTPUT_BASE_DIR}/timings.csv")
# Plots directory
DEFAULT_PLOTS_DIR = os.path.join(CWD, f"{OUTPUT_BASE_DIR}/plots")


def fetch_dataset(path: str) -> pl.LazyFrame:
    """Fetch TPC-H dataset found at path

    Args:
        path (str): the path to the dataset

    Returns:
        pl.LazyFrame: the dataset placed in a polars
        dataframe
    """
    pass


def get_query_answer(query: int, base_dir: str = ANSWERS_BASE_DIR) -> pl.LazyFrame:
    """Retrieve the answer to query in the form
    of a polars dataframe

    Args:
        query (int): the TPC-H query number
        base_dir (str, optional): the answers directory. Defaults to ANSWERS_BASE_DIR.

    Returns:
        pl.LazyFrame: the answer to query in a polars
        dataframe
    """
    pass


def test_results(query: int, result_df: pl.DataFrame):
    """Test the results of a query

    Args:
        query (int): the TPC-H query number to test
        result_df (pl.DataFrame): the results from running
        the query
    """
    pass


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
