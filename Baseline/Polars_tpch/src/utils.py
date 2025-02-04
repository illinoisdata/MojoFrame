import os

# The filetype of the input data
FILE_TYPE = os.environ.get("FILE_TYPE", "parquet")
# TODO : ADD DATA PATH
# Dataset directory
DATA_DIR = os.environ.get("DATA_DIR", "../data")
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
