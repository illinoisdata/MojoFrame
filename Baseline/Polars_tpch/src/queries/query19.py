import polars as pl

from src.utils import utils
from src.utils.timerutil import TPCHTimer

Q_NUM = 19


def q():
    with TPCHTimer('Data load time for Query {Q_NUM}'):
        pass

    q_final = pl.LazyFrame()

    utils.run_query(Q_NUM, q_final)
