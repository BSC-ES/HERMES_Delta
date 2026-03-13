from pandas import concat, Series, DataFrame
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def sum_result(result: dict, path: str) -> None:
    """
    Computes the partial sum of each variable's data in the `result` dictionary across multiple MPI ranks.
    Each rank calculates the sum of its assigned data, and these results are gathered and combined into a
    DataFrame in `RANK == 0`, with each column representing a rank's partial results. A "TOTAL" column
    is added to store the sum of all ranks for each variable.

    Parameters
    ----------
    path : str
        Proxy path.
    result : dict
        Dictionary where keys are variable names and values are dictionaries containing data to sum,
        with `data` being a numpy-compatible array or list that can be summed.

    Returns
    -------
    None
        The function prints the DataFrame with partial results and a TOTAL column on `RANK == 0`.
    """

    # Barrier to synchronize ranks before starting
    COMM.Barrier()

    # Initialize a Series to store the partial sums for this rank
    aux_result = Series({var_name: var_info['data'].sum() for var_name, var_info in result.items()}, name=RANK)

    # Gather all partial results to the root process
    gathered_results = COMM.gather(aux_result, root=0)

    # In RANK == 0, combine results into a DataFrame
    if RANK == 0:
        # Concatenate all Series into a DataFrame with each column representing a different rank's results
        result_df = concat(gathered_results, axis=1)
        result_df.columns = [f"Rank_{i}" for i in range(SIZE)]

        # Add a "TOTAL" column that sums across all ranks for each variable
        result_df["TOTAL"] = result_df.sum(axis=1)

        print("EVALUATING RESULT")
        print(result_df)
        result_df.to_csv(path.replace('proxy.nc', f"result_{str(SIZE).zfill(2)}.csv"))

    # Synchronize again before finishing
    COMM.Barrier()
    return None


def sum_emissions(emissions: DataFrame, path):
    emissions.to_csv(path.replace('proxy.nc', f"result_{str(RANK).zfill(2)}_{str(SIZE).zfill(2)}.csv"))
