from mpi4py import MPI
from pandas import DataFrame
from geopandas import GeoDataFrame
from gc import collect


def redistribute_dataframes(dataframe: DataFrame or GeoDataFrame, comm: MPI.Comm) -> DataFrame or GeoDataFrame:
    """
    Redistribute dataframes across MPI ranks, where some ranks have empty dataframes and others have large dataframes.
    The function splits large dataframes and sends half of the data to ranks with empty dataframes.

    Parameters
    ----------
    dataframe : pandas.DataFrame or geopandas.GeoDataFrame
        The dataframe held by each rank. Some ranks may have empty dataframes, while others have large ones.

    comm : mpi4py.MPI.Comm
        The MPI communicator used to coordinate the redistribution across ranks.

    Returns
    -------
    dataframe : pandas.DataFrame or geopandas.GeoDataFrame
        The redistributed dataframe for each rank. Ranks that initially had large dataframes
        will have smaller ones, and ranks that initially had empty dataframes will receive data.

    Notes
    -----
    - This function assumes that all ranks have either pandas.DataFrame or geopandas.GeoDataFrame.
    - The redistribution plan is created by rank 0 and then executed by all ranks.
    - The function currently splits the large dataframe into two halves; additional strategies
      might be required for different splitting strategies or larger-scale data management.

    """
    rank = comm.Get_rank()

    # All ranks communicate their DataFrame size to the root
    df_size = len(dataframe)
    # print(f"Rank {rank}: df_size 1: {df_size} {dataframe.head()}")
    all_sizes = comm.gather(df_size, root=0)

    if rank == 0:
        # Root identifies which ranks have empty DataFrames and which have large ones
        empty_ranks = [i for i, size in enumerate(all_sizes) if size == 0]
        large_ranks = sorted([(i, size) for i, size in enumerate(all_sizes) if size > 0], key=lambda x: x[1],
                             reverse=True)

        # Create a redistribution plan
        redistribution_plan = []
        for empty_rank in empty_ranks:
            if not large_ranks:
                break
            large_rank, large_size = large_ranks.pop(0)
            split_size = large_size // 2
            redistribution_plan.append((large_rank, empty_rank, split_size))

    else:
        redistribution_plan = None

    # Send the redistribution plan to all ranks
    redistribution_plan = comm.bcast(redistribution_plan, root=0)

    # Execute redistribution according to the plan
    for source_rank, target_rank, split_size in redistribution_plan:
        if rank == source_rank:
            # Split the DataFrame into two parts
            df_to_send = dataframe.iloc[:split_size]
            dataframe = dataframe.iloc[split_size:]
            # Send half to the target_rank
            comm.send(df_to_send, dest=target_rank)
        elif rank == target_rank:
            # Receive half of the DataFrame from the source_rank
            dataframe = comm.recv(source=source_rank)
    collect()
    # print(f"Rank {rank}: df_size 2: {len(dataframe)}, {dataframe.head()}")
    return dataframe


def redistribute_dataframes_with_balancing(dataframe: DataFrame or GeoDataFrame, comm: MPI.Comm,
                                           small_threshold: float = 0.5) -> DataFrame or GeoDataFrame:
    """
    Redistribute dataframes across MPI ranks, where some ranks have empty dataframes, others have large dataframes,
    and others have small dataframes that also receive additional data to balance the distribution.

    Parameters
    ----------
    dataframe : pandas.DataFrame or geopandas.GeoDataFrame
        The dataframe held by each rank. Some ranks may have empty dataframes, while others have large or small ones.

    comm : mpi4py.MPI.Comm
        The MPI communicator used to coordinate the redistribution across ranks.

    small_threshold : float, optional
        The fraction of the average DataFrame size below which a DataFrame is considered small. Default is 0.5 (50%).

    Returns
    -------
    dataframe : pandas.DataFrame or geopandas.GeoDataFrame
        The redistributed dataframe for each rank. Ranks that initially had large dataframes
        will have smaller ones, and ranks that initially had empty or small dataframes will receive additional data.

    Notes
    -----
    - This function assumes that all ranks have either pandas.DataFrame or geopandas.GeoDataFrame.
    - The redistribution plan is created by rank 0 and then executed by all ranks.
    - The function currently splits the large dataframe into two halves; additional strategies
      might be required for different splitting strategies or larger-scale data management.
    """
    rank = comm.Get_rank()

    # All ranks communicate their DataFrame size to the root
    df_size = len(dataframe)
    all_sizes = comm.gather(df_size, root=0)

    if rank == 0:
        # Calculate average size
        average_size = sum(all_sizes) / len(all_sizes)

        # Define small and empty ranks
        small_ranks = [i for i, size in enumerate(all_sizes) if 0 < size < small_threshold * average_size]
        empty_ranks = [i for i, size in enumerate(all_sizes) if size == 0]

        # Sort large ranks by size
        large_ranks = sorted([(i, size) for i, size in enumerate(all_sizes) if size > average_size], key=lambda x: x[1],
                             reverse=True)

        # Create a redistribution plan
        redistribution_plan = []
        for target_rank in empty_ranks + small_ranks:
            if not large_ranks:
                break
            large_rank, large_size = large_ranks.pop(0)
            split_size = large_size // 2
            redistribution_plan.append((large_rank, target_rank, split_size))
            # Reinsert the large_rank if it still has more data to give after splitting
            new_size = large_size - split_size
            if new_size > average_size:
                large_ranks.append((large_rank, new_size))
                large_ranks = sorted(large_ranks, key=lambda x: x[1], reverse=True)

    else:
        redistribution_plan = None

    # Send the redistribution plan to all ranks
    redistribution_plan = comm.bcast(redistribution_plan, root=0)

    # Execute redistribution according to the plan
    for source_rank, target_rank, split_size in redistribution_plan:
        if rank == source_rank:
            # Split the DataFrame into two parts
            df_to_send = dataframe.iloc[:split_size]
            dataframe = dataframe.iloc[split_size:]
            # Send half to the target_rank
            comm.send(df_to_send, dest=target_rank)
        elif rank == target_rank:
            # Receive half of the DataFrame from the source_rank
            dataframe = comm.recv(source=source_rank)

    collect()
    return dataframe
