import multiprocessing as mp
import numpy as np
import h5py
import os
import fbpca
from functools import partial
import time
import json
from tqdm import tqdm


def compile_hankel(time_series: np.ndarray, end_index: int, window_size: int, n_windows: int) -> np.ndarray:
    """
    This function constructs a hankel matrix from a 1D time series. Please make sure constructing the matrix with
    the given parameters (end index, window size, etc.) is possible, as this function does no checks due to
    performance reasons.

    :param time_series: 1D array with float values as the time series
    :param end_index: the index (point in time) where the time series starts
    :param window_size: the size of the windows cut from the time series
    :param n_windows: the amount of time series in the matrix
    :return: The hankel matrix with lag one
    """

    # make an empty matrix to place the values
    #
    # almost no faster way:
    # https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy
    hankel = np.empty((window_size, n_windows))

    # go through the time series and make the hankel matrix
    for cx in range(n_windows):
        hankel[:, cx] = time_series[(end_index-window_size-cx):(end_index-cx)]
    return hankel


def process_signal(signal_key: str, window_size: int, n_windows: int, random_rank: int, subspace_repetitions: int,
                   hdf_path: str, result_path='results', svd_function: str = "svd") -> None:

    # check whether the functional parameters are valid
    if svd_function == "svd":
        assert random_rank == -1, f'You specified a random rank of {random_rank}, but you use svd which requires -1.'
        assert subspace_repetitions == -1, f'You specified subspace iterations: {subspace_repetitions} ' \
                                           f'but you use svd which requires -1.'
        svd_function_handle = partial(np.linalg.svd, full_matrices=False)
    elif svd_function == "rsvd":
        assert isinstance(random_rank, int) and random_rank > 0, f'Random rank needs to be a positive ' \
                                                                 f'integer you specified {random_rank}.'
        assert isinstance(subspace_repetitions, int) and subspace_repetitions >= 0, f'Subspace iterations need to be' \
                                                                                    f' a positive integer you ' \
                                                                                    f'specified {subspace_repetitions}.'
        svd_function_handle = partial(fbpca.pca, k=random_rank, raw=True, n_iter=subspace_repetitions)
    else:
        raise ValueError(f'Function name {svd_function} is not available.')

    # open the file and get the signals from it into RAM
    with h5py.File(hdf_path, 'r') as filet:
        signal = filet[signal_key][:]

    # compute the amount of chunks we need to make and check whether we have at least one
    chunk_length = window_size + n_windows - 1
    chunk_number = signal.shape[0]//chunk_length
    if not chunk_number:
        return

    # make the directory path
    signal_dir = os.path.join(result_path, f"window_size_{window_size}")

    # go over the chunks and compute the svd and save the result of the svd within the
    vector_collector = dict()
    vector_collector["process_times"] = np.zeros((chunk_number,))
    for chx in range(chunk_number):

        # create the matrix and compute the decomposition
        hankel = compile_hankel(signal, (chx + 1) * chunk_length, window_size, n_windows)

        # keep track of the computation time
        start_time = time.perf_counter_ns()

        # compute the decomposition
        left_vectors, eigenvalues, _ = svd_function_handle(hankel)

        # keep track of the time it took
        elapsed = time.perf_counter_ns() - start_time
        vector_collector["process_times"][chx] = elapsed

        # make the result float32 for memory savings
        left_vectors = left_vectors.astype("float32")
        eigenvalues = eigenvalues.astype("float32")

        # save into the vector collection
        name = f"Chunk__{chx * chunk_length}_to_{(chx + 1) * chunk_length}"
        vector_collector[f"{name}__left_vectors"] = left_vectors
        vector_collector[f"{name}__eigenvalues"] = eigenvalues

    # save the result in the corresponding files
    name = f"{signal_key}__{window_size}x{n_windows}__{chunk_number}.npz"
    name = os.path.join(signal_dir, name)
    np.savez_compressed(name, **vector_collector)


def main_computing(result_path: str = "result", file_path: str = "UCRArchive_2018.hdf5", decomposition: str = "rsvd"):

    # check whether the result path already exists
    if os.path.isdir(result_path):
        raise ValueError(f"The folder {result_path} already exists and we might overwrite results.")

    # create different window sizes and specify the amount of windows
    window_sizes = [int(ele) for ele in np.ceil(np.geomspace(10, 500, num=10))[::-1]]
    window_number = 100

    # specify the decomposition and the different parameters
    # check whether the functional parameters are valid
    if decomposition == "svd":
        random_rank = -1
        subspace_repetitions = -1
    elif decomposition == "rsvd":
        random_rank = 5
        subspace_repetitions = 2
    else:
        raise ValueError(f'Function name {decomposition} is not available.')

    # get the signals keys from the hdf5 file, so we can iterate over them
    with h5py.File(file_path, 'r') as filet:
        signal_keys = list(filet.keys())

    # make the directory so we can save data
    os.mkdir(result_path)
    for window_size in window_sizes:
        os.mkdir(os.path.join(result_path, f"window_size_{window_size}"))

    # write the config into the data directory
    with open(os.path.join(result_path, 'config.json'), 'w') as filet:
        text = json.dumps({"dataset": file_path,
                           "window_number": window_number,
                           "window_sizes": window_sizes,
                           "results": result_path,
                           })
        filet.write(text)

    # go over different window sizes and keep a progress bar (go from the longest as they have the longest duration)
    for window_size in window_sizes:

        # make the function partial, so we only have one input argument
        function_handle = partial(process_signal,
                                  window_size=window_size,
                                  n_windows=window_number,
                                  random_rank=random_rank,
                                  subspace_repetitions=subspace_repetitions,
                                  hdf_path=file_path,
                                  result_path=result_path,
                                  svd_function=decomposition)
        # make multiprocessing to make use of multicore cpus
        with mp.Pool(mp.cpu_count()) as pp:
            desc = f'SVD with window size {window_size}'
            # as the execution of imap_unordered is greedy, we need to iterate over the return values, but
            # since it returns nothing we can just pass. Unfortunately, the greedy version is the only one
            # which to the best of my knowledge supports unordered execution.
            for result in tqdm(pp.imap_unordered(function_handle, signal_keys), desc=desc, total=len(signal_keys)):
                pass


if __name__ == "__main__":
    main_computing(result_path="result", file_path="UCRArchive_2018.hdf5", decomposition="svd")
