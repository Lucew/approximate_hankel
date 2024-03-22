# make sure to find all files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import methods of interest
import utils.fastHankel as fh
from changepointComparison import transform
from preprocessing.changepoint_simulator import ChangeSimulator
from threadpoolctl import threadpool_limits
import numpy as np
import numba as nb
import timeit
try:
    import torch
    is_torch_available = True
except ModuleNotFoundError:
    is_torch_available = False
import concurrent.futures


def print_table(my_tuples: list[dict]):
    """
    Pretty print a list of dictionaries (my_dict) as a dynamically sized table.
    """

    # get the column list and check whether it is available in all dicts
    col_keys = [key for key in my_tuples[0].keys()]
    cols = {key: [f"{key}"] for key in col_keys}
    for idx, tupled in enumerate(my_tuples):

        # check whether they are available
        assert all(ele in tupled for ele in col_keys), (f"Tuple {idx} does not have the expected cols: {col_keys},"
                                                        f" it has: {list(tupled.keys())}")

        # insert the tuples into the columns
        for key, val in cols.items():
            val.append(f"{tupled[key]}")

    # check the column width for each column
    col_width = []
    for key in col_keys:
        col_width.append((key, max(map(len, cols[key]))))

    # print the stuff
    # formatting = "|" + "|".join(f"{{{idx}: <{width}" for idx, (_, width) in enumerate(col_width))
    header = "|".join(key.center(width+5) for key, width in col_width)
    print(header)
    print("-"*len(header))
    for tupled in my_tuples:
        text = "|".join(f"{tupled[key]}".ljust(width+5) for key, width in col_width)
        print(text)


def probe_hankel_fft_from_matrix(time_series, window_length: int, window_number: int, comment: str):

    # get the final index of the time series
    end_idx = window_length + window_number - 1

    # get the hankel matrix
    hankel_matrix = fh.compile_hankel_parallel(time_series, end_idx, window_length, window_number, lag=1)

    # get the fft representation from the matrix
    hankel_fft1, fft_len1, _ = fh.hankel_fft_from_matrix(hankel_matrix)

    # get the fft representation directly from the signal
    hankel_fft2, fft_len2, _ = fh.get_fast_hankel_representation(time_series, end_idx, window_length, window_number,
                                                                 lag=1)
    assert fft_len1 == fft_len2, "FFT length are different...."
    return fh.evaluate_closeness(hankel_fft1, hankel_fft2, comment)


def probe_fast_hankel_matmul(hankel_repr, l_windows, fft_len, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fh.fast_hankel_matmul(hankel_repr, l_windows, fft_len, other_matrix, lag)
    way_two = fh.normal_hankel_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_hankel_left_matmul(hankel_repr, l_windows, fft_len, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fh.fast_hankel_left_matmul(hankel_repr, l_windows, fft_len, other_matrix, lag)
    way_two = fh.normal_hankel_left_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_parallel_hankel_matmul(hankel_repr, l_windows, fft_len, hankel_matrix, other_matrix, lag, threadpool,
                                      comment: str):
    way_one = fh.fast_parallel_hankel_matmul(hankel_repr, l_windows, fft_len, other_matrix, lag, threadpool)
    way_two = fh.normal_hankel_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_parallel_hankel_left_matmul(hankel_repr, n_windows, fft_len, hankel_matrix, other_matrix, lag,
                                           threadpool, comment: str):
    way_one = fh.fast_parallel_hankel_left_matmul(hankel_repr, n_windows, fft_len, other_matrix, lag, threadpool)
    way_two = fh.normal_hankel_left_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_numba_hankel_matmul(hankel_repr, l_windows, fft_len, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fh.fast_numba_hankel_matmul(hankel_repr, l_windows, fft_len, other_matrix, lag)
    way_two = fh.normal_hankel_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_numba_hankel_left_matmul(hankel_repr, n_windows, fft_len, hankel_matrix, other_matrix, lag,
                                        comment: str):
    way_one = fh.fast_numba_hankel_left_matmul(hankel_repr, n_windows, fft_len, other_matrix, lag)
    way_two = fh.normal_hankel_left_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_convolve_hankel_matmul(hankel_repr, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fh.fast_convolve_hankel_matmul(hankel_repr, other_matrix, lag)
    way_two = fh.normal_hankel_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_fftconvolve_hankel_matmul(hankel_repr, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fh.fast_fftconv_hankel_matmul(hankel_repr, other_matrix, lag)
    way_two = fh.normal_hankel_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_fftconvolve_hankel_left_matmul(hankel_repr, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fh.fast_fftconv_hankel_left_matmul(hankel_repr, other_matrix, lag)
    way_two = fh.normal_hankel_left_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_torch_hankel_matmul(hankel_fft, hankel_matrix, other_matrix, other_matrix_torch, lag, comment: str):
    way_one = fh.fast_torch_hankel_matmul(hankel_fft, other_matrix_torch, lag)
    way_two = fh.normal_hankel_matmul(hankel_matrix, other_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def probe_fast_hankel_inner_product(hankel_repr, hankel_matrix, l_windows, n_windows, lag, comment: str):
    way_one = fh.fast_hankel_inner(hankel_repr, l_windows, n_windows, lag)
    way_two = fh.normal_hankel_inner(hankel_matrix)
    return fh.evaluate_closeness(way_one, way_two, comment)


def test_parallelization():
    # define a window length
    window_length = 5000

    # define the function of interest
    function = "naive rsvd"

    # define the number of threads
    threads = [1, 2, 4]

    # compute the parameters
    lag = window_length // 3
    sig_length = 2*window_length - 1 + lag
    rnd_state = np.random.RandomState(5)

    # go through multiple random seeds
    nb.set_num_threads(4)
    for idx in range(10):

        # simulate a signal
        sig_gen = ChangeSimulator(sig_length, window_length+lag//2, rnd_state)
        transform(sig_gen.mean_change(), window_length, window_length, lag, sig_length, function, rnd_state)

        # test the hankel creation scripts
        sig = sig_gen.mean_change()
        hankel_future_naive = fh.compile_hankel_naive(sig, sig_length, window_length, window_length, safe=True)
        hankel_future_parallel = fh.compile_hankel_parallel(sig, sig_length, window_length, window_length, safe=True)
        assert np.array_equal(hankel_future_naive, hankel_future_parallel), 'Creation of hankel matrix is at odds.'

    # run the transformation for different thread sizes
    for thr in threads:
        with threadpool_limits(limits=thr):
            nb.set_num_threads(thr)
            tmp = lambda: transform(sig_gen.mean_change(), window_length, window_length, lag, sig_length, function, rnd_state)
            print(f'Decomposition {thr} threads:', timeit.timeit(tmp, number=10))
            tmp = lambda: fh.compile_hankel_parallel(sig_gen.mean_change(), sig_length, window_length, window_length)
            print(f'Hankel parallel construction {thr} threads:', timeit.timeit(tmp, number=100))
            tmp = lambda: fh.compile_hankel_naive(sig_gen.mean_change(), sig_length, window_length, window_length)
            print(f'Hankel naive construction {thr} threads:', timeit.timeit(tmp, number=100))
            print()


def test_matmul():
    # define some window length
    limit_threads = 12
    l_windows = 5000
    n_windows = 5000
    lag = 1
    run_num = 500

    # create a time series of a certain length
    n = 300000
    # ts = np.random.uniform(size=(n,))*1000
    ts = np.linspace(0, n, n+1)

    # create a matrix to multiply by
    k = 15
    multi = np.random.uniform(size=(n_windows, k))

    multi2 = np.random.uniform(size=(k, l_windows))

    # get the final index of the time series
    end_idx = l_windows + lag * (n_windows - 1)

    # get both hankel representations
    hankel_rfft, fft_len, signal = fh.get_fast_hankel_representation(ts, end_idx, l_windows, n_windows, lag)
    hankel = fh.compile_hankel_parallel(ts, end_idx, l_windows, n_windows, lag)

    # use the torch function
    if is_torch_available:
        torch_sig = torch.from_numpy(signal)
        torch_sig = torch_sig[None, None, :]
        torch_mat = torch.from_numpy(multi)
        torch_mat = torch_mat[:, None, :]
        torch_mat = torch_mat.transpose(0, 2)

    # create the threadpool executor for the parallel execution
    pool = concurrent.futures.ThreadPoolExecutor(limit_threads)
    nb.set_num_threads(limit_threads)

    # test the faster multiplication
    results = list()
    results.append(probe_fast_hankel_matmul(hankel_rfft, l_windows, fft_len, hankel, multi, lag, 'Matmul working?'))
    results.append(probe_fast_hankel_left_matmul(hankel_rfft, n_windows, fft_len, hankel, multi2, lag, 'Left Matmul working?'))

    results.append(probe_fast_parallel_hankel_matmul(hankel_rfft, l_windows, fft_len, hankel, multi, lag, pool, 'Parallel Matmul working?'))
    results.append(probe_fast_parallel_hankel_left_matmul(hankel_rfft, n_windows, fft_len, hankel, multi2, lag, pool, 'Parallel Left Matmul working?'))

    results.append(probe_fast_numba_hankel_matmul(hankel_rfft[:, 0], l_windows, fft_len, hankel, multi, lag,'Parallel numba Matmul working?'))
    results.append(probe_fast_numba_hankel_left_matmul(hankel_rfft[:, 0], n_windows, fft_len, hankel, multi2, lag,'Parallel numba Left Matmul working?'))
    if is_torch_available:
        results.append(probe_fast_torch_hankel_matmul(torch_sig, hankel, multi, torch_mat, lag, 'Matmul torch working?'))
        results.append(probe_fast_convolve_hankel_matmul(signal, hankel, multi, lag, 'Matmul convolve working?'))

    results.append(probe_fast_fftconvolve_hankel_matmul(signal, hankel, multi, lag, 'Matmul fftconvolve working?'))
    results.append(probe_fast_fftconvolve_hankel_left_matmul(signal, hankel, multi2, lag, 'Left Matmul fftconvolve working?'))

    results.append(probe_hankel_fft_from_matrix(signal, l_windows, n_windows, "FFT representation from Hankel matrix?"))
    # results.append(probe_fast_hankel_inner_product(signal, hankel, l_windows, n_windows, lag, 'Inner product working?'))
    print_table(results)

    # check for execution time of both approaches
    print()
    header = f"Measure some times for {run_num} repetitions using {limit_threads} threads and hankel of size {l_windows}*{n_windows}"
    print(header)
    print("-"*len(header))
    with threadpool_limits(limits=limit_threads):

        print("Times for rfft signal:")
        rfft_time = timeit.timeit(lambda: fh.get_fast_hankel_representation(ts, end_idx, l_windows, n_windows, lag), number=run_num) / run_num * 1000
        print(rfft_time)

        print("Times for Own:")
        print(timeit.timeit(lambda: fh.fast_hankel_matmul(hankel_rfft, l_windows, fft_len, multi, lag, workers=limit_threads), number=run_num)/run_num*1000+rfft_time)
        print(timeit.timeit(lambda: fh.fast_hankel_left_matmul(hankel_rfft, n_windows, fft_len, multi2, lag, workers=limit_threads), number=run_num) / run_num * 1000+rfft_time)

        print("Times for Own parallel:")
        print(timeit.timeit(lambda: fh.fast_parallel_hankel_matmul(hankel_rfft, l_windows, fft_len, multi, lag, threadpool=pool), number=run_num) / run_num * 1000 + rfft_time)
        print(timeit.timeit(lambda: fh.fast_parallel_hankel_left_matmul(hankel_rfft, n_windows, fft_len, multi2, lag, threadpool=pool), number=run_num) / run_num * 1000 + rfft_time)

        print("Times for numba parallel:")
        print(timeit.timeit(lambda: fh.fast_numba_hankel_matmul(hankel_rfft[:, 0], l_windows, fft_len, multi, lag), number=run_num) / run_num * 1000 + rfft_time)
        print(timeit.timeit(lambda: fh.fast_numba_hankel_left_matmul(hankel_rfft[:, 0], n_windows, fft_len, multi2, lag), number=run_num) / run_num * 1000 + rfft_time)

        print("Times for FFTconv:")
        print(timeit.timeit(lambda: fh.fast_fftconv_hankel_matmul(signal, multi, lag, workers=limit_threads), number=run_num) / run_num * 1000)
        print(timeit.timeit(lambda: fh.fast_fftconv_hankel_left_matmul(signal, multi2, lag, workers=limit_threads), number=run_num) / run_num * 1000)
        # if is_torch_available:
            # print(timeit.timeit(lambda: fh.fast_torch_hankel_matmul(torch_sig, torch_mat, lag), number=run_num) / run_num * 1000)
            # print(timeit.timeit(lambda: fh.fast_convolve_hankel_matmul(signal, multi, lag), number=run_num) / run_num * 1000)

        print("Times for Naive:")
        print(timeit.timeit(lambda: fh.normal_hankel_matmul(hankel, multi), number=run_num)/run_num*1000)
        print(timeit.timeit(lambda: fh.normal_hankel_left_matmul(hankel, multi2), number=run_num) / run_num * 1000)

        print("Times for Naive inner Product:")
        # print(timeit.timeit(lambda: fh.normal_hankel_inner(hankel), number=run_num) / run_num * 1000)
        # print(timeit.timeit(lambda: fh.fast_hankel_inner(signal, l_windows, n_windows, lag), number=run_num) / run_num * 1000)


if __name__ == "__main__":
    test_matmul()
    test_parallelization()
