"""
This file contains only some internal code that has been used to compare, time and optimize some functions before
usage.
"""
import rocket_fft
import scipy.fft as spfft
import numpy as np
import concurrent.futures
import timeit
from utils.fastHankel import fast_hankel_matmul as fast_hankel_matmul2
from threadpoolctl import threadpool_limits
import numba


def get_fast_hankel_representation(time_series, end_index, length_windows, number_windows,
                                   lag=1) -> (np.ndarray, int, np.ndarray):
    signal = time_series[end_index-lag*(number_windows-1)-length_windows:end_index]
    fft_len = spfft.next_fast_len(signal.shape[0]+number_windows, True)
    hankel_rfft = spfft.rfft(signal, n=fft_len, axis=0)
    return hankel_rfft, fft_len, signal


def fast_hankel_vecmul(hankel_fft: np.ndarray, fft_shape: int, l_windows: int, m: int, vector: np.ndarray,
                       result_buffer: np.ndarray, index: int):

    # compute the fft of the vector
    fft_x = spfft.rfft(vector[:, index], n=fft_shape, workers=1)

    # multiply the ffts with each other to do the convolution in frequency domain and convert it back
    # and save it into the output buffer
    result_buffer[:, index] = spfft.irfft(hankel_fft*fft_x, n=fft_shape, workers=1)[(m-1):(m-1)+l_windows]


def fast_hankel_matmul_parallel(hankel_fft: np.ndarray, l_windows, fft_shape: int, other_matrix: np.ndarray,
                                threadpool: concurrent.futures.ThreadPoolExecutor):

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((l_windows, n))

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # make the multithreading using a process pool
    futures = [threadpool.submit(fast_hankel_vecmul,
                                 hankel_fft, fft_shape,
                                 l_windows,
                                 m,
                                 other_matrix,
                                 result_buffer, idx)
               for idx in range(other_matrix.shape[1])]
    concurrent.futures.wait(futures)
    return result_buffer


@numba.njit(parallel=True)
def fast_hankel_numba(hankel_fft: np.ndarray, l_windows: int, fft_shape: int, other_matrix: np.ndarray, lag: int = 1):

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros) to make up for the lag
    if lag > 1:
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((l_windows, n))

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in numba.prange(n):

        # compute the fft of the vector
        fft_x = spfft.rfft(other_matrix[:, index], n=fft_shape, workers=1)

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        fft_x *= hankel_fft
        result_buffer[:, index] = spfft.irfft(fft_x, n=fft_shape, workers=1)[(m - 1):(m - 1) + l_windows]
    return result_buffer


@numba.njit(parallel=True)
def fast_hankel_numba2(hankel_fft: np.ndarray, l_windows: int, fft_shape: int, other_matrix: np.ndarray, lag: int = 1):

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros) to make up for the lag
    if lag > 1:
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # compute the fft of the vector
    fft_x = spfft.rfft(other_matrix, n=fft_shape, axis=0, workers=6)

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((fft_x.shape[0], n), dtype="complex")

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in numba.prange(n):

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        result_buffer[:, index] = hankel_fft*fft_x[:, index]
    result_buffer = spfft.irfft(result_buffer, n=fft_shape, axis=0, workers=6)[(m - 1): (m - 1) + l_windows, :]
    return result_buffer


@numba.njit(parallel=True)
def fast_hankel_numba3(hankel_fft: np.ndarray, l_windows: int, fft_shape: int, other_matrix: np.ndarray, lag: int = 1):

    # get the shape of the other matrix
    m, n = other_matrix.shape
    k = hankel_fft.shape[0]

    # make fft of x (while padding x with zeros) to make up for the lag
    if lag > 1:
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((l_windows, n))

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in numba.prange(n):

        # compute the fft of the vector
        other_array = np.zeros(fft_shape)
        other_array[:m] = other_matrix[:, index]
        complex_buffer = np.zeros(fft_shape, dtype=np.complex128)
        rocket_fft.r2c(other_array, complex_buffer, axes=np.array([0], dtype=np.int64), forward=True, fct=1.0, nthreads=1)

        # multiply the ffts with each other to do the convolution in frequency domain
        complex_buffer[:k] *= hankel_fft

        # transform back
        result_array = np.empty(fft_shape)
        rocket_fft.c2r(complex_buffer, result_array, axes=np.array([0], dtype=np.int64), forward=False, fct=1/fft_shape, nthreads=1)
        result_buffer[:, index] = result_array[(m - 1): (m - 1) + l_windows]
    return result_buffer


def main():
    # define some window length
    limit_threads = 4
    l_windows = 10000
    n_windows = 10000
    lag = 1
    run_num = 100

    # create a time series of a certain length
    n = 300000
    # ts = np.random.uniform(size=(n,))*1000
    ts = np.linspace(0, n, n + 1)

    # create a matrix to multiply by
    cr = np.random.default_rng(5)
    multi = cr.uniform(size=(n_windows, 10))

    # get the final index of the time series
    end_idx = l_windows + lag * (n_windows - 1)

    # get both hankel representations
    hankel_rfft, fft_len, signal = get_fast_hankel_representation(ts, end_idx, l_windows, n_windows, lag)

    # run the jit compilation once
    numba.set_num_threads(limit_threads)
    fast_hankel_numba(hankel_rfft, l_windows, fft_len, multi)
    fast_hankel_numba(hankel_rfft, l_windows, fft_len, multi)
    fast_hankel_numba2(hankel_rfft, l_windows, fft_len, multi)
    fast_hankel_numba3(hankel_rfft, l_windows, fft_len, multi)

    # execute the function and measure the time
    with threadpool_limits(limits=limit_threads):
        with concurrent.futures.ThreadPoolExecutor(limit_threads) as pool:

            print("Best numba implementation")
            print(timeit.timeit(lambda: fast_hankel_numba(hankel_rfft, l_windows, fft_len, multi), number=run_num)/run_num*1000)

            print("Other numba implementations")
            print(timeit.timeit(lambda: fast_hankel_numba2(hankel_rfft, l_windows, fft_len, multi), number=run_num) / run_num * 1000)
            print(timeit.timeit(lambda: fast_hankel_numba3(hankel_rfft, l_windows, fft_len, multi), number=run_num) / run_num * 1000)

            print("Own Multithreading")
            print(timeit.timeit(lambda: fast_hankel_matmul_parallel(hankel_rfft, l_windows, fft_len, multi, threadpool=pool), number=run_num) / run_num * 1000)
            hankel_rfft2 = hankel_rfft[:, None]
            print(timeit.timeit(lambda: fast_hankel_matmul2(hankel_rfft2, l_windows, fft_len, multi, lag, workers=limit_threads), number=run_num)/run_num*1000)
            # print(fast_hankel_numba.parallel_diagnostics(level=4))
            # check whether the results are the same
            # a = fast_hankel_matmul_parallel(hankel_rfft, l_windows, fft_len, multi, threadpool=pool)
            a = fast_hankel_numba3(hankel_rfft, l_windows, fft_len, multi)
            b = fast_hankel_matmul2(hankel_rfft2, l_windows, fft_len, multi, lag, workers=limit_threads)
            print(np.allclose(a, b))


if __name__ == "__main__":
    main()
