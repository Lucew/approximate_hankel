import time
import timeit

import numpy as np
import pandas as pd
import scipy as sp
import numba as nb
try:
    import torch
    is_torch_available = True
except ModuleNotFoundError:
    is_torch_available = False

from threadpoolctl import threadpool_limits
import os
import concurrent.futures


@nb.njit()
def compile_hankel_naive(time_series: np.ndarray, end_index: int, window_size: int, rank: int,
                         lag: int = 1, safe: bool = False) -> np.ndarray:
    """
    This function constructs a hankel matrix from a 1D time series. Please make sure constructing the matrix with
    the given parameters (end index, window size, etc.) is possible, as this function does no checks due to
    performance reasons.

    :param time_series: 1D array with float values as the time series
    :param end_index: the index (point in time) where the time series starts
    :param window_size: the size of the windows cut from the time series
    :param rank: the amount of time series in the matrix
    :param lag: the lag between the time series of the different columns
    :param safe: whether we should check the construction (slow)
    :return: The hankel matrix with lag one
    """

    # make an empty matrix to place the values
    #
    # almost no faster way:
    # https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy
    hankel = np.empty((window_size, rank))
    if safe:
        hankel.fill(np.NAN)

    # go through the time series and make the hankel matrix
    for cx in range(rank):
        hankel[:, -cx-1] = time_series[(end_index-window_size-cx*lag):(end_index-cx*lag)]

    # check that we did not make a mistake
    if safe:
        assert np.all(~np.isnan(hankel)), "Something is off, there are still NAN in the numpy array."
    return hankel


@nb.njit(parallel=True)
def compile_hankel_parallel(time_series: np.ndarray, end_index: int, window_size: int, rank: int,
                            lag: int = 1, safe: bool = False) -> np.ndarray:
    """
    This function constructs a hankel matrix from a 1D time series. Please make sure constructing the matrix with
    the given parameters (end index, window size, etc.) is possible, as this function does no checks due to
    performance reasons.

    :param time_series: 1D array with float values as the time series
    :param end_index: the index (point in time) where the time series starts
    :param window_size: the size of the windows cut from the time series
    :param rank: the amount of time series in the matrix
    :param lag: the lag between the time series of the different columns
    :param safe: whether we should check the construction (slow)
    :return: The hankel matrix with lag one
    """

    # make an empty matrix to place the values
    #
    # almost no faster way:
    # https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy
    hankel = np.empty((window_size, rank))
    if safe:
        hankel.fill(np.NAN)

    # go through the off-diagonals of the hankel matrix in parallel
    sig_start = end_index-window_size-rank+1
    for dx in nb.prange(window_size+rank-1):

        # get the corresponding value from the time series
        val = time_series[sig_start+dx]

        # get starting indices of the diagonal
        rx = int(min(dx, window_size - 1))
        cx = int(max(0, dx - window_size + 1))

        # set all the values on the off diagonals of the hankel matrix
        for _ in range(min(rx, rank-cx-1)+1):
            hankel[rx, cx] = val
            rx -= 1
            cx += 1

    # check that we did not make a mistake
    if safe:
        assert np.all(~np.isnan(hankel)), "Something is off, there are still NAN in the numpy array."
    return hankel


def hankel_fft_from_matrix(hankel_matrix: np.ndarray):

    # reconstruct the signal from the first row and the last column
    signal = np.concatenate((hankel_matrix[0, :-1], hankel_matrix[:, -1]))

    # get the optimal fft length using scipy as explained in get_fast_hankel
    fft_len = sp.fft.next_fast_len(signal.shape[0] + hankel_matrix.shape[1], True)

    # Workers are not necessary as we expect a 1D time series
    hankel_rfft = sp.fft.rfft(signal, n=fft_len, axis=0).reshape(-1, 1)
    return hankel_rfft, fft_len, signal


def get_fast_hankel_representation(time_series, end_index, length_windows, number_windows,
                                   lag=1) -> (np.ndarray, int, np.ndarray):

    # get the last column of the hankel matrix. The reason for that is that we will use an algorithm for Toeplitz
    # matrices to speed up the multiplication and Hankel[:, ::-1] == Toeplitz.
    #
    # The algorithm requires the first Toeplitz column and therefore the last Hankel Column
    #
    # It also requires the inverse of the row columns ignoring the last element. For reference see:
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # additionally we only need the combined vector from both for our multiplication. In order to employ the
    # fastest DFT possible, the vector needs have a length with a clean power of two, which we create using
    # zero padding
    #
    # We can also use the signal itself, as multiplying a vector with the hankel matrix is effectively a convolution
    # over the signal. Therefore, we can use the fft convolution with way lower complexity or any other built in
    # convolution library!

    # get column and row, or we can also do it by using the signal itself!
    # last_column = time_series[end_index-length_windows:end_index]
    # row_without_last_element = time_series[end_index-lag*(number_windows-1)-length_windows:end_index-length_windows]
    signal = time_series[end_index-lag*(number_windows-1)-length_windows:end_index]

    # get the length of the matrices
    # col_length = last_column.shape[0]
    # row_length = row_without_last_element.shape[0]
    # combined_length = col_length + row_length

    # compute the padded vector length for an optimal fft length. Here we can use the built-in scipy function that
    # computes the perfect fft length optimized for their implementation, so it is even faster than with powers
    # of two!
    # fft_len = 1 << int(np.ceil(np.log2(combined_length)))
    # fft_len = sp.fft.next_fast_len(combined_length, True)
    fft_len = sp.fft.next_fast_len(signal.shape[0]+number_windows, True)

    # compute the fft over the padded hankel matrix
    # if we would pad in the middle like this: we would introduce no linear phase to the fft
    #
    # padded_toeplitz = np.concatenate((last_column, np.zeros((fft_len - combined_length)), row_without_last_element))
    #
    # but we want to use the built-in padding functionality of the fft in scipy, so we pad at the end like can be seen
    # here.
    # This introduces a linear phase and a shift to the fft, which we need to account for in the reverse
    # functions.
    #
    # More details see:
    # https://dsp.stackexchange.com/questions/82273/why-to-pad-zeros-at-the-middle-of-sequence-instead-at-the-end-of-the-sequence
    # https://dsp.stackexchange.com/questions/83461/phase-of-an-fft-after-zeropadding
    #
    # Workers are not necessary as we expect a 1D time series
    hankel_rfft = sp.fft.rfft(signal, n=fft_len, axis=0).reshape(-1, 1)
    return hankel_rfft, fft_len, signal


def fast_hankel_matmul(hankel_fft: np.ndarray, l_windows, fft_shape: int, other_matrix: np.ndarray, lag, workers=None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if workers is None:
        workers = os.cpu_count()//2

    # save the shape of the matrix
    ndim = other_matrix.ndim
    if ndim == 1:
        other_matrix = other_matrix[:, None]
    elif ndim == 2:
        pass
    else:
        raise ValueError("Other matrix has to have an ndim of one or two.")
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros)
    if lag > 1:
        out = np.zeros((lag*m-lag+1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out
    fft_x = sp.fft.rfft(np.flipud(other_matrix), n=fft_shape, workers=workers, axis=0)

    # compute the inverse fft and take into account the offset due to circular convolution and zero padding as explained
    # in
    # https://dsp.stackexchange.com/questions/82273/why-to-pad-zeros-at-the-middle-of-sequence-instead-at-the-end-of-the-sequence
    # and
    # https://dsp.stackexchange.com/questions/83461/phase-of-an-fft-after-zeropadding
    mat_times_x = sp.fft.irfft(hankel_fft*fft_x, axis=0, n=fft_shape, workers=workers)[(m-1)*lag:(m-1)*lag+l_windows, :]

    return_shape = (l_windows,) if ndim == 1 else (l_windows, n)
    return mat_times_x.reshape(*return_shape)


def fast_hankel_left_matmul(hankel_fft: np.ndarray, n_windows, fft_shape: int, other_matrix: np.ndarray, lag: int,
                            workers: int = None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if workers is None:
        workers = os.cpu_count()//2

    # transpose the other matrix
    other_matrix = other_matrix.T

    # save the shape of the matrix
    ndim = other_matrix.ndim
    if ndim == 1:
        other_matrix = other_matrix[:, None]
    elif ndim == 2:
        pass
    else:
        raise ValueError("Other matrix has to have an ndim of one or two.")
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros)
    fft_x = sp.fft.rfft(np.flipud(other_matrix), n=fft_shape, workers=workers, axis=0)

    # compute the inverse fft
    mat_times_x = sp.fft.irfft(hankel_fft * fft_x, axis=0, n=fft_shape, workers=workers)[(m-1):(m-1)+n_windows*lag:lag]

    return_shape = (n_windows,) if ndim == 1 else (n_windows, n)
    return mat_times_x.reshape(*return_shape).T


def fast_hankel_vecmul(hankel_fft: np.ndarray, fft_shape: int, l_windows: int, m: int, vector: np.ndarray, lag: int,
                       result_buffer: np.ndarray, index: int):

    # compute the fft of the vector
    fft_x = sp.fft.rfft(vector[:, index], n=fft_shape, workers=1)

    # multiply the ffts with each other to do the convolution in frequency domain and convert it back
    # and save it into the output buffer.
    # also take into account the phase shift as explained in the non-parallel version of this function
    result_buffer[:, index] = sp.fft.irfft(hankel_fft*fft_x, n=fft_shape, workers=1)[(m-1)*lag:(m-1)*lag+l_windows]


def fast_parallel_hankel_matmul(hankel_fft: np.ndarray, l_windows, fft_shape: int, other_matrix: np.ndarray, lag: int,
                                threadpool: concurrent.futures.ThreadPoolExecutor):

    # check whether we need to make the hankel fft 1D
    hankel_ndim = hankel_fft.ndim
    if hankel_ndim == 1:
        pass
    elif hankel_ndim == 2 and hankel_fft.shape[1] == 1:
        hankel_fft = hankel_fft[:, 0]
    else:
        raise ValueError("Hankel representation is off.")

    # make fft of x (while padding x with zeros) to make up for the lag
    m, n = other_matrix.shape
    if lag > 1:
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((l_windows, n))

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # make the multithreading using a process pool
    futures = [threadpool.submit(fast_hankel_vecmul, hankel_fft, fft_shape, l_windows, m, other_matrix, lag,
                                 result_buffer, idx)
               for idx in range(other_matrix.shape[1])]
    concurrent.futures.wait(futures)
    return result_buffer


def fast_hankel_left_vecmul(hankel_fft: np.ndarray, fft_shape: int, n_windows: int, m: int, vector: np.ndarray,
                            lag: int, result_buffer: np.ndarray, index: int):

    # compute the fft of the vector
    fft_x = sp.fft.rfft(vector[:, index], n=fft_shape, workers=1)

    # multiply the ffts with each other to do the convolution in frequency domain and convert it back
    # and save it into the output buffer.
    # also take into account the phase shift as explained in the non-parallel version of this function
    result_buffer[:, index] = sp.fft.irfft(hankel_fft*fft_x, n=fft_shape, workers=1)[(m-1):(m-1)+n_windows*lag:lag]


def fast_parallel_hankel_left_matmul(hankel_fft: np.ndarray, n_windows, fft_shape: int, other_matrix: np.ndarray,
                                     lag: int, threadpool: concurrent.futures.ThreadPoolExecutor):

    # check whether we need to make the hankel fft 1D
    hankel_ndim = hankel_fft.ndim
    if hankel_ndim == 1:
        pass
    elif hankel_ndim == 2 and hankel_fft.shape[1] == 1:
        hankel_fft = hankel_fft[:, 0]
    else:
        raise ValueError("Hankel representation is off.")

    # transpose the other matrix
    other_matrix = other_matrix.T

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((n_windows, n))

    # make the multithreading using a process pool
    futures = [threadpool.submit(fast_hankel_left_vecmul, hankel_fft, fft_shape, n_windows, m, other_matrix, lag,
                                 result_buffer, idx)
               for idx in range(other_matrix.shape[1])]
    concurrent.futures.wait(futures)
    return result_buffer.T


@nb.njit(parallel=True)
def fast_numba_hankel_matmul(hankel_fft: np.ndarray, l_windows: int, fft_shape: int, other_matrix: np.ndarray,
                             lag: int):

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros) to make up for the lag
    if lag > 1:
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
    else:
        out = other_matrix

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((l_windows, n))

    # flip the other matrix
    out = np.flipud(out)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in nb.prange(n):

        # compute the fft of the vector
        fft_x = sp.fft.rfft(out[:, index], n=fft_shape)

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        result_buffer[:, index] = sp.fft.irfft(hankel_fft*fft_x, n=fft_shape)[(m-1)*lag:(m-1)*lag+l_windows]
    return result_buffer


@nb.njit(parallel=True)
def fast_numba_hankel_left_matmul(hankel_fft: np.ndarray, n_windows: int, fft_shape: int, other_matrix: np.ndarray,
                                  lag: int):

    # transpose the other matrix
    other_matrix = other_matrix.T

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((n_windows, n))

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in nb.prange(n):

        # compute the fft of the vector
        fft_x = sp.fft.rfft(other_matrix[:, index], n=fft_shape)

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        result_buffer[:, index] = sp.fft.irfft(hankel_fft*fft_x, n=fft_shape)[(m-1):(m-1)+n_windows*lag:lag]
    return result_buffer.T


def fast_fftconv_hankel_matmul(hankel_signal: np.ndarray, other_matrix: np.ndarray, lag: int, workers: int = None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if not workers:
        workers = os.cpu_count()//2

    if lag > 1:
        m, n = other_matrix.shape
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out
    result = np.stack([sp.signal.fftconvolve(hankel_signal, other_matrix[::-1, col], mode="valid") for col in range(other_matrix.shape[1])]).T
    return result


def fast_fftconv_hankel_left_matmul(hankel_signal: np.ndarray, other_matrix: np.ndarray, lag: int, workers: int =None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if not workers:
        workers = 1
    result = np.stack([sp.signal.fftconvolve(hankel_signal, other_matrix[row, ::-1], mode="valid")[::lag] for row in range(other_matrix.shape[0])])
    return result


def fast_convolve_hankel_matmul(hankel_signal: np.ndarray, other_matrix: np.ndarray, lag: int):
    if lag > 1:
        m, n = other_matrix.shape
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out
    result = np.stack([np.convolve(hankel_signal, other_matrix[::-1, col], mode="valid") for col in range(other_matrix.shape[1])]).T
    return result


# normal type annotations do not work with conditional imports!
def fast_torch_hankel_matmul(hankel_repr: "torch.Tensor", other_matrix: "torch.Tensor", lag: int):
    if not is_torch_available:
        raise ImportError("Torch not available")
    with torch.no_grad():
        if lag > 1:
            m, b, n = other_matrix.shape
            out = torch.zeros((m, b, lag * n - lag + 1), dtype=other_matrix.dtype)
            out[:, :, ::lag] = other_matrix
            other_matrix = out
        result = torch.nn.functional.conv1d(hankel_repr, other_matrix)
        result = result[0, :, :].transpose(0, 1)
        return result.detach().cpu().numpy()


@nb.njit()
def fast_hankel_inner(hankel_repr: np.ndarray, window_length, window_number, lag):

    # create a matrix that contains the hankel matrix
    inner_prod = np.zeros((window_number, window_number))
    for cx in range(window_number):
        tmp = np.convolve(hankel_repr[cx*lag:], hankel_repr[cx*lag:cx*lag+window_length][::-1], "valid")[::lag]
        inner_prod[cx, cx:] = tmp
        inner_prod[cx:, cx] = tmp
    return inner_prod


def normal_hankel_matmul(hankel, other):
    return hankel @ other


def normal_hankel_left_matmul(hankel, other):
    return other @ hankel


def normal_hankel_inner(hankel):
    return hankel.T @ hankel


def run_measurements(thread_counts: list[int],
                     window_lengths: list[int],
                     window_numbers: list[int],
                     signal_scaling: list[int],
                     other_matrix_dimensions: list[int],
                     other_matrix_scaling: list[int],
                     lags: list[int],
                     runs: int = 1000):

    # get the performance counter
    perc = time.perf_counter_ns

    # try to import tqdm
    from tqdm import tqdm

    # make a dict to save the values
    values = {"Naive Execution Time (over all Runs) Left Product Other@Hankel": [],
              "Naive Execution Time (over all Runs) Right Product Hankel@Other": [],
              "FFT Execution Time (over all Runs) Left Product Other@Hankel": [],
              "FFT Execution Time (over all Runs) Right Product Hankel@Other": [],
              "Parallel FFT Execution Time (over all Runs) Left Product Other@Hankel": [],
              "Parallel FFT Execution Time (over all Runs) Right Product Hankel@Other": [],
              "Window Length": [],
              "Window Number": [],
              "Signal Scaling": [],
              "Lag": [],
              "Other Matrix Dim.": [],
              "Other Matrix Scaling": [],
              "Thread Count": [],
              "Runs": [],
              "Median Diff. Left Product Other@Hankel": [],
              "Median Diff. Right Product Hankel@Other": [],
              "Max. Diff. Left Product Other@Hankel": [],
              "Max. Diff. Right Product Hankel@Other": [],
              "Std. Diff. Left Product Other@Hankel": [],
              "Std. Diff. Right Product Hankel@Other": [],
              "Machine Precision": []}

    # create a generator to make the tuples
    tuple_generator = [(wl, wn, sc, om, omsc, lag, tc)
                       for wl in window_lengths
                       for wn in window_numbers
                       for lag in lags
                       for sc in signal_scaling
                       for om in other_matrix_dimensions
                       for omsc in other_matrix_scaling
                       for tc in thread_counts]

    for (wl, wn, sc, om, omsc, lag, tc) in tqdm(tuple_generator, "Compute all the tuples"):

        # limit the threads used by numba
        nb.set_num_threads(tc)

        # this limits the threads for numpy (at least for our version)
        with threadpool_limits(limits=tc):

            # save the parameters
            values["Window Length"].append(wl)
            values["Window Number"].append(wn)
            values["Signal Scaling"].append(sc)
            values["Lag"].append(lag)
            values["Other Matrix Dim."].append(om)
            values["Other Matrix Scaling"].append(omsc)
            values["Thread Count"].append(tc)
            values["Runs"].append(runs)

            # compute the necessary length for the signal
            end_idx = wl + lag * (wn - 1)

            # make the random signal with the given scale
            signal = np.random.uniform(size=(end_idx+10,))*sc

            # create the matrix representation
            hankel_rfft, fft_len, signal = get_fast_hankel_representation(signal, end_idx, wl, wn, lag)
            hankel = compile_hankel_parallel(signal, end_idx, wl, wn, lag)

            # measure the time to get the matrix representation
            repr_time = timeit.timeit(lambda: get_fast_hankel_representation(signal, end_idx, wl, wn, lag),
                                      number=runs,
                                      timer=perc)

            # create the other two matrices we want to multiply with
            other_right = np.random.uniform(size=(wn, om))*omsc
            other_left = np.random.uniform(size=(om, wl))*omsc

            # measure the multiplication time for naive implementation
            naive_time_right = timeit.timeit(lambda: normal_hankel_matmul(hankel, other_right),
                                             number=runs,
                                             timer=perc)
            values["Naive Execution Time (over all Runs) Right Product Hankel@Other"].append(naive_time_right)
            naive_time_left = timeit.timeit(lambda: normal_hankel_left_matmul(hankel, other_left),
                                            number=runs,
                                            timer=perc)
            values["Naive Execution Time (over all Runs) Left Product Other@Hankel"].append(naive_time_left)

            # measure the multiplication time for fft implementation
            fft_time_right = timeit.timeit(lambda: fast_hankel_matmul(hankel_rfft, wl, fft_len,
                                                                      other_right, lag, workers=tc),
                                           number=runs,
                                           timer=perc)
            values["FFT Execution Time (over all Runs) Right Product Hankel@Other"].append(fft_time_right+repr_time)
            fft_time_left = timeit.timeit(lambda: fast_hankel_left_matmul(hankel_rfft, wn, fft_len,
                                                                          other_left, lag, workers=tc),
                                          number=runs,
                                          timer=perc)
            values["FFT Execution Time (over all Runs) Left Product Other@Hankel"].append(fft_time_left+repr_time)

            # measure the multiplication time for parallel fft implementation (before compute once to trigger the
            # jit compilation of the functions
            fast_numba_hankel_matmul(hankel_rfft[:, 0], wl, fft_len, other_right, lag)
            fast_numba_hankel_left_matmul(hankel_rfft[:, 0], wn, fft_len, other_left, lag)
            fft_time_right = timeit.timeit(lambda: fast_numba_hankel_matmul(hankel_rfft[:, 0], wl, fft_len, other_right,
                                                                            lag),
                                           number=runs,
                                           timer=perc)
            values["Parallel FFT Execution Time (over all Runs) Right Product Hankel@Other"].append(fft_time_right +
                                                                                                    repr_time)
            fft_time_left = timeit.timeit(lambda: fast_numba_hankel_left_matmul(hankel_rfft[:, 0], wn, fft_len,
                                                                                other_left, lag),
                                          number=runs,
                                          timer=perc)
            values["Parallel FFT Execution Time (over all Runs) Left Product Other@Hankel"].append(fft_time_left +
                                                                                                   repr_time)

            # compute the products for error estimation
            naive_right_product = normal_hankel_matmul(hankel, other_right)
            naive_left_product = normal_hankel_left_matmul(hankel, other_left)
            fft_right_product = fast_hankel_matmul(hankel_rfft, wl, fft_len, other_right, lag, workers=tc)
            fft_left_product = fast_hankel_left_matmul(hankel_rfft, wn, fft_len, other_left, lag, workers=tc)

            # compute the errors for right product
            right_error = evaluate_closeness(naive_right_product, fft_right_product, "")
            values["Median Diff. Right Product Hankel@Other"].append(right_error["Median Diff."])
            values["Max. Diff. Right Product Hankel@Other"].append(right_error["Max. Diff."])
            values["Std. Diff. Right Product Hankel@Other"].append(right_error["Std. Diff."])
            values["Machine Precision"].append(right_error["Machine Precision"])

            # compute the errors for left product
            left_error = evaluate_closeness(naive_left_product, fft_left_product, "")
            values["Median Diff. Left Product Other@Hankel"].append(left_error["Median Diff."])
            values["Max. Diff. Left Product Other@Hankel"].append(left_error["Max. Diff."])
            values["Std. Diff. Left Product Other@Hankel"].append(left_error["Std. Diff."])
            assert right_error["Machine Precision"] == left_error["Machine Precision"], "Something odd with eps."

    # create a dataframe and save the results
    df = pd.DataFrame(values)
    df.to_csv("Results_HankelMult_OwnParallel.csv")


def trigger_numba_matmul_jit():

    # make some definitions
    limit_threads = 2
    l_windows = 1000
    n_windows = 1000

    # get the final index of the time series
    end_idx = l_windows + (n_windows - 1)

    # create a time series of a certain length
    n = end_idx+10
    ts = np.linspace(0, n, n + 1)

    # create a matrix to multiply by
    k = 15
    multi = np.random.uniform(size=(n_windows, k))

    # get hankel representations
    hankel_rfft, fft_len, signal = get_fast_hankel_representation(ts, end_idx, l_windows, n_windows, 1)

    # trigger the jit compilations
    nb.set_num_threads(limit_threads)
    fast_numba_hankel_left_matmul(hankel_rfft[:, 0], n_windows, fft_len, multi, 1)
    fast_numba_hankel_matmul(hankel_rfft[:, 0], l_windows, fft_len, multi, 1)


if __name__ == "__main__":
    run_measurements(thread_counts=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     window_lengths=list(np.geomspace(10, 20_000, num=30, dtype=int))[::-1],
                     window_numbers=list(np.geomspace(10, 20_000, num=30, dtype=int))[::-1],
                     signal_scaling=[1],
                     other_matrix_dimensions=[5, 10, 20, 50],
                     other_matrix_scaling=[1],
                     lags=[1],
                     runs=50)
