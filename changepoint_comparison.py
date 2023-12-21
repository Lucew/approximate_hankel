import collections
import numpy as np
import scipy.linalg

from fastHankel import (fast_hankel_matmul, fast_hankel_left_matmul, get_fast_hankel_representation,
                        hankel_fft_from_matrix)
import scipy as sp
import numba as nb
import h5py
import time
import pandas as pd
from tqdm import tqdm


"""
Academic Sources:

[1]
Idé, Tsuyoshi, and Koji Tsuda.
"Change-point detection using krylov subspace learning."
Proceedings of the 2007 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2007.

[2]
Halko, N. and Martinsson, P. G. and Tropp, J. A.
Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions
SIAM Review, Volume 53, Number 2, pP 217-288, 2011.
"""


########################################################################################################################
# ------------------------------------ POWER ITERATIONS ---------------------------------------------------------------#
########################################################################################################################
def power_method(a_matrix: np.ndarray, x_vector: np.ndarray, n_iterations: int) -> (float, np.ndarray):
    """
    This function searches the largest (dominant) eigenvalue and corresponding eigenvector by repeated multiplication
    of the matrix A with an initial vector. It assumes a dominant eigenvalue bigger than the second one, otherwise
    it won't converge.

    For proof and explanation look at:
    https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html
    """

    # go through the iterations and continue to scale the returned vector, so we do not reach extreme values
    # during the iteration we scale the vector by its maximum as we can than easily extract the eigenvalue
    c_matrix = a_matrix.T @ a_matrix
    for _ in range(n_iterations):

        # multiplication with a_matrix.T @ a_matrix as can be seen in explanation of
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        x_vector = c_matrix @ x_vector

        # scale the vector so we keep the values in bound
        x_vector = x_vector / np.max(x_vector)

    # get the normed eigenvector
    x_vector = x_vector / np.linalg.norm(x_vector)

    # get the corresponding eigenvalue
    mat_vec = a_matrix @ x_vector
    eigenvalue = np.linalg.norm(mat_vec)
    return eigenvalue, mat_vec/eigenvalue


########################################################################################################################
# -------------------------------- HANKEL REPRESENTATIONS ------------------------------------------------------------ #
########################################################################################################################
@nb.njit()
def compile_hankel_naive(time_series: np.ndarray, end_index: int, window_size: int, rank: int,
                         lag: int = 1) -> np.ndarray:
    """
    This function constructs a hankel matrix from a 1D time series. Please make sure constructing the matrix with
    the given parameters (end index, window size, etc.) is possible, as this function does no checks due to
    performance reasons.

    :param time_series: 1D array with float values as the time series
    :param end_index: the index (point in time) where the time series starts
    :param window_size: the size of the windows cut from the time series
    :param rank: the amount of time series in the matrix
    :param lag: the lag between the time series of the different columns
    :return: The hankel matrix with lag one
    """

    # make an empty matrix to place the values
    #
    # almost no faster way:
    # https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy
    hankel = np.empty((window_size, rank))

    # go through the time series and make the hankel matrix
    for cx in range(rank):
        hankel[:, -cx-1] = time_series[(end_index-window_size-cx*lag):(end_index-cx*lag)]
    return hankel


def compile_hankel_fft(time_series: np.ndarray, end_index: int, window_size: int, rank: int,
                       lag: int = 1) -> (np.ndarray, int, np.ndarray):
    return get_fast_hankel_representation(time_series, end_index, window_size, rank, lag=lag)


########################################################################################################################
# --------------------------- Randomized SVD SST---------------------------------------------------------------------- #
########################################################################################################################

def randomized_hankel_svd_fft(hankel_fft: np.ndarray, fft_length: int, k: int, subspace_iteration_q: int,
                              oversampling_p: int, length_windows: int, number_windows: int,
                              random_state: np.random.RandomState):
    """
    Function for the randomized singular vector decomposition using [1].
    Implementation modified from: https://pypi.org/project/fbpca/
    """

    # get the parameter l from the paper
    sample_length_l = k+oversampling_p
    assert 1.25*sample_length_l < min(length_windows, number_windows)

    # Apply A to a random matrix, obtaining Q.
    random_matrix_omega = random_state.uniform(low=-1, high=1, size=(number_windows, sample_length_l))
    projection_matrix_q = fast_hankel_matmul(hankel_fft, length_windows, fft_length, random_matrix_omega, lag=1)

    # Form a matrix Q whose columns constitute a well-conditioned basis for the columns of the earlier Q.
    if subspace_iteration_q == 0:
        (projection_matrix_q, _) = sp.linalg.qr(projection_matrix_q, mode='economic')
    if subspace_iteration_q > 0:
        (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(subspace_iteration_q):

        # Q = fast_hankel_matmul(Q.T, A).conj().T
        projection_matrix_q = fast_hankel_left_matmul(hankel_fft, number_windows, fft_length,
                                                      projection_matrix_q.T, lag=1).T

        (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)

        # Q = mult(A, Q)
        projection_matrix_q = fast_hankel_matmul(hankel_fft, length_windows, fft_length, projection_matrix_q,
                                                 lag=1)

        if it + 1 < subspace_iteration_q:
            (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)
        else:
            (projection_matrix_q, _) = sp.linalg.qr(projection_matrix_q, mode='economic')

    # SVD Q'*A to obtain approximations to the singular values and right singular vectors of A; adjust the left singular
    # vectors of Q'*A to approximate the left singular vectors of A.
    lower_space_hankel = fast_hankel_left_matmul(hankel_fft, number_windows, fft_length, projection_matrix_q.T, lag=1)
    (R, s, Va) = sp.linalg.svd(lower_space_hankel, full_matrices=False)
    U = projection_matrix_q.dot(R)

    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    return U[:, :k], s[:k], Va[:k, :]


def randomized_hankel_svd_naive(hankel_matrix: np.ndarray, k: int, subspace_iteration_q: int,
                                oversampling_p: int, length_windows: int, number_windows: int,
                                random_state: np.random.RandomState):
    """
    Function for the randomized singular vector decomposition using [1].
    Implementation modified from: https://pypi.org/project/fbpca/
    """

    # get the parameter l from the paper
    sample_length_l = k+oversampling_p
    assert 1.25*sample_length_l < min(length_windows, number_windows)

    # Apply A to a random matrix, obtaining Q.
    random_matrix_omega = random_state.uniform(low=-1, high=1, size=(number_windows, sample_length_l))
    projection_matrix_q = hankel_matrix@random_matrix_omega

    # Form a matrix Q whose columns constitute a well-conditioned basis for the columns of the earlier Q.
    if subspace_iteration_q == 0:
        (projection_matrix_q, _) = sp.linalg.qr(projection_matrix_q, mode='economic')
    if subspace_iteration_q > 0:
        (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(subspace_iteration_q):

        # QA
        projection_matrix_q = (projection_matrix_q.T @ hankel_matrix).T

        (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)

        # AAQ
        projection_matrix_q = hankel_matrix @ projection_matrix_q

        if it + 1 < subspace_iteration_q:
            (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)
        else:
            (projection_matrix_q, _) = sp.linalg.qr(projection_matrix_q, mode='economic')

    # SVD Q'*A to obtain approximations to the singular values and right singular vectors of A; adjust the left singular
    # vectors of Q'*A to approximate the left singular vectors of A.
    lower_space_hankel = projection_matrix_q.T @ hankel_matrix
    (R, s, Va) = sp.linalg.svd(lower_space_hankel, full_matrices=False)
    U = projection_matrix_q.dot(R)

    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    return U[:, :k], s[:k], Va[:k, :]


def rsvd_score_naive(hankel_matrix: np.ndarray, eigvec_future: np.ndarray, k: int, subspace_iteration_q: int,
                     oversampling_p: int, length_windows: int, number_windows: int,
                     random_state: np.random.RandomState):

    # get the eigenvectors and eigenvalues
    left_eigenvectors, _, _ = randomized_hankel_svd_naive(hankel_matrix, k, subspace_iteration_q, oversampling_p,
                                                          length_windows, number_windows, random_state)

    # make the multiplication
    scores = left_eigenvectors.T @ eigvec_future
    return 1 - (scores.T @ scores).sum()


def rsvd_score_fft(hankel_fft: np.ndarray, eigvec_future: np.ndarray, fft_length: int, k: int,
                   subspace_iteration_q: int, oversampling_p: int, length_windows: int, number_windows: int,
                   random_state: np.random.RandomState):
    # get the eigenvectors and eigenvalues
    left_eigenvectors, _, _ = randomized_hankel_svd_fft(hankel_fft, fft_length, k, subspace_iteration_q, oversampling_p,
                                                        length_windows, number_windows, random_state)

    # make the multiplication
    scores = left_eigenvectors.T @ eigvec_future
    return 1 - (scores.T @ scores).sum()


########################################################################################################################
# -------------------------------- IKA SST --------------------------------------------------------------------------- #
########################################################################################################################


def implicit_krylov_approximation_naive(hankel_past: np.ndarray, eigvec_future: np.ndarray, rank: int,
                                        lanczos_rank: int) -> (float, np.ndarray):
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in
    [1].
    """

    # compute the tridiagonal matrix from the past hankel matrix
    alphas, betas = lanczos_naive(hankel_past, eigvec_future, lanczos_rank)

    # compute the singular value decomposition of the tridiagonal matrix (only the biggest)
    _, eigvecs = tridiagonal_eigenvalues(alphas, betas, rank)

    # compute the similarity score as defined in the ika sst paper and also return our u for the
    # feedback loop in figure 3 of the paper
    return 1 - (eigvecs[0, :] * eigvecs[0, :]).sum()


def implicit_krylov_approximation_fft(hankel_fft: np.ndarray, fft_length: int, length_windows: int,
                                      eigvec_future: np.ndarray, rank: int, lanczos_rank: int) -> (float, np.ndarray):
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in
    [1]. It uses the fft for the matrix multiplication.
    """

    # compute the tridiagonal matrix from the past hankel matrix
    alphas, betas = lanczos_fft(hankel_fft, fft_length, length_windows, eigvec_future, lanczos_rank)

    # compute the singular value decomposition of the tridiagonal matrix (only the biggest)
    _, eigvecs = tridiagonal_eigenvalues(alphas, betas, rank)

    # compute the similarity score as defined in the ika sst paper and also return our u for the
    # feedback loop in figure 3 of the paper
    return 1 - (eigvecs[0, :] * eigvecs[0, :]).sum()


def lanczos_naive(a_matrix: np.ndarray, r_0: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """
    This function computes the tri-diagonalization matrix from the square matrix C which is the result of the lanczos
    algorithm.

    The algorithm has been described and proven in [1].
    """

    # save the initial vector
    r_i = r_0
    q_i = np.zeros_like(r_i)

    # initialization of the diagonal elements
    alphas = np.zeros(shape=(k + 1,), dtype=np.float64)
    betas = np.ones(shape=(k + 1,), dtype=np.float64)

    # Subroutine 1 of the paper
    for j in range(k):
        # compute r_(j+1)
        new_q = r_i / betas[j]

        # compute inner product of new_q and hankel matrix
        inner_prod = a_matrix @ new_q

        # compute the new alpha
        alphas[j + 1] = new_q.T @ inner_prod

        # compute the new r
        r_i = inner_prod - alphas[j + 1] * new_q - betas[j] * q_i

        # compute the next beta
        betas[j + 1] = np.linalg.norm(r_i)

        # update the previous q
        q_i = new_q

    return alphas[1:], betas[1:-1]


def lanczos_fft(hankel_fft: np.ndarray, fft_length: int, length_windows: int, r_0: np.ndarray,
                k: int) -> (np.ndarray, np.ndarray):
    """
    This function computes the tri-diagonalization matrix from the square matrix C which is the result of the lanczos
    algorithm. It uses the fft for the matrix multiplication.

    The algorithm has been described and proven in [1]
    """

    # save the initial vector
    r_i = r_0
    q_i = np.zeros_like(r_i)

    # initialization of the diagonal elements
    alphas = np.zeros(shape=(k + 1,), dtype=np.float64)
    betas = np.ones(shape=(k + 1,), dtype=np.float64)

    # Subroutine 1 of the paper
    for j in range(k):
        # compute r_(j+1)
        new_q = r_i / betas[j]

        # compute inner product of new_q and hankel matrix
        inner_prod = fast_hankel_matmul(hankel_fft, length_windows, fft_length, new_q, lag=1)

        # compute the new alpha
        alphas[j + 1] = new_q.T @ inner_prod

        # compute the new r
        r_i = inner_prod - alphas[j + 1] * new_q - betas[j] * q_i

        # compute the next beta
        betas[j + 1] = np.linalg.norm(r_i)

        # update the previous q
        q_i = new_q

    return alphas[1:], betas[1:-1]


def tridiagonal_eigenvalues(alphas: np.ndarray, betas: np.ndarray, amount=-1):
    """
    This function uses a fast approach for symmetric tridiagonal matrices to calculate the [amount] highest eigenvalues
    and corresponding eigenvectors.
    """

    # check whether we need to use default parameters
    if amount < 0:
        amount = alphas.shape[0]

    # assertions about shape and dimensions as well as amount of eigenvectors
    assert 0 < amount <= alphas.shape[0], 'We can only calculate one to size of matrix eigenvalues.'
    assert alphas.ndim == 1, 'The alphas need to be vectors.'
    assert betas.ndim == 1, 'The betas need to be vectors.'
    assert alphas.shape[0] - 1 == betas.shape[0], 'Alpha size needs to be exactly one bigger than beta size.'

    # compute the decomposition
    eigenvalues, eigenvectors = sp.linalg.eigh_tridiagonal(d=alphas, e=betas, select='i',
                                                           select_range=(alphas.shape[0]-amount, alphas.shape[0]-1))

    # return them to be in sinking order
    return eigenvalues[::-1], eigenvectors[:, ::-1]


########################################################################################################################
# --------------------------------- Exact SST ------------------------------------------------------------------------ #
########################################################################################################################


def exact_svd(hankel_matrix: np.ndarray, eigvec_future: np.ndarray, rank: int) -> np.ndarray:
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in
    [1]. It uses the fft for the matrix multiplication.
    """

    # compute the singular value decomposition of the hankel matrix
    left_eigenvectors, _, _ = np.linalg.svd(hankel_matrix, full_matrices=False)
    left_eigenvectors = left_eigenvectors[:, :rank]

    # compute the similarity score as defined in the ika sst paper and also return our u for the
    # feedback loop in figure 3 of the paper
    scores = left_eigenvectors.T @ eigvec_future
    return 1 - (scores.T @ scores).sum()


########################################################################################################################
# ------------------------------- TRANSFORM FUNCTION ----------------------------------------------------------------- #
########################################################################################################################


def transform(time_series: np.ndarray, window_length: int, window_number: int, lag: int, end_idx: int,
              key: str, random_state: np.random.RandomState, power_iterations: int = 20) -> float:

    # check that the time series fits and we did not make an error
    assert len(time_series) >= end_idx, f"Time series is too short ({time_series.shape}) for start: {end_idx}."

    # create random vector for power iterations with the future hankel matrix as described in [1]
    x0 = random_state.rand(window_number)
    x0 /= np.linalg.norm(x0)

    # add small noise to the data so the ika sst does not break
    time_series += random_state.normal(scale=1e-4)

    if key == "fft rsvd":

        # compile the future hankel matrix (H2)
        hankel_future, fft_length, _ = compile_hankel_fft(time_series, end_idx, window_length, window_number)

        # get the first singular matrix vector of future hankel matrix
        x0, _, _ = randomized_hankel_svd_fft(hankel_future, fft_length, k=1, subspace_iteration_q=2,
                                             oversampling_p=2, length_windows=window_length,
                                             number_windows=window_number, random_state=random_state)

        # compile the past hankel matrix (H1)
        hankel_past, fft_length, _ = compile_hankel_fft(time_series, end_idx - lag, window_length, window_number)

        # compute the scoring using the ika naive implementation
        score = rsvd_score_fft(hankel_past, x0, fft_length, k=5, subspace_iteration_q=2, oversampling_p=2,
                               length_windows=window_length, number_windows=window_number, random_state=random_state)

    elif key == "naive rsvd":

        # compile the future hankel matrix (H2)
        hankel_future = compile_hankel_naive(time_series, end_idx, window_length, window_number)

        # get the first singular vector of the future hankel matrix
        x0, _, _ = randomized_hankel_svd_naive(hankel_future, k=1, subspace_iteration_q=2, oversampling_p=2,
                                               length_windows=window_length, number_windows=window_number,
                                               random_state=random_state)

        # compile the past hankel matrix (H1)
        hankel_past = compile_hankel_naive(time_series, end_idx - lag, window_length, window_number)

        # compute the scoring using the ika naive implementation
        score = rsvd_score_naive(hankel_past, x0, k=5, subspace_iteration_q=2, oversampling_p=2,
                                 length_windows=window_length, number_windows=window_number, random_state=random_state)

    elif key == "fft ika":

        # compile the future hankel matrix (H2)
        hankel_future = compile_hankel_naive(time_series, end_idx, window_length, window_number)

        # make the power iterations
        _, x0 = power_method(hankel_future, x0, power_iterations)

        # compile the past hankel matrix (H1) and compute outer product C as in the paper
        hankel_past = compile_hankel_naive(time_series, end_idx - lag, window_length, window_number)
        hankel_past = hankel_past @ hankel_past.T

        # compute the scoring using the ika naive implementation
        score = implicit_krylov_approximation_naive(hankel_past, x0, 5, 9)

    elif key == "naive ika":

        # compile the future hankel matrix (H2)
        hankel_future = compile_hankel_naive(time_series, end_idx, window_length, window_number)

        # make the power iterations
        _, x0 = power_method(hankel_future, x0, power_iterations)

        # compile the past hankel matrix (H1) and compute outer product C as in the paper
        hankel_past = compile_hankel_naive(time_series, end_idx - lag, window_length, window_number)
        hankel_past = hankel_past @ hankel_past.T

        # compute the scoring using the ika naive implementation
        score = implicit_krylov_approximation_naive(hankel_past, x0, 5, 9)
    elif key == "naive svd":

        # compile the future hankel matrix (H2)
        hankel_future = compile_hankel_naive(time_series, end_idx, window_length, window_number)

        # get the largest eigenvalue from decomposition
        x0, _, _ = np.linalg.svd(hankel_future, full_matrices=False)
        x0 = x0[:, 0]

        # compile the past hankel matrix (H1)
        hankel_past = compile_hankel_naive(time_series, end_idx - lag, window_length, window_number)

        # compute the scoring using the ika naive implementation
        score = exact_svd(hankel_past, x0, 5)
    else:
        raise ValueError(f"Key {key} not known.")

    # check the score for negativity (which is not possible)
    assert score >= 0-np.finfo(float).eps*1000, f"Score is negative {score}."
    return score


########################################################################################################################
# ------------------------------- Comparison Function ---------------------------------------------------------------- #
########################################################################################################################


def process_signal(signal_key: str, window_length: int, hdf_path: str, result_keys: list[str],
                   reference: str = "naive svd") -> dict[str:(float, float, int)]:

    # get the signal into RAM
    with h5py.File(hdf_path, 'r') as filet:
        signal = filet[signal_key][:]

    # specify the keys of the functions we plan to use (and make sure they are unique)
    function_keys = ["naive svd", "naive rsvd", "fft rsvd", "naive ika", "fft ika"]
    assert len(set(function_keys)) == len(function_keys), f"Function keys must not contain duplicates."
    assert reference == function_keys[0], f"{reference} has to be the first function key. Specified: {function_keys}."

    # create the results dict
    results = {col: [] for col in result_keys}

    # compute the amount of chunks we need to make and check whether we have at least one
    lag = window_length//3
    chunk_length = 2*window_length - 1 + lag
    chunk_number = signal.shape[0]//chunk_length
    if not chunk_number:
        return results

    # go over the chunks and compute the svd and save the result of the svd within the
    for chx in range(chunk_number):

        # create a random state so every function uses the same random state reliably
        # mainly to take care of the future vector generation
        seed = np.random.randint(1, 10_000_000)
        rnd_state = np.random.RandomState(seed)

        # make a comparison value
        cmp_val = -1

        # go over all the functions
        for key in function_keys:

            # compute the end index
            end_idx = (chx+1)*chunk_length

            # compute the result
            start = time.perf_counter_ns()
            score = transform(signal, window_length, window_length, lag, end_idx, key, rnd_state)
            elapsed = time.perf_counter_ns() - start

            # check whether we have computed the reference value
            if key == reference:
                cmp_val = score
            else:
                assert cmp_val >= 0, "We do not have a valid compare value, something is fishy."

            # keep (value, error, time)
            name = f"{signal_key}__{chx*chunk_length}_to_{(chx+1)*chunk_length}"
            results["identifier"].append(name)
            results["method"].append(key)
            results["score"].append(score)
            results["true-score"].append(cmp_val-score)
            results["time"].append(elapsed)
            results["cmp val"].append(cmp_val)
            results["random seed"].append(seed)
            results["window lengths"].append(window_length)

            # assert that every list in results has equal lengths
            assert len(set(len(values) for values in results.values())) == 1, "Something went wrong with the results."
    return results


def run_comparison():

    # create different window sizes and specify the amount of windows
    window_sizes = [int(ele) for ele in np.ceil(np.geomspace(100, 2000, num=100))[::-1]]

    # get the signal keys from the hdf5 file
    hdf_path = "UCRArchive_2018.hdf5"
    with h5py.File(hdf_path) as filet:
        signal_keys = list(filet.keys())

    # make the example results dict
    results = {"identifier": [],
               "method": [],
               "score": [],
               "true-score": [],
               "time": [],
               "cmp val": [],
               "random seed": [],
               "window lengths": []}

    # go through the signals and window sizes and compute the values
    for window_size in window_sizes:
        for signal_key in tqdm(signal_keys, desc=f"Computing for window size {window_size}"):
            tmp_results = process_signal(signal_key, window_size, hdf_path, list(results.keys()))
            for key in results:
                results[key].extend(tmp_results[key])

        # check whether results are empty
        if not results[list(results.keys())[0]]:
            continue

        # put into dataframe
        df = pd.DataFrame(results)

        # make a debug print for all the methods
        methods = list(df["method"].unique())
        print(f"\nWindow Size: {window_size} [supp. {df[df['method'] == methods[0]].shape[0]}].")
        print("------------------------------------")
        for method in methods:
            tmp_df = df[df['method'] == method]
            mape = tmp_df['true-score'].abs().mean()
            elapsed = tmp_df["time"].mean()
            print(f"Method {method:<15} error: {mape:0.10f} time: {elapsed/1_000_000:0.5f}.")

        # save it under the window size and clear the results
        df.to_csv(f"Results_WindowSize_{window_size}.csv")

        # clear the lists
        for value in results.values():
            value.clear()

run_comparison()
