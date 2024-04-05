import numpy as np
import scipy.linalg
import scipy as sp
import numba as nb
import h5py
import time
import pandas as pd
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import argparse
import numba

# hankel matmul functions
from utils.fastHankel import fast_numba_hankel_matmul as fast_hankel_matmul
from utils.fastHankel import fast_numba_hankel_left_matmul as fast_hankel_left_matmul
from utils.fastHankel import get_fast_hankel_representation
from utils.fastHankel import trigger_numba_matmul_jit
from utils.fastHankel import compile_hankel_parallel

# simulations of signals
from preprocessing import changepoint_simulator as cps

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

def compile_hankel_fft(time_series: np.ndarray, end_index: int, window_size: int, rank: int,
                       lag: int = 1) -> (np.ndarray, int, np.ndarray):
    hankel_rfft, fft_len, signal = get_fast_hankel_representation(time_series, end_index, window_size, rank, lag=lag)
    return hankel_rfft[:, 0], fft_len, signal


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
        projection_matrix_q = fast_hankel_matmul(hankel_fft, length_windows, fft_length, projection_matrix_q, lag=1)

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
# --------------------------------- Implicitly restarted Lanczos bi-diagonalization SST ------------------------------ #
########################################################################################################################

# this code is taken from and inspired by:
# https://github.com/bwlewis/irlbpy and by the algorithms in the paper (especially algorithm 2.1)
# Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
# J. Baglama and L. Reichel, SIAM J. Sci. Comput.
# 2005


def orthogonalize(matrix1: np.ndarray, matrix2: np.ndarray):
    """
    Orthogonalize a vector or matrix Y against the columns of the matrix X.
    This function requires that the column dimension of Y is less than X, and
    that Y and X have the same number of rows.
    """
    return matrix1 - matrix2 @ (matrix2.T @ matrix1)


# Simple utility function used to check linear dependencies during computation
def invcheck(x):
    eps2 = 2*np.finfo(float).eps
    if x > eps2:
        x = 1/x
    else:
        x = 0
    # warnings.warn("Ill-conditioning encountered, result accuracy may be poor")
    return x


def irlb(hankel_matrix: np.ndarray, nu: int, tol: float = 0.0001, maxit: int = 50):
    """Estimate a few of the largest singular values and corresponding singular
    vectors of matrix using the implicitly restarted Lanczos bidiagonalization
    method of Baglama and Reichel, see:

    Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
    J. Baglama and L. Reichel, SIAM J. Sci. Comput.
    2005

    Keyword arguments:
    tol -- An estimation tolerance. Smaller means more accurate estimates.
    maxit -- Maximum number of Lanczos iterations allowed.

    Given an input matrix A of dimension j * k, and an input desired number
    of singular values nu, the function returns a tuple X with five entries:

    X[0] A j * nu matrix of estimated left singular vectors.
    X[1] A vector of length nu of estimated singular values.
    X[2] A k * nu matrix of estimated right singular vectors.
    X[3] The number of Lanczos iterations run.
    X[4] The number of matrix-vector products run.

    The algorithm estimates the truncated singular value decomposition:
    A.dot(X[2]) = X[0]*X[1].
    """
    m = hankel_matrix.shape[0]
    n = hankel_matrix.shape[1]

    m_b = min((nu+20, 3*nu, n))  # Working dimension size
    mprod = 0
    it = 0
    j = 0
    k = nu
    smax = 1

    # initialize the helper matrices
    V = np.zeros((n, m_b))
    W = np.zeros((m, m_b))
    F = np.zeros((n, 1))
    B = np.zeros((m_b, m_b))
    V[:, 0] = np.random.randn(n)  # Initial vector
    V[:, 0] = V[:, 0]/np.linalg.norm(V)  # normalize initial vector

    # define some stuff, so it exists (but throws an error if not initialized properly)
    left_eigvecs = None
    eigvals = None
    right_eigvecs = None

    # go through the iterations (use for in range instead of while as it is marginally faster)
    # and break as soon as we hit convergence
    for it in range(maxit):
        if it > 0:
            j = k
        W[:, j] = hankel_matrix @ V[:, j]
        mprod += 1
        if it > 0:
            W[:, j] = orthogonalize(W[:, j], W[:, 0:j])  # NB W[:,0:j] selects columns 0,1,...,j-1
        s = np.linalg.norm(W[:, j])
        sinv = invcheck(s)
        W[:, j] = sinv*W[:, j]

        # Lanczos process
        while j < m_b:
            F = W[:, j] @ hankel_matrix
            mprod += 1
            F = F - s*V[:, j]
            F = orthogonalize(F, V[:, 0:j+1])
            fn = np.linalg.norm(F)
            fninv = invcheck(fn)
            F  = fninv * F
            if j < m_b-1:
                V[:, j+1] = F
                B[j, j] = s
                B[j, j+1] = fn
                W[:, j+1] = hankel_matrix @ V[:, j+1]
                mprod += 1
                # One step of classical Gram-Schmidt...
                W[:, j+1] = W[:, j+1] - fn*W[:, j]
                # ...with full re-orthogonalization
                W[:, j+1] = orthogonalize(W[:, j+1], W[:, 0:(j+1)])
                s = np.linalg.norm(W[:, j+1])
                sinv = invcheck(s)
                W[:, j+1] = sinv * W[:, j+1]
            else:
                B[j, j] = s
            j += 1

        # end of the lanczos process
        left_eigvecs, eigvals, right_eigvecs = np.linalg.svd(B)
        R = fn * left_eigvecs[m_b-1, :]  # Residuals
        if it < 1:
            smax = eigvals[0]  # Largest Ritz value
        else:
            smax = max((eigvals[0], smax))

        conv = sum(np.abs(R[0:nu]) < tol * smax)
        if conv < nu:  # Not converged yet
            k = max(conv+nu, k)
            k = min(k, m_b-3)
        else:
            break

        # Update the Ritz vectors
        V[:, 0:k] = V[:, 0:m_b].dot(right_eigvecs.transpose()[:, 0:k])
        V[:, k] = F
        B = np.diag(eigvals)
        B[0:k, k] = R[0:k]
        # Update the left approximate singular vectors
        W[:, 0:k] = W[:, 0:m_b].dot(left_eigvecs[:, 0:k])

    U = W[:, 0:m_b].dot(left_eigvecs[:, 0:nu])
    V = V[:, 0:m_b].dot(right_eigvecs.transpose()[:, 0:nu])
    return U, eigvals[0:nu], V, it, mprod


def irlb_fft(hankel_fft_matrix: np.ndarray, nu: int, fft_length: int, windows_number: int, windows_length: int,
             tol: float = 0.0001, maxit: int = 50):
    """Estimate a few of the largest singular values and corresponding singular
    vectors of matrix using the implicitly restarted Lanczos bidiagonalization
    method of Baglama and Reichel, see:

    Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
    J. Baglama and L. Reichel, SIAM J. Sci. Comput.
    2005

    Keyword arguments:
    tol -- An estimation tolerance. Smaller means more accurate estimates.
    maxit -- Maximum number of Lanczos iterations allowed.

    Given an input matrix A of dimension j * k, and an input desired number
    of singular values nu, the function returns a tuple X with five entries:

    X[0] A j * nu matrix of estimated left singular vectors.
    X[1] A vector of length nu of estimated singular values.
    X[2] A k * nu matrix of estimated right singular vectors.
    X[3] The number of Lanczos iterations run.
    X[4] The number of matrix-vector products run.

    The algorithm estimates the truncated singular value decomposition:
    A.dot(X[2]) = X[0]*X[1].
    """
    m = windows_length
    n = windows_number

    m_b = min((nu+20, 3*nu, n))  # Working dimension size
    mprod = 0
    it = 0
    j = 0
    k = nu
    smax = 1

    # initialize the helper matrices
    V = np.zeros((n, m_b))
    W = np.zeros((m, m_b))
    F = np.zeros((n, 1))
    B = np.zeros((m_b, m_b))
    V[:, 0] = np.random.randn(n)  # Initial vector
    V[:, 0] = V[:, 0]/np.linalg.norm(V)  # normalize initial vector

    # define some stuff, so it exists (but throws an error if not initialized properly)
    left_eigvecs = None
    eigvals = None
    right_eigvecs = None

    # go through the iterations (use for in range instead of while as it is marginally faster)
    # and break as soon as we hit convergence
    for it in range(maxit):
        if it > 0:
            j = k

        # W[:, j] = hankel_matrix @ V[:, j] using fft
        W[:, j] = fast_hankel_matmul(hankel_fft_matrix, windows_length, fft_length, V[:, j:j+1], lag=1)

        mprod += 1
        if it > 0:
            W[:, j] = orthogonalize(W[:, j], W[:, 0:j])  # NB W[:,0:j] selects columns 0,1,...,j-1
        s = np.linalg.norm(W[:, j])
        sinv = invcheck(s)
        W[:, j] = sinv*W[:, j]

        # Lanczos process
        while j < m_b:
            # F = W[:, j] @ hankel_matrix using fft
            F = fast_hankel_left_matmul(hankel_fft_matrix, windows_number, fft_length, W[:, j:j+1].T, lag=1)
            mprod += 1
            F = F - s*V[:, j]
            F = orthogonalize(F, V[:, 0:j+1])
            fn = np.linalg.norm(F)
            fninv = invcheck(fn)
            F  = fninv * F
            if j < m_b-1:
                V[:, j+1] = F
                B[j, j] = s
                B[j, j+1] = fn
                # W[:, j+1] = hankel_matrix @ V[:, j+1] using fft
                W[:, j + 1] = fast_hankel_matmul(hankel_fft_matrix, windows_length, fft_length, V[:, j+1:j+2], lag=1)[:, 0]
                mprod += 1
                # One step of classical Gram-Schmidt...
                W[:, j+1] = W[:, j+1] - fn*W[:, j]
                # ...with full re-orthogonalization
                W[:, j+1] = orthogonalize(W[:, j+1], W[:, 0:(j+1)])
                s = np.linalg.norm(W[:, j+1])
                sinv = invcheck(s)
                W[:, j+1] = sinv * W[:, j+1]
            else:
                B[j, j] = s
            j += 1

        # end of the lanczos process
        left_eigvecs, eigvals, right_eigvecs = np.linalg.svd(B)
        R = fn * left_eigvecs[m_b-1, :]  # Residuals
        if it < 1:
            smax = eigvals[0]  # Largest Ritz value
        else:
            smax = max((eigvals[0], smax))

        conv = sum(np.abs(R[0:nu]) < tol * smax)
        if conv < nu:  # Not converged yet
            k = max(conv+nu, k)
            k = min(k, m_b-3)
        else:
            break

        # Update the Ritz vectors
        V[:, 0:k] = V[:, 0:m_b].dot(right_eigvecs.transpose()[:, 0:k])
        V[:, k] = F
        B = np.diag(eigvals)
        B[0:k, k] = R[0:k]
        # Update the left approximate singular vectors
        W[:, 0:k] = W[:, 0:m_b].dot(left_eigvecs[:, 0:k])

    U = W[:, 0:m_b].dot(left_eigvecs[:, 0:nu])
    V = V[:, 0:m_b].dot(right_eigvecs.transpose()[:, 0:nu])
    return U, eigvals[0:nu], V, it, mprod


def rayleigh_ritz(hankel_matrix: np.ndarray, eigvec_future: np.ndarray, rank: int):
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in
    [1].
    """

    # compute the singular value decomposition of the hankel matrix
    left_eigenvectors, *_ = irlb(hankel_matrix, rank, np.finfo(float).eps)

    # compute the similarity score as defined in the ika sst paper and also return our u for the
    # feedback loop in figure 3 of the paper
    scores = left_eigenvectors.T @ eigvec_future
    return 1 - (scores.T @ scores).sum()


def rayleigh_ritz_fft(hankel_fft_matrix: np.ndarray, eigvec_future: np.ndarray, rank: int,
                      fft_length: int, windows_number: int, windows_length: int):
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in
    [1].
    """

    # compute the singular value decomposition of the hankel matrix
    left_eigenvectors, *_ = irlb_fft(hankel_fft_matrix, rank, fft_length, windows_number, windows_length,
                                     np.finfo(float).eps)

    # compute the similarity score as defined in the ika sst paper and also return our u for the
    # feedback loop in figure 3 of the paper
    scores = left_eigenvectors.T @ eigvec_future
    return 1 - (scores.T @ scores).sum()


########################################################################################################################
# --------------------------------- Exact SST ------------------------------------------------------------------------ #
########################################################################################################################


def exact_svd(hankel_matrix: np.ndarray, eigvec_future: np.ndarray, rank: int) -> np.ndarray:
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in
    [1].
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
              key: str, random_state: np.random.RandomState, power_iterations: int = 10) -> (float, int, int):

    # check that the time series fits and we did not make an error
    assert len(time_series) >= end_idx, f"Time series is too short ({time_series.shape}) for start: {end_idx}."

    # create random vector for power iterations with the future hankel matrix as described in [1]
    x0 = random_state.rand(window_number)
    x0 /= np.linalg.norm(x0)

    # add small noise to the data so the ika sst does not break
    time_series += random_state.normal(scale=1e-4)

    # make a variable that measures hankel time
    hankel_construction_time = 0
    decomposition_time = 0
    if key == "fft rsvd":

        # compile the future hankel matrix (H2)
        start = time.perf_counter_ns()
        hankel_future, fft_length, _ = compile_hankel_fft(time_series, end_idx, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # get the first singular matrix vector of future hankel matrix
        start = time.perf_counter_ns()
        x0, _, _ = randomized_hankel_svd_fft(hankel_future, fft_length, k=1, subspace_iteration_q=3,
                                             oversampling_p=14, length_windows=window_length,
                                             number_windows=window_number, random_state=random_state)
        decomposition_time += time.perf_counter_ns() - start

        # compile the past hankel matrix (H1)
        start = time.perf_counter_ns()
        hankel_past, fft_length, _ = compile_hankel_fft(time_series, end_idx - lag, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # compute the scoring using the ika naive implementation
        start = time.perf_counter_ns()
        score = rsvd_score_fft(hankel_past, x0, fft_length, k=5, subspace_iteration_q=3, oversampling_p=10,
                               length_windows=window_length, number_windows=window_number, random_state=random_state)
        decomposition_time += time.perf_counter_ns() - start

    elif key == "naive rsvd":

        # compile the future hankel matrix (H2)
        start = time.perf_counter_ns()
        hankel_future = compile_hankel_parallel(time_series, end_idx, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # get the first singular vector of the future hankel matrix
        start = time.perf_counter_ns()
        x0, _, _ = randomized_hankel_svd_naive(hankel_future, k=1, subspace_iteration_q=3, oversampling_p=14,
                                               length_windows=window_length, number_windows=window_number,
                                               random_state=random_state)
        decomposition_time += time.perf_counter_ns() - start

        # compile the past hankel matrix (H1)
        start = time.perf_counter_ns()
        hankel_past = compile_hankel_parallel(time_series, end_idx - lag, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # compute the scoring using the ika naive implementation
        start = time.perf_counter_ns()
        score = rsvd_score_naive(hankel_past, x0, k=5, subspace_iteration_q=3, oversampling_p=10,
                                 length_windows=window_length, number_windows=window_number, random_state=random_state)
        decomposition_time += time.perf_counter_ns() - start

    elif key == "naive ika":

        # compile the future hankel matrix (H2)
        start = time.perf_counter_ns()
        hankel_future = compile_hankel_parallel(time_series, end_idx, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # make the power iterations
        start = time.perf_counter_ns()
        _, x0 = power_method(hankel_future, x0, power_iterations)
        decomposition_time += time.perf_counter_ns() - start

        # compile the past hankel matrix (H1) and compute outer product C as in the paper
        start = time.perf_counter_ns()
        hankel_past = compile_hankel_parallel(time_series, end_idx - lag, window_length, window_number)
        hankel_past = hankel_past @ hankel_past.T
        hankel_construction_time += time.perf_counter_ns() - start

        # compute the scoring using the ika naive implementation
        start = time.perf_counter_ns()
        score = implicit_krylov_approximation_naive(hankel_past, x0, 5, 9)
        decomposition_time += time.perf_counter_ns() - start
    elif key == "naive svd":

        # compile the future hankel matrix (H2)
        start = time.perf_counter_ns()
        hankel_future = compile_hankel_parallel(time_series, end_idx, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # get the largest eigenvalue from decomposition
        start = time.perf_counter_ns()
        x0, _, _ = np.linalg.svd(hankel_future, full_matrices=False)
        x0 = x0[:, 0]
        decomposition_time += time.perf_counter_ns() - start

        # compile the past hankel matrix (H1)
        start = time.perf_counter_ns()
        hankel_past = compile_hankel_parallel(time_series, end_idx - lag, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # compute the scoring using the ika naive implementation
        start = time.perf_counter_ns()
        score = exact_svd(hankel_past, x0, 5)
        decomposition_time += time.perf_counter_ns() - start
    elif key == "naive irlb":

        # compile the future hankel matrix (H2)
        start = time.perf_counter_ns()
        hankel_future = compile_hankel_parallel(time_series, end_idx, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # get the largest eigenvalue from decomposition
        start = time.perf_counter_ns()
        x0, *_ = irlb(hankel_future, 1)
        decomposition_time += time.perf_counter_ns() - start

        # compile the past hankel matrix (H1)
        start = time.perf_counter_ns()
        hankel_past = compile_hankel_parallel(time_series, end_idx - lag, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # compute the scoring using the naive irlb
        start = time.perf_counter_ns()
        score = rayleigh_ritz(hankel_past, x0, 5)
        decomposition_time += time.perf_counter_ns() - start

    elif key == "fft irlb":  # TODO Fix the implementation of fft irlb (has errors too high)
        raise ValueError("FFT IRLB currently not working.")
        # compile the future hankel matrix (H2)
        start = time.perf_counter_ns()
        hankel_future, fft_length, _ = compile_hankel_fft(time_series, end_idx, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # get the first singular matrix vector of future hankel matrix
        start = time.perf_counter_ns()
        x0, *_ = irlb_fft(hankel_future, 1, fft_length, window_number, window_length)
        decomposition_time += time.perf_counter_ns() - start

        # compile the past hankel matrix (H1)
        start = time.perf_counter_ns()
        hankel_past, fft_length, _ = compile_hankel_fft(time_series, end_idx - lag, window_length, window_number)
        hankel_construction_time += time.perf_counter_ns() - start

        # compute the scoring using the naive irlb
        start = time.perf_counter_ns()
        score = rayleigh_ritz_fft(hankel_past, x0, 5, fft_length, window_number, window_length)
        decomposition_time += time.perf_counter_ns() - start
    else:
        raise ValueError(f"Key {key} not known.")

    # check the score for negativity (which is not possible)
    if score <= 0-np.finfo(float).eps*1000:
        print(f"Score is negative {score} for method {key}.")
    return score, decomposition_time, hankel_construction_time


########################################################################################################################
# ------------------------------- Comparison Function ---------------------------------------------------------------- #
########################################################################################################################


def process_signal(signal_key: str, window_length: int, hdf_path: str, result_keys: list[str],
                   reference: str = "naive svd", thread_limit: int = 6) -> dict[str:(float, float, int)]:

    # get the signal into the RAM
    with h5py.File(hdf_path, 'r') as filet:
        signal = filet[signal_key][:]

    # specify the keys of the functions we plan to use (and make sure they are unique)
    # function_keys = ["naive svd", "naive rsvd", "naive irlb", "fft irlb", "fft rsvd", "naive ika"]
    function_keys = ["naive svd", "naive rsvd", "naive irlb", "fft rsvd", "naive ika"]
    assert len(set(function_keys)) == len(function_keys), f"Function keys must not contain duplicates."
    assert reference == function_keys[0], f"{reference} has to be the first function key. Specified: {function_keys}."

    # create the results dict
    results = {col: [] for col in result_keys}

    # check whether the signal has nones or inf
    if np.any(np.logical_or(np.isnan(signal), np.isinf(signal))):
        print(f"Skipped Signal: {signal_key} as it contains NaN or Inf.")
        return results

    # compute the number of chunks we need to make and check whether we have at least one
    lag = window_length//3
    chunk_length = 2*window_length - 1 + lag
    chunk_number = signal.shape[0]//chunk_length
    if not chunk_number:
        return results

    # set the numba thread limit
    numba.set_num_threads(thread_limit)

    # limit the threads for the BLAS and numpy multiplications as well as for numba
    nb.set_num_threads(thread_limit)
    with threadpool_limits(limits=thread_limit):

        # go over the chunks and compute the svd and save the result of the svd
        for chx in range(chunk_number):

            # create a random state so every function uses the same random state reliably
            # mainly to take care of the future vector generation
            seed = np.random.randint(1, 10_000_000)
            rnd_state = np.random.RandomState(seed)

            # make a comparison value
            cmp_val = np.NAN

            # go over all the functions
            for key in function_keys:

                # compute the end index
                end_idx = (chx+1)*chunk_length

                # compute the result
                try:
                    score, decomposition_time, hankel_time = transform(signal, window_length, window_length, lag,
                                                                       end_idx, key, rnd_state)
                except np.linalg.LinAlgError:
                    print(f"There is something wrong with signal {signal_key} in chunk {chx}.")
                    break
                except ValueError:
                    print(f"There is something wrong with signal {signal_key} in chunk {chx}.")
                    break

                # check whether we have computed the reference value
                if key == reference:
                    cmp_val = score

                    # check whether the comparison value is negative (too much)
                    if cmp_val < -10*np.finfo(float).eps:
                        print(f"Compare value is way lower than zero {cmp_val} for signal {signal_key} in chunk {chx}.")
                        break
                else:
                    assert cmp_val != np.NAN, "We do not have a valid compare value, something is fishy."

                # keep (value, error, time)
                name = f"{signal_key}__{chx*chunk_length}_to_{(chx+1)*chunk_length}"
                results["identifier"].append(name)
                results["method"].append(key)
                results["score"].append(score)
                results["true-score"].append(cmp_val-score)
                results["decomposition time"].append(decomposition_time)
                results["hankel construction time"].append(hankel_time)
                results["cmp val"].append(cmp_val)
                results["random seed"].append(seed)
                results["window lengths"].append(window_length)
                results["max. threads"].append(thread_limit)

                # assert that every list in results has equal lengths
                assert len(set(len(values) for values in results.values())) == 1, "Something went wrong with the results."
    return results


def process_simulated_signal(window_length: int, result_keys: list[str], reference: str = "naive svd",
                             thread_limit: int = 6) -> dict[str:(float, float, int)]:

    # specify the keys of the functions we plan to use (and make sure they are unique)
    function_keys = ["naive svd", "naive rsvd", "naive irlb", "fft rsvd", "naive ika"]
    assert len(set(function_keys)) == len(function_keys), f"Function keys must not contain duplicates."
    assert reference == function_keys[0], f"{reference} has to be the first function key. Specified: {function_keys}."

    # create the results dict
    results = {col: [] for col in result_keys}

    # set the numba thread limit
    numba.set_num_threads(thread_limit)

    # limit the threads for the BLAS and numpy multiplications as well as for numba
    nb.set_num_threads(thread_limit)
    with threadpool_limits(limits=thread_limit):

        # create a random state so every function uses the same random state reliably
        # mainly to take care of the future vector generation
        seed = np.random.randint(1, 10_000_000)
        rnd_state = np.random.RandomState(seed)

        # create the signal generator
        lag = window_length // 3
        sig_length = 2*window_length - 1 + lag
        sig_gen = cps.ChangeSimulator(sig_length, window_length+lag//2, rnd_state)

        # go over all types of simulated signals
        for name, signal in sig_gen.yield_signals():

            # make a comparison value
            cmp_val = np.NAN

            # go over all the functions
            for key in function_keys:

                # compute the result
                score, decomposition_time, hankel_time = transform(signal, window_length, window_length, lag,
                                                                   sig_length, key, rnd_state)

                # check whether we have computed the reference value
                if key == reference:
                    cmp_val = score

                    # check whether the comparison value is negative (too much)
                    if cmp_val < -10*np.finfo(float).eps:
                        assert cmp_val != np.NAN, f"Compare value is way lower than zero: {cmp_val}."
                else:
                    assert cmp_val != np.NAN, "We do not have a valid compare value, something is fishy."

                # keep (value, error, time)
                results["Generator"].append(name)
                results["method"].append(key)
                results["score"].append(score)
                results["true-score"].append(cmp_val-score)
                results["decomposition time"].append(decomposition_time)
                results["hankel construction time"].append(hankel_time)
                results["cmp val"].append(cmp_val)
                results["random seed"].append(seed)
                results["window lengths"].append(window_length)
                results["max. threads"].append(thread_limit)

                # assert that every list in results has equal lengths
                assert len(set(len(values) for values in results.values())) == 1, "Something went wrong with the results."
    return results


def run_comparison():

    # create different window sizes and specify the number of windows
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
               "decomposition time": [],
               "hankel construction time": [],
               "cmp val": [],
               "random seed": [],
               "window lengths": [],
               "max. threads": []}

    # trigger the jit compilation of the numba functions for hankel matmul so they are not measured
    trigger_numba_matmul_jit()

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
        methods = sorted(list(df["method"].unique()), key=lambda x: x[::-1])
        print(f"\nWindow Size: {window_size} [supp. {df[df['method'] == methods[0]].shape[0]}].")
        print("------------------------------------")
        for method in methods:
            tmp_df = df[df['method'] == method]
            mape = tmp_df['true-score'].abs().mean()
            elapsed = tmp_df["decomposition time"].mean() + tmp_df["hankel construction time"].mean()
            print(f"Method {method:<15} error: {mape:0.10f} time: {elapsed/1_000_000:0.5f}.")

        # save it under the window size and clear the results
        df.to_csv(f"Results_WindowSize_{window_size}.csv")

        # clear the lists
        for value in results.values():
            value.clear()


def run_simulated_comparison():

    # create different window sizes and specify the number of windows
    window_sizes = [int(ele) for ele in np.ceil(np.geomspace(100, 5000, num=30))[::-1]]

    # define the threadlimits used
    threadlimits = [1, 2, 4, 6, 8, 10]

    # make the example results dict
    results = {"Generator": [],
               "method": [],
               "score": [],
               "true-score": [],
               "decomposition time": [],
               "hankel construction time": [],
               "cmp val": [],
               "random seed": [],
               "window lengths": [],
               "max. threads": []}

    # trigger the jit compilation of the numba functions for hankel matmul so they are not measured
    trigger_numba_matmul_jit()

    # go through the signals and window sizes and compute the values
    for window_size in window_sizes:
        for thread_lim in threadlimits:
            for _ in tqdm(range(100), desc=f"Computing for window size {window_size} with {thread_lim} threads"):
                tmp_results = process_simulated_signal(window_size, list(results.keys()), thread_limit=thread_lim)
                for key in results:
                    results[key].extend(tmp_results[key])

        # check whether results are empty
        if not results[list(results.keys())[0]]:
            continue

        # put into dataframe
        df = pd.DataFrame(results)

        # make a debug print for all the methods
        methods = sorted(list(df["method"].unique()), key=lambda x: x[::-1])
        print(f"\nWindow Size threadlim {thread_lim}: {window_size} [supp. {df[df['method'] == methods[0]].shape[0]}].")
        print("------------------------------------")
        for method in methods:
            tmp_df = df[df['method'] == method]
            mape = tmp_df['true-score'].abs().mean()
            elapsed = tmp_df["decomposition time"].mean() + tmp_df["hankel construction time"].mean()
            print(f"Method {method:<15} error: {mape:0.10f} time: {elapsed/1_000_000:0.5f}.")

        # save it under the window size and clear the results
        df.to_csv(f"Results_simulated_WindowSize_{window_size}.csv")

        # clear the lists
        for value in results.values():
            value.clear()


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def main():
    parser = argparse.ArgumentParser(description='Changepoint algorithms speed and accuracy comparison.')
    parser.add_argument('-sim', '--simulated', type=boolean_string, default=True,
                        help='Specifies whether to use simulated or real signals.')
    args = parser.parse_args()
    if args.simulated:
        print("Running comparison on simulated signals.")
        run_simulated_comparison()
    else:
        print("Running comparison on real signals.")
        run_comparison()


if __name__ == '__main__':
    main()
