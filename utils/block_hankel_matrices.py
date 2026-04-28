import time
import timeit

import numpy as np
import scipy.fft as spfft


def block_hankel_matvec_fft(B, x, fft_length: int):
    """
    Multiply a block Hankel matrix by a block vector using FFT.

    Parameters
    ----------
    B : ndarray, shape (p + q - 1, m, n)
        Block sequence. B[k] is the k-th matrix block.
    x : ndarray, shape (q, n)
        Block vector. x[j] is the j-th vector block.

    Returns
    -------
    y : ndarray, shape (p, m)
        Output block vector.
    """
    B = np.asarray(B)
    x = np.asarray(x)

    num_blocks, m, n = B.shape
    q, n_x = x.shape

    if n_x != n:
        raise ValueError("Block dimensions do not match.")

    # Since B has length p + q - 1
    p = num_blocks - q + 1
    if p <= 0:
        raise ValueError("Need len(B) >= len(x).")

    # Hankel matvec:
    # y_i = sum_j B[i + j] x_j
    #
    # This is obtained from convolution of B with reversed x.
    x_rev = x[::-1]

    FB = np.fft.rfft(B, n=fft_length, axis=0)       # shape (fft_len, m, n)
    FX = np.fft.rfft(x_rev, n=fft_length, axis=0)   # shape (fft_len, n)

    # For each frequency ell:
    # FY[ell] = FB[ell] @ FX[ell]
    # FY = np.einsum("lmn,ln->lm", FB, FX)
    FY = (FB @ FX[..., None])[..., 0]

    conv = np.fft.irfft(FY, n=fft_length, axis=0)

    # Extract the Hankel part
    y = conv[q - 1 : q - 1 + p]

    # If inputs are real, remove tiny imaginary roundoff
    return np.real_if_close(y)


def block_hankel_matvec_direct(B, x):
    """
    Direct reference implementation.
    """
    B = np.asarray(B)
    x = np.asarray(x)

    num_blocks, m, n = B.shape
    q = x.shape[0]
    p = num_blocks - q + 1

    y = np.zeros((p, m), dtype=np.result_type(B, x))

    for i in range(p):
        for j in range(q):
            y[i] += B[i + j] @ x[j]

    return y


def block_hankel_matvec_einsum(B, x):
    """
    Direct block Hankel matvec using einsum.

    B has shape (p + q - 1, m, n)
    x has shape (q, n)

    Returns y with shape (p, m)
    """
    B = np.asarray(B)
    x = np.asarray(x)

    num_blocks, m, n = B.shape
    q, n_x = x.shape

    if n_x != n:
        raise ValueError("Block dimensions do not match.")

    p = num_blocks - q + 1

    # hankel_indices[i, j] = i + j
    hankel_indices = np.arange(p)[:, None] + np.arange(q)[None, :]

    # BH[i, j] = B[i + j]
    # Shape: (p, q, m, n)
    BH = B[hankel_indices]

    # y[i, a] = sum_{j,b} BH[i,j,a,b] * x[j,b]
    y = np.einsum("ijab,jb->ia", BH, x)

    return y

def main():
    # -------------------------
    # Example
    # -------------------------

    p = 70  # number of block rows
    q = p  # number of block columns
    m = 2  # rows per block
    n = 1  # columns per block

    # Block sequence B_0, ..., B_{p+q-2}
    B = np.arange((p + q - 1) * m * n).reshape(p + q - 1, m, n) + 1
    # B = np.random.random((p + q - 1) * m * n).reshape(p + q - 1, m, n) + 1

    # Block vector x_0, ..., x_{q-1}
    x = np.arange(q * n).reshape(q, n) + 1
    # x = np.random.random(q * n).reshape(q, n) + 1

    # get the best fft length
    fft_length = spfft.next_fast_len(p+q-1, real=True)

    # first run for jit compiler
    y_fft = block_hankel_matvec_fft(B, x, fft_length)
    y_direct = block_hankel_matvec_einsum(B, x)

    # second run
    run_num = 10
    fft_time = timeit.timeit(lambda: block_hankel_matvec_fft(B, x, fft_length), number=run_num) / run_num * 1000
    print('FFT multiplication took:', fft_time)
    direct_time = timeit.timeit(lambda: block_hankel_matvec_einsum(B, x), number=run_num) / run_num * 1000
    print('Direct multiplication took:', direct_time)
    print('direct/FFT time', direct_time / fft_time, '(if result > 1 fft is faster)')

    y_fft = block_hankel_matvec_fft(B, x, fft_length)
    y_direct = block_hankel_matvec_einsum(B, x)
    y_direct_naive = block_hankel_matvec_direct(B, x)

    """
    print("B blocks:")
    for k, Bk in enumerate(B):
        print(f"B[{k}] =\n{Bk}\n")

    print("x blocks:")
    for j, xj in enumerate(x):
        print(f"x[{j}] = {xj}")

    print("\ny from FFT:")
    print(y_fft)

    print("\ny from direct multiplication:")
    print(y_direct)
    """
    print("\nDo they match?")
    print(np.allclose(y_fft, y_direct))
    print(np.all(np.equal(y_direct, y_direct_naive)))

if __name__ == "__main__":
    main()