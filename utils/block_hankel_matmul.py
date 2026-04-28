import numpy as np
import scipy.fft as spfft
import timeit


def block_hankel_left_matmat_direct(B, A):
    """
    Compute A @ H directly, where H is a block Hankel matrix.

    H has block structure:

        H[i, j] = B[i + j]

    Parameters
    ----------
    B : ndarray, shape (p + q - 1, m, n)
        Block sequence. B[k] is an m x n matrix block.

    A : ndarray
        Either:
        - dense form: shape (s, p*m)
        - block form: shape (p, s, m)

    Returns
    -------
    Z : ndarray
        If A was dense, returns dense shape (s, q*n).
        If A was block-form, returns block shape (q, s, n).
    """
    B = np.asarray(B)
    A = np.asarray(A)

    num_blocks, m, n = B.shape

    return_dense = False

    if A.ndim == 2:
        # A is dense: shape (s, p*m)
        s, cols = A.shape

        if cols % m != 0:
            raise ValueError("A has incompatible number of columns.")

        p = cols // m
        A_blocks = A.reshape(s, p, m).transpose(1, 0, 2)
        return_dense = True

    elif A.ndim == 3:
        # A is already block-form: shape (p, s, m)
        p, s, m_A = A.shape

        if m_A != m:
            raise ValueError("Block dimensions do not match.")

        A_blocks = A

    else:
        raise ValueError("A must have shape (s, p*m) or (p, s, m).")

    q = num_blocks - p + 1

    if q <= 0:
        raise ValueError("Need len(B) >= p.")

    Z_blocks = np.zeros((q, s, n), dtype=np.result_type(B, A))

    for j in range(q):
        for i in range(p):
            Z_blocks[j] += A_blocks[i] @ B[i + j]

    if return_dense:
        return Z_blocks.transpose(1, 0, 2).reshape(s, q * n)

    return Z_blocks


def block_hankel_left_matmat_fft(B, A, fft_len: int):
    """
    Compute A @ H using FFT, where H is a block Hankel matrix.

    H has block structure:

        H[i, j] = B[i + j]

    Parameters
    ----------
    B : ndarray, shape (p + q - 1, m, n)
        Block sequence. B[k] is an m x n matrix block.

    A : ndarray
        Either:
        - dense form: shape (s, p*m)
        - block form: shape (p, s, m)

    Returns
    -------
    Z : ndarray
        If A was dense, returns dense shape (s, q*n).
        If A was block-form, returns block shape (q, s, n).
    """
    B = np.asarray(B)
    A = np.asarray(A)

    num_blocks, m, n = B.shape

    return_dense = False

    if A.ndim == 2:
        # A is dense: shape (s, p*m)
        s, cols = A.shape

        if cols % m != 0:
            raise ValueError("A has incompatible number of columns.")

        p = cols // m
        A_blocks = A.reshape(s, p, m).transpose(1, 0, 2)
        return_dense = True

    elif A.ndim == 3:
        # A is already block-form: shape (p, s, m)
        p, s, m_A = A.shape

        if m_A != m:
            raise ValueError("Block dimensions do not match.")

        A_blocks = A

    else:
        raise ValueError("A must have shape (s, p*m) or (p, s, m).")

    q = num_blocks - p + 1

    if q <= 0:
        raise ValueError("Need len(B) >= p.")

    # For left multiplication:
    #
    # Z_j = sum_i A_i B_{i+j}
    #
    # This is obtained from convolution of reversed A with B.
    A_rev = A_blocks[::-1]

    FA = np.fft.fft(A_rev, n=fft_len, axis=0)  # shape (fft_len, s, m)
    FB = np.fft.fft(B,     n=fft_len, axis=0)  # shape (fft_len, m, n)

    # Frequency-wise matrix-matrix multiplication:
    #
    # FZ[ell] = FA[ell] @ FB[ell]
    #
    # Shapes:
    #   FA[ell] : (s, m)
    #   FB[ell] : (m, n)
    #   FZ[ell] : (s, n)
    FZ = FA @ FB                                # shape (fft_len, s, n)

    conv = np.fft.ifft(FZ, n=fft_len, axis=0)

    # Extract the desired Hankel result
    Z_blocks = conv[p - 1 : p - 1 + q]          # shape (q, s, n)

    Z_blocks = np.real_if_close(Z_blocks)

    if return_dense:
        return Z_blocks.transpose(1, 0, 2).reshape(s, q * n)

    return Z_blocks


def block_hankel_right_matmat_direct_strided(B, X):
    """
    Compute H @ X without FFT, using a vectorized einsum.

    H has block structure:

        H[i, j] = B[i + j]

    Parameters
    ----------
    B : ndarray, shape (p + q - 1, m, n)
        Block sequence. B[k] is an m x n matrix block.

    X : ndarray
        Either:
        - dense form: shape (q*n, r)
        - block form: shape (q, n, r)

    Returns
    -------
    Y : ndarray
        If X was dense, returns dense shape (p*m, r).
        If X was block-form, returns block shape (p, m, r).
    """
    B = np.asarray(B)
    X = np.asarray(X)

    num_blocks, m, n = B.shape
    return_dense = False

    if X.ndim == 2:
        rows, r = X.shape

        if rows % n != 0:
            raise ValueError("X has incompatible number of rows.")

        q = rows // n
        X_blocks = X.reshape(q, n, r)
        return_dense = True

    elif X.ndim == 3:
        q, n_X, r = X.shape

        if n_X != n:
            raise ValueError("Block dimensions do not match.")

        X_blocks = X

    else:
        raise ValueError("X must have shape (q*n, r) or (q, n, r).")

    p = num_blocks - q + 1

    if p <= 0:
        raise ValueError("Need len(B) >= q.")

    # sliding_window_view gives shape (p, m, n, q)
    B_windows = np.lib.stride_tricks.sliding_window_view(B, window_shape=q, axis=0)

    # Move the window axis so that:
    # BH[i, j] = B[i + j]
    # BH has shape (p, q, m, n)
    BH = np.moveaxis(B_windows, -1, 1)

    # Y[i, a, c] = sum_{j,b} BH[i,j,a,b] * X_blocks[j,b,c]
    Y_blocks = np.einsum("ijmn,jnr->imr", BH, X_blocks, optimize=True)

    Y_blocks = np.real_if_close(Y_blocks)

    if return_dense:
        return Y_blocks.reshape(p * m, r)

    return Y_blocks


def block_hankel_right_matmat_fft(B, X, fft_len: int):
    """
    Compute H @ X using FFT, where H is a block Hankel matrix.

    H has block structure:

        H[i, j] = B[i + j]

    Parameters
    ----------
    B : ndarray, shape (p + q - 1, m, n)
        Block sequence. B[k] is an m x n matrix block.

    X : ndarray
        Either:
        - dense form: shape (q*n, r)
        - block form: shape (q, n, r)

    Returns
    -------
    Y : ndarray
        If X was dense, returns dense shape (p*m, r).
        If X was block-form, returns block shape (p, m, r).
    """
    B = np.asarray(B)
    X = np.asarray(X)

    num_blocks, m, n = B.shape

    return_dense = False

    if X.ndim == 2:
        # X is dense: shape (q*n, r)
        rows, r = X.shape

        if rows % n != 0:
            raise ValueError("X has incompatible number of rows.")

        q = rows // n
        X_blocks = X.reshape(q, n, r)
        return_dense = True

    elif X.ndim == 3:
        # X is already block-form: shape (q, n, r)
        q, n_X, r = X.shape

        if n_X != n:
            raise ValueError("Block dimensions do not match.")

        X_blocks = X

    else:
        raise ValueError("X must have shape (q*n, r) or (q, n, r).")

    p = num_blocks - q + 1

    if p <= 0:
        raise ValueError("Need len(B) >= q.")

    # Hankel multiplication:
    #
    # Y_i = sum_j B_{i+j} X_j
    #
    # This becomes convolution of B with reversed X.
    X_rev = X_blocks[::-1]

    FB = np.fft.fft(B,     n=fft_len, axis=0)  # shape (fft_len, m, n)
    FX = np.fft.fft(X_rev, n=fft_len, axis=0)  # shape (fft_len, n, r)

    # Frequency-wise matrix-matrix multiplication:
    #
    # FY[ell] = FB[ell] @ FX[ell]
    #
    # Shapes:
    #   FB[ell] : (m, n)
    #   FX[ell] : (n, r)
    #   FY[ell] : (m, r)
    FY = FB @ FX                                # shape (fft_len, m, r)

    conv = np.fft.ifft(FY, n=fft_len, axis=0)

    # Extract the desired Hankel result
    Y_blocks = conv[q - 1 : q - 1 + p]          # shape (p, m, r)

    Y_blocks = np.real_if_close(Y_blocks)

    if return_dense:
        return Y_blocks.reshape(p * m, r)

    return Y_blocks


def block_hankel_left_matmat_direct_strided(B, A):
    """
    Compute A @ H without FFT, using a vectorized einsum.

    H has block structure:

        H[i, j] = B[i + j]

    Parameters
    ----------
    B : ndarray, shape (p + q - 1, m, n)
        Block sequence. B[k] is an m x n matrix block.

    A : ndarray
        Either:
        - dense form: shape (s, p*m)
        - block form: shape (p, s, m)

    Returns
    -------
    Z : ndarray
        If A was dense, returns dense shape (s, q*n).
        If A was block-form, returns block shape (q, s, n).
    """
    B = np.asarray(B)
    A = np.asarray(A)

    num_blocks, m, n = B.shape
    return_dense = False

    if A.ndim == 2:
        s, cols = A.shape

        if cols % m != 0:
            raise ValueError("A has incompatible number of columns.")

        p = cols // m
        A_blocks = A.reshape(s, p, m).transpose(1, 0, 2)
        return_dense = True

    elif A.ndim == 3:
        p, s, m_A = A.shape

        if m_A != m:
            raise ValueError("Block dimensions do not match.")

        A_blocks = A

    else:
        raise ValueError("A must have shape (s, p*m) or (p, s, m).")

    q = num_blocks - p + 1

    if q <= 0:
        raise ValueError("Need len(B) >= p.")

    # sliding_window_view gives shape (q, m, n, p)
    B_windows = np.lib.stride_tricks.sliding_window_view(B, window_shape=p, axis=0)

    # Move the window axis so that:
    # BH[j, i] = B[j + i]
    # BH has shape (q, p, m, n)
    BH = np.moveaxis(B_windows, -1, 1)

    # Z[j, s, n] = sum_{i,m} A_blocks[i,s,m] * BH[j,i,m,n]
    Z_blocks = np.einsum("ism,jimn->jsn", A_blocks, BH, optimize=True)

    Z_blocks = np.real_if_close(Z_blocks)

    if return_dense:
        return Z_blocks.transpose(1, 0, 2).reshape(s, q * n)

    return Z_blocks


def build_block_hankel(B, p, q):
    """
    Explicitly build the full block Hankel matrix H.

    B has shape (p + q - 1, m, n).
    H has shape (p*m, q*n).
    """
    _, m, n = B.shape

    H = np.zeros((p * m, q * n), dtype=B.dtype)

    for i in range(p):
        for j in range(q):
            H[i*m:(i+1)*m, j*n:(j+1)*n] = B[i + j]

    return H


# -------------------------
# Example
# -------------------------

p = 70   # block rows of H
q = p   # block columns of H
m = 2   # rows per block
n = 1   # cols per block
s = 10   # rows of left multiplier A

B = np.arange((p + q - 1) * m * n).reshape(p + q - 1, m, n) + 1

# Dense left multiplier A, shape (s, p*m)
A = np.arange(s * p * m).reshape(s, p * m) + 1

H = build_block_hankel(B, p, q)

# get the best fft length
fft_length = spfft.next_fast_len(p+q-1, real=True)

Z_explicit = A @ H
Z_direct = block_hankel_left_matmat_direct(B, A)
Z_fft = block_hankel_left_matmat_fft(B, A, fft_length)
Z_strided = block_hankel_left_matmat_direct_strided(B, A)

run_num = 100
fft_time = timeit.timeit(lambda: block_hankel_left_matmat_fft(B, A, fft_length), number=run_num) / run_num * 1000
print('FFT multiplication took:', fft_time)
direct_time = timeit.timeit(lambda: A @ H, number=run_num) / run_num * 1000
print('Direct multiplication took:', direct_time)
direct_time = timeit.timeit(lambda: block_hankel_left_matmat_direct_strided(B, A), number=run_num) / run_num * 1000
print('Strided multiplication took:', direct_time)
print('direct/FFT time', direct_time / fft_time, '(if result > 1 fft is faster)')

print("\nDirect matches explicit?")
print(np.allclose(Z_direct, Z_explicit))

print("FFT matches explicit?")
print(np.allclose(Z_fft, Z_explicit))

print("Strided matches explicit?")
print(np.allclose(Z_strided, Z_explicit))

print()
A2 = np.arange(q*n*s).reshape(q*n, s) + 1
Z_explicit = H @ A2
Z_fft = block_hankel_right_matmat_fft(B, A2, fft_length)
Z_strided = block_hankel_right_matmat_direct_strided(B, A2)

run_num = 100
fft_time = timeit.timeit(lambda: block_hankel_right_matmat_fft(B, A2, fft_length), number=run_num) / run_num * 1000
print('FFT multiplication took:', fft_time)
direct_time = timeit.timeit(lambda: H @ A2, number=run_num) / run_num * 1000
print('Direct multiplication took:', direct_time)
direct_time = timeit.timeit(lambda: block_hankel_right_matmat_direct_strided(B, A2), number=run_num) / run_num * 1000
print('Strided multiplication took:', direct_time)
print('direct/FFT time', direct_time / fft_time, '(if result > 1 fft is faster)')

print("FFT matches explicit?")
print(np.allclose(Z_fft, Z_explicit))

print("Strided matches explicit?")
print(np.allclose(Z_strided, Z_explicit))