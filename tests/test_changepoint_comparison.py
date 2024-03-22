from changepointComparison import transform, compile_hankel_parallel, compile_hankel_naive
from preprocessing.changepoint_simulator import ChangeSimulator
from threadpoolctl import threadpool_limits
import numpy as np
import numba as nb
import timeit


# define a window length
window_length = 5000

# define the function of interest
function = "naive rsvd"

# define the number of threads
threads = [1, 2, 4, 6, 8, 12]

# compute the parameters
lag = window_length // 3
sig_length = 2*window_length - 1 + lag
rnd_state = np.random.RandomState(5)

# simulate a signal
sig_gen = ChangeSimulator(sig_length, window_length+lag//2, rnd_state)
transform(sig_gen.mean_change(), window_length, window_length, lag, sig_length, function, rnd_state)

# test the hankel creation scripts
nb.set_num_threads(4)
sig = sig_gen.mean_change()
hankel_future_naive = compile_hankel_naive(sig, sig_length, window_length, window_length, safe=True)
hankel_future_parallel = compile_hankel_parallel(sig, sig_length, window_length, window_length, safe=True)
print(np.array_equal(hankel_future_naive, hankel_future_parallel))

# run the transformation for different thread sizes
for thr in threads:
    with threadpool_limits(limits=thr):
        nb.set_num_threads(thr)
        tmp = lambda: transform(sig_gen.mean_change(), window_length, window_length, lag, sig_length, function, rnd_state)
        print(f'Decomposition {thr} threads:', timeit.timeit(tmp, number=10))
        tmp = lambda: compile_hankel_parallel(sig_gen.mean_change(), sig_length, window_length, window_length)
        print(f'Hankel parallel construction {thr} threads:', timeit.timeit(tmp, number=100))
        tmp = lambda: compile_hankel_naive(sig_gen.mean_change(), sig_length, window_length, window_length)
        print(f'Hankel naive construction {thr} threads:', timeit.timeit(tmp, number=100))
        print()
