import multiprocessing as mp
import numpy as np
import numba as nb
import h5py
import pandas as pd
from functools import partial
from tqdm import tqdm
from threadpoolctl import threadpool_limits
from utils.fastHankel import compile_hankel_parallel
from changepointComparison import randomized_hankel_svd_naive, randomized_hankel_svd_fft, compile_hankel_fft
from preprocessing.changepoint_simulator import ChangeSimulator
import argparse


def process_signal(signal: tuple[str, np.ndarray, int], window_length: int, signal_length: int,
                   eigenvalues: int, reconstructions: int,
                   subspace_iterations: list[int], oversampling_constants: list[int],
                   thread_limit: int = 1) -> dict[str: list]:

    # unwrap name and signal (we only have this tuple, so we can use imap_unordered for multiprocessing)
    name, signal, seed = signal

    # create a random seed
    random_state = np.random.RandomState(seed)

    # limit the threads for the BLAS and numpy multiplications
    nb.set_num_threads(thread_limit)
    with threadpool_limits(limits=thread_limit):

        # construct the hankel matrix (including fft representation
        hankel = compile_hankel_parallel(signal, signal_length, window_length, window_length)
        hankel_fft, fft_len, _ = compile_hankel_fft(signal, signal_length, window_length, window_length)

        # compute the complete decomposition of the matrix
        left_eigenvectors, svd_vals, right_eigenvectors = np.linalg.svd(hankel, full_matrices=False, hermitian=True)
        summed = sum(svd_vals)
        median_value = np.median(svd_vals)

        # make the svd values
        svd_values = []
        for idx, val in enumerate(svd_vals[:eigenvalues]):
            svd_values.append((f'eigenvalue {idx}', val))

        # compute the reconstruction errors
        svd_error = []
        for idx in range(1, reconstructions+1):
            diff = hankel - np.dot(left_eigenvectors[:, :idx] * svd_vals[:idx], right_eigenvectors[:idx, :])
            svd_error.append((f'reconstruction svd {idx}', np.linalg.norm(diff)))

        # go over all the combinations from subspace iterations and oversampling constants
        # create a generator to make the tuples
        tuple_generator = ((rank, q, p)
                           for rank in range(1, eigenvalues+1)
                           for q in subspace_iterations
                           for p in oversampling_constants)
        rsvd_naive_results = []
        rsvd_fft_results = []
        for rank, q, p in tuple_generator:

            # compute the rsvd with the appropriate parametrization
            left_eigenvectors, vals, right_eigenvectors = randomized_hankel_svd_naive(hankel, rank, q, p, window_length,
                                                                                      window_length, random_state)
            diff = hankel - np.dot(left_eigenvectors[:, :idx] * vals[:idx], right_eigenvectors[:idx, :])
            rsvd_naive_results.append((f'reconstruction rsvd naive q={q}, p={p}, rank={rank}', np.linalg.norm(diff)))

            # compute the rsvd with the appropriate parametrization
            left_eigenvectors, vals, right_eigenvectors = randomized_hankel_svd_fft(hankel_fft, fft_len, rank, q, p,
                                                                                    window_length, window_length,
                                                                                    random_state)
            diff = hankel - np.dot(left_eigenvectors[:, :idx] * vals[:idx], right_eigenvectors[:idx, :])
            rsvd_fft_results.append((f'reconstruction rsvd fft q={q}, p={p}, rank={rank}', np.linalg.norm(diff)))
    return name, seed, svd_values, svd_error, summed, median_value, rsvd_naive_results, rsvd_fft_results


def real_decomposition():

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

    # get the signal into the RAM
    # with h5py.File("UCRArchive_2018.hdf5", 'r') as filet:
        # signal = filet[signal_key][:]


# create a signal generator
def signal_generator(repetitions: int, signal_length: int, simulation=True):

    if simulation:
        for idx in range(repetitions):
            # make a new signal generator
            seed = np.random.randint(1, 10_000_000)
            rnd_state = np.random.RandomState(seed)
            sig_gen = ChangeSimulator(signal_length, signal_length // 2, rnd_state)
            for name, signal in sig_gen.yield_signals():
                yield name, signal, seed


def signal_loader(signal_information: list[tuple[str, int]], signal_length: int):
    with h5py.File('UCRArchive_2018.hdf5') as filet:
        for signal_name, segments in signal_information:
            sig = filet[signal_name]
            for segment in range(segments):
                yield f'{signal_name}_{segment*signal_length}', sig[segment*signal_length:(segment+1)*signal_length], 0


def get_signal_length(window_size: int):
    return 2 * window_size - 1


def run_decomposition(simulation=True):

    # create different window sizes and specify the number of windows
    if simulation:
        window_sizes = [100, 200, 1000, 5000]
        window_sizes = window_sizes[::-1]
    else:
        window_sizes = [100, 200, 500, 1000]
        window_sizes = window_sizes[::-1]

    # define some parameters
    eigenvalues = 20
    subspace_iterations = [0, 1, 2, 3]
    oversampling_constants = [0, 2, 5]
    signal_number = 500

    # if we do the comparison for the real signals we need to figure out which signals have the correct
    # length for our simulations
    usable_signals = {wl: [] for wl in window_sizes}
    if not simulation:
        with h5py.File('UCRArchive_2018.hdf5') as filet:

            # get all the signal keys from the hdf5 file
            signals = list(filet.keys())

            # go through the signals and check whether they can be used
            for sig in tqdm(signals, desc='Find all usable signals from file'):

                # get the signal data from the file
                sig_data = filet[sig]

                # go through the window lengths
                for wl in window_sizes:
                    wg = get_signal_length(wl)
                    if sig_data.shape[0] > wg:
                        usable_signals[wl].append((sig, sig_data.shape[0]//wg))

    # make a results dict
    results = {'signal identifier': [], 'seed': [], 'eigenvalue sum': [], 'median eigenvalue': []}

    # make all the names for the eigenvalues
    for idx in range(eigenvalues):
        results[f'eigenvalue {idx}'] = []
    # make all the names for the reconstructions
    for idx in range(1, eigenvalues+1):
        results[f'reconstruction svd {idx}'] = []

    # make the names for the methods
    tuple_generator = ((rank, q, p)
                       for rank in range(1, eigenvalues+1)
                       for q in subspace_iterations
                       for p in oversampling_constants)
    for rank, q, p in tuple_generator:
        results[f'reconstruction rsvd naive q={q}, p={p}, rank={rank}'] = []
        results[f'reconstruction rsvd fft q={q}, p={p}, rank={rank}'] = []

    # make multiprocessing to make use of multicore cpus
    for window_size in window_sizes:

        # compute the end index of the signal
        sig_length = get_signal_length(window_size)

        # create the function with a function handle
        function_handle = partial(process_signal,
                                  window_length=window_size,
                                  signal_length=sig_length,
                                  eigenvalues=eigenvalues,
                                  reconstructions=eigenvalues,
                                  subspace_iterations=subspace_iterations,
                                  oversampling_constants=oversampling_constants)

        # estimate the cardinality
        if simulation:
            # define the number of signals and the signal generator
            seed = np.random.randint(1, 10_000_000)
            rnd_state = np.random.RandomState(seed)
            sig_gen = ChangeSimulator(sig_length, sig_length // 2, rnd_state)

            # check how many different signals the signal generator makes
            different_signals = [ele for ele in sig_gen.yield_signals()]
            number_simulations = len(different_signals)
            card = number_simulations * signal_number

        else:
            card = sum(ele[1] for ele in usable_signals[window_size])

        # create the signal generator
        if simulation:
            sig_gen = signal_generator(signal_number, sig_length, simulation)
        else:
            sig_gen = signal_loader(usable_signals[window_size], sig_length)

        # make the description
        desc = f'Decompositions for Window Size {window_size}'
        with mp.Pool(mp.cpu_count()) as pp:
            for result in tqdm(pp.imap_unordered(function_handle, sig_gen, chunksize=10), desc=desc,
                               total=card):

                # unpack the result
                name, seed, svd_vals, svd_error, eigensum, median_eigval, rsvd_naive_results, rsvd_fft_results = result

                # save the result into the dict
                results['signal identifier'].append(name)
                results['seed'].append(seed)
                results['eigenvalue sum'].append(eigensum)
                results['median eigenvalue'].append(median_eigval)
                for name, val in svd_vals:
                    results[name].append(val)
                for name, val in svd_error:
                    results[name].append(val)
                for name, val in rsvd_naive_results:
                    results[name].append(val)
                for name, val in rsvd_fft_results:
                    results[name].append(val)

        # put into dataframe
        df = pd.DataFrame(results)

        # make a debug print for all the methods
        print(f"\nWindow Size: {window_size} [supp. {df.shape[0]}].")
        print("------------------------------------")

        # go over all the different reconstructions
        for col in df.columns:
            if not col.startswith('reconstruction'): continue
            if not col.endswith(' 1') and not col.endswith('=1'): continue
            print(f"{col}: {df[col].mean()}")

        # save it under the window size and clear the results
        df.to_csv(f"Decomposition{'_simulated' if simulation else ''}_Results_WindowSize_{window_size}.csv")

        # clear the lists
        for value in results.values():
            value.clear()


def svd_hankel_signal(signal: tuple[str, np.ndarray, int], window_length: int, signal_length: int,
                      thread_limit: int = 1) -> dict[str: list]:

    # unwrap name and signal (we only have this tuple, so we can use imap_unordered for multiprocessing)
    name, signal, seed = signal

    # limit the threads for the BLAS and numpy multiplications
    nb.set_num_threads(thread_limit)
    with threadpool_limits(limits=thread_limit):

        # construct the hankel matrix (including fft representation
        hankel = compile_hankel_parallel(signal, signal_length, window_length, window_length)
        hankel_fft, fft_len, _ = compile_hankel_fft(signal, signal_length, window_length, window_length)

        # compute the complete decomposition of the matrix
        svd_vals_real = np.linalg.eigvalsh(hankel)

    # get all the negative eigenvalues and create a list from it
    names = []
    window_sizes = []
    eigenvalue_numbers = []
    eigenvalues = []
    for idx, val in enumerate(svd_vals_real):
        if val <= 0:
            names.append(name)
            window_sizes.append(window_length)
            eigenvalue_numbers.append(idx)
            eigenvalues.append(val)
    return names, window_sizes, eigenvalue_numbers, eigenvalues


def run_negative_check(simulation=True):
    # create different window sizes and specify the number of windows
    if simulation:
        window_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
        window_sizes = window_sizes[::-1]
    else:
        window_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        window_sizes = window_sizes[::-1]
    signal_number = 500


    # if we do the comparison for the real signals, we need to figure out which signals have the correct
    # length for our simulations
    usable_signals = {wl: [] for wl in window_sizes}
    if not simulation:
        with h5py.File('UCRArchive_2018.hdf5') as filet:

            # get all the signal keys from the hdf5 file
            signals = list(filet.keys())

            # go through the signals and check whether they can be used
            for sig in tqdm(signals, desc='Find all usable signals from file'):

                # get the signal data from the file
                sig_data = filet[sig]

                # go through the window lengths
                for wl in window_sizes:
                    wg = get_signal_length(wl)
                    if sig_data.shape[0] > wg:
                        usable_signals[wl].append((sig, sig_data.shape[0] // wg))

    # make a results dict
    results = {'signal identifier': [], 'window size': [], 'eigenvalue number': [], 'eigenvalue': []}

    # make multiprocessing to make use of multicore cpus
    for window_size in window_sizes:

        # compute the end index of the signal
        sig_length = get_signal_length(window_size)

        # create the function with a function handle
        function_handle = partial(svd_hankel_signal,
                                  window_length=window_size,
                                  signal_length=sig_length,
                                  thread_limit=1)

        # estimate the cardinality
        if simulation:
            # define the number of signals and the signal generator
            seed = np.random.randint(1, 10_000_000)
            rnd_state = np.random.RandomState(seed)
            sig_gen = ChangeSimulator(sig_length, sig_length // 2, rnd_state)

            # check how many different signals the signal generator makes
            different_signals = [ele for ele in sig_gen.yield_signals()]
            number_simulations = len(different_signals)
            card = number_simulations * signal_number

        else:
            card = sum(ele[1] for ele in usable_signals[window_size])

        # create the signal generator
        if simulation:
            sig_gen = signal_generator(signal_number, sig_length, simulation)
        else:
            sig_gen = signal_loader(usable_signals[window_size], sig_length)

        # make the description
        desc = f'Check Eigenvalues for Window Size {window_size} in {"simulated" if simulation else "real"} signals'
        with mp.Pool(mp.cpu_count()) as pp:
            for result in tqdm(pp.imap_unordered(function_handle, sig_gen, chunksize=10), desc=desc,
                               total=card):

                # unpack the result
                names, window_sizes, eigenvalue_numbers, eigenvalues = result

                # save the result into the dict
                results = {'signal identifier': [], 'window size': [], 'eigenvalue number': [], 'eigenvalue': []}
                results['signal identifier'].extend(names)
                results['window size'].extend(window_sizes)
                results['eigenvalue number'].extend(eigenvalue_numbers)
                results['eigenvalue'].extend(eigenvalues)

        # put into dataframe
        df = pd.DataFrame(results)

        # make a debug print for all the methods
        print(f"\nWindow Size: {window_size} [supp. {df.shape[0]}].")
        print("------------------------------------")
        print(f"There were: {df.shape[0]} negative eigenvalues.\n")

        # save it under the window size and clear the results
        df.to_csv(f"Negative_Eigenvalues{'_simulated' if simulation else ''}_WindowSize_{window_size}.csv")

        # clear the lists
        for value in results.values():
            value.clear()


def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Changepoint algorithms speed and accuracy comparison.')
    parser.add_argument('-sim', '--simulated', type=boolean_string, default=True,
                        help='Specifies whether to use simulated or real signals.')
    parser.add_argument('-neg', '--negative', type=boolean_string, default=False,
                        help='Specifies whether to check for zero or negative eigenvalues.')
    args = parser.parse_args()
    if args.simulated:
        print("Running comparison on simulated signals.")
        if args.negative:
            run_negative_check(simulation=True)
        else:
            run_decomposition(simulation=True)
    else:
        print("Running comparison on real signals.")
        if args.negative:
            run_negative_check(simulation=False)
        else:
            run_decomposition(simulation=False)
