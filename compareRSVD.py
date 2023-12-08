import fbpca
import numpy as np
import computeSVD as csvd
from glob import glob
import os
import h5py
import time
from functools import partial
from tqdm import tqdm
import multiprocessing as mp


def rename_files():
    # find all the window lengths
    window_sizes = list(int(folder.split('_')[-1][:-1]) for folder in glob(os.path.join("result", "*/")))

    # find all the window lengths
    for window_size in window_sizes:
        file_paths = list(os.path.join(root, file) for root, _, files in os.walk(f"result/window_size_{window_size}")
                          for file in files if file.endswith(".npz"))
        for file in tqdm(file_paths, desc=f"Rename files for window size {window_size}"):
            # split the name and rename it
            name_list = file.split("__")
            if name_list[1].endswith("x100"):
                continue
            name_list[1] = f"{name_list[1]}x100"
            os.rename(file, "__".join(name_list))


def process_signal(svd_result_file: str, hdf_path: str, random_ranks: list[int], subspace_repetitions: list[int],
                   additional_ranks: list[int]) -> None:
    # extract all the information we need from the result file we are working with
    only_file = os.path.splitext(os.path.split(svd_result_file)[-1])[0]
    signal_key, dimensions, chunk_number, _ = only_file.split("__")
    chunk_number = int(chunk_number)
    dimensions = [int(ele) for ele in dimensions.split("x")]
    assert len(dimensions) == 2, f'We expected dimensions to have for INTxINT but got {dimensions}.'

    # load the data and find the corresponding chunk information
    svd_results = np.load(svd_result_file)

    # find the different chunks
    chunk_names = set("__".join(ele.split("__")[:-1]) for ele in svd_results.keys() if ele.startswith("Chunk"))
    assert len(chunk_names) == chunk_number, (f'We expected {chunk_number} chunks from '
                                              f'{svd_result_file} but got {len(chunk_names)}.')

    # find the extensions of the chunks
    chunk_extensions = set(ele.split("__")[-1] for ele in svd_results.keys() if ele.startswith("Chunk"))
    assert (len(chunk_extensions) == 2
            and "eigenvalues" in chunk_extensions
            and "left_vectors" in chunk_extensions), \
        (f"We expected the extensions of the chunks in the file to be [eigenvalues, left_vectors] "
         f"but found: {chunk_extensions}.")

    # create generator over all combinations and check that the approximated rank is smaller than the
    # maximum possible rank of the matrix (as it is done in fbpca line 1519)
    #
    # This makes sure we are still computing and approximation
    parameter_generator = [(rd_rank, sbs_rep, add_rank)
                           for rd_rank in random_ranks
                           for sbs_rep in subspace_repetitions
                           for add_rank in additional_ranks if (rd_rank+add_rank) < min(dimensions)/1.25]

    # make a small function to create the identifier for each combination
    def make_identifier(rd_rank: int, sbs_rep: int, add_rank: int):
        return f"rnd_rank_{rd_rank}__sbsp_rep_{sbs_rep}__add_rank_{add_rank}"

    # make a dict to save the results
    vector_collector = dict()

    # create all the timing measurements
    for rd_rank, sbs_rep, add_rank in parameter_generator:
        ident = make_identifier(rd_rank, sbs_rep, add_rank)
        vector_collector[f"rsvd_process_times__{ident}"] = np.zeros((chunk_number,))

    # open the file and get the signals from it into RAM
    with h5py.File(hdf_path, 'r') as filet:
        signal = filet[signal_key][:]

    # go over the chunks and compute the svd and save the result of the svd within the
    for chx, name in enumerate(chunk_names):

        # get the end position of the name of the chunk
        end_position = int(name.split("__")[1].split("_to_")[1])

        # create the matrix and compute the decomposition
        hankel = csvd.compile_hankel(signal, end_position, dimensions[0], dimensions[1])

        # get the corresponding vectors from the svd
        svd_eigenvalues = svd_results[f'{name}__eigenvalues']
        svd_left_vectors = svd_results[f'{name}__left_vectors']

        # go over all hyperparameters specified
        for rd_rank, sbs_rep, add_rank in parameter_generator:

            # create the hyperparameter string identifier
            identifier = make_identifier(rd_rank, sbs_rep, add_rank)

            # make a safety copy of the hankel matrix
            hankel = hankel.copy()

            # keep track of the computation time
            start_time = time.perf_counter_ns()

            # compute the decomposition
            left_vectors, eigenvalues, _ = fbpca.pca(hankel, k=rd_rank, raw=True, n_iter=sbs_rep, l=rd_rank + add_rank)

            # keep track of the time it took
            elapsed = time.perf_counter_ns() - start_time
            vector_collector[f"rsvd_process_times__{identifier}"][chx] = elapsed

            # make the result float32 for memory savings
            left_vectors = left_vectors.astype("float32")
            eigenvalues = eigenvalues.astype("float32")

            # save into the vector collection
            tmpn = f"{name}__{identifier}"
            vector_collector[f"{tmpn}__left_vectors_difference"] = np.sum(np.square(left_vectors
                                                                                    - svd_left_vectors[:, :rd_rank]),
                                                                          axis=1)
            vector_collector[f"{tmpn}__eigenvalues_difference"] = np.square(eigenvalues-svd_eigenvalues[:rd_rank])

    # save the result in the corresponding files
    name = f"{signal_key}__{dimensions[0]}x{dimensions[1]}__{chunk_number}__rsvd_difference.npz"
    name = os.path.join(*os.path.split(svd_result_file)[:-1], name)
    np.savez_compressed(name, **vector_collector)


def main(file_path: str, result_path: str, random_rank: list[int], subspace_repetitions: list[int],
         additional_rank: list[int]):

    # find all the folders with different window sizes
    window_folders = glob(os.path.join(result_path, "*/"))

    # find and check number of windows that it is only one single window number within all the folders
    n_windows = set(file.split("__")[1].split("x")[-1]
                    for _, _, files in os.walk(result_path) for file in files if file.endswith(".npz"))
    assert len(n_windows) == 1, f"There is more than one n_windows in {result_path}: {n_windows}"

    # make the function partial, so we only have one input argument
    function_handle = partial(process_signal,
                              hdf_path=file_path,
                              random_ranks=random_rank,
                              subspace_repetitions=subspace_repetitions,
                              additional_ranks=additional_rank)

    # go over different window sizes and keep a progress bar (go from the longest as they have the longest duration)
    for window_folder in window_folders:

        # find all the files within the corresponding folder
        file_list = glob(os.path.join(window_folder, "*__svd.npz"))

        # make multiprocessing to make use of multicore cpus
        with mp.Pool(mp.cpu_count()) as pp:
            desc = f'RSVD for folder {window_folder}'
            # as the execution of imap_unordered is greedy, we need to iterate over the return values, but
            # since it returns nothing we can just pass. Unfortunately, the greedy version is the only one
            # which to the best of my knowledge supports unordered execution.
            for result in tqdm(pp.imap_unordered(function_handle, file_list, chunksize=100),
                               desc=desc, total=len(file_list)):
                pass


if __name__ == "__main__":
    main(file_path="UCRArchive_2018.hdf5",
         result_path="result",
         random_rank=[5, 7, 10],
         subspace_repetitions=[0, 1, 2, 3],
         additional_rank=[0, 2, 4])
