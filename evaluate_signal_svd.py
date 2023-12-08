import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp


def find_files(result_path: str, window_size: int) -> list[str]:

    # find the correct folder
    target_folder = ""
    for folder in glob(os.path.join(result_path, "*/")):
        if folder.endswith(f'_{window_size}{os.path.sep}'):
            target_folder = folder

    # check whether we found the corresponding folder
    assert target_folder, f'We could not find the correct folder for window size {window_size} in {result_path}.'

    # get the path to all the files
    return glob(os.path.join(target_folder, '*__svd.npz'))


def extract_data(file_path: str) -> (list[str], list[np.ndarray]):
    # load the file
    filet = np.load(file_path)

    # get the interesting keys
    key_list = [key for key in filet.keys() if key.endswith('eigenvalues')]
    indices = []
    eigenvalues = []
    for key in key_list:

        # insert the eigenvalues into the dict
        eigenvalue_array = filet[key]
        indices.append(key)

        # normalize the eigenvectors
        eigenvalue_array = eigenvalue_array / np.sum(eigenvalue_array)
        eigenvalues.append(eigenvalue_array)
    assert len(indices) == int(os.path.splitext(file_path)[0].split("__")[-2]), f"Something is wrong @{file_path}."
    return indices, eigenvalues


def load_eigenvalues(file_list: list[str], window_size: int) -> pd.DataFrame:

    # check for the size of the eigenvalues
    filet = np.load(file_list[0])
    key_list = [key for key in filet.keys() if key.endswith('eigenvalues')]
    vals = filet[key_list[0]].shape[0]

    # make a list of eigenvalues
    eigenvalues = {f"Val {idx}": [] for idx in range(vals)}
    eigenvalues["index"] = []
    with mp.Pool(mp.cpu_count()//2) as pp:

        # read the data in unordered fashion using multiple processes
        for indices, eigs in tqdm(pp.imap_unordered(extract_data, file_list, chunksize=150),
                                  desc=f'Loading data (window size {window_size})', total=len(file_list)):

            # put in the names of each signal and corresponding chunks
            eigenvalues["index"].extend(indices)

            # go over the eigenvalues for each chunk within a signal and save the value in the column
            # of the future dataframe
            for eig_array in eigs:
                for idx, ei in enumerate(eig_array):
                    eigenvalues[f"Val {idx}"].append(ei)

    df = pd.DataFrame(data=eigenvalues)
    df = df.set_index("index")
    return df


def etl_eigenvalues(result_path: str = "result") -> list[str]:

    # find all the window lengths
    window_sizes = list(int(folder.split('_')[-1][:-1]) for folder in glob(os.path.join(result_path, "*/")))

    # go over the window length and get the eigenvalues into dataframe
    names = []
    for window_length in window_sizes:

        # create the name and check whether it is already saved
        name = os.path.join(result_path, f"eigenvalues_window_{window_length}")
        names.append(name)
        if os.path.isfile(name):
            print(f"Result dataframe for window length {window_length} already exists in current directory.")
            continue

        # load the data and save it as a dataframe
        file_list = find_files("result", window_size=window_length)
        df = load_eigenvalues(file_list, window_length)
        df.to_parquet(name)
    return names


def plot_signals(df: pd.DataFrame, window_length: int):
    ax = df.boxplot(column=list(df.columns)[:10], return_type='axes')
    ax.set_yscale('log')
    ax.set_ylim([10e-5, 1])
    ax.set_title(f'Histograms of Eigenvalues for {df.shape[0]} Hankel-Matrices\nwith window length {window_length}')
    plt.show()


def main():
    # create the dataframes
    dataframes = etl_eigenvalues()
    dataframes.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)

    # go over the dataframes and load the frames
    for frame in tqdm(dataframes, desc='Load and plot dataframes'):
        df = pd.read_parquet(frame)
        plot_signals(df, int(frame.split('_')[-1]))


if __name__ == '__main__':
    main()
