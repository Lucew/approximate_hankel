import collections
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp


def find_files(result_path: str, window_size: int, suffix: str) -> list[str]:

    # find the correct folder
    target_folder = ""
    for folder in glob(os.path.join(result_path, "*/")):
        if folder.endswith(f'_{window_size}{os.path.sep}'):
            target_folder = folder

    # check whether we found the corresponding folder
    assert target_folder, f'We could not find the correct folder for window size {window_size} in {result_path}.'

    # get the path to all the files
    return glob(os.path.join(target_folder, f'*{suffix}'))


def extract_data(file_path: str) -> (list[str], list[np.ndarray]):
    # load the file
    filet = np.load(file_path)

    # get the process times from the file
    process_times = [key for key in filet.keys() if key.split("__")[0].endswith("process_times")]

    # get the parameters from the name of the file
    file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
    parameters = file_name.split("__")

    # get the different parametrization information and times
    information = {(key, tuple(int(ele) for ele in parameters[1].split("x"))): filet[key] for key in process_times}
    for key in process_times:
        assert filet[key].shape[0] == int(parameters[2]), f"Something is wrong with {file_path}."
    return information


def load_timings(result_path: str) -> pd.DataFrame:

    # find all the files that we have
    file_list = [file
                 for folder in glob(os.path.join(result_path, "*"))
                 for file in glob(os.path.join(folder, "*.npz"))]

    # make a dict to save the values
    values = collections.defaultdict(lambda: collections.defaultdict(list))

    # create the name and check whether it is already saved
    name = os.path.join(result_path, f"Timings.parquet")
    if os.path.isfile(name):
        print("Timings dataframe already available")
        return pd.read_parquet(name)

    with mp.Pool(mp.cpu_count()//2) as pp:

        # read the data in unordered fashion using multiple processes
        for information in tqdm(pp.imap_unordered(extract_data, file_list, chunksize=150),
                                desc=f'Loading data', total=len(file_list)):
            for (key, (dim1, dim2)), vals in information.items():
                values[(dim1, dim2)][key].extend(val for val in vals)

    # check that dim2 is only one value
    dim2_set = set(key[1] for key in values.keys())
    assert len(dim2_set) == 1, f"There are results with a different second dimension: {dim2}."

    # create numpy arrays from the lists
    for vals in values.values():
        for key, val in vals.items():
            vals[key] = np.array(val)

    # create a dataframe from the values for each dimensionality
    df = pd.DataFrame.from_dict(values, orient="index")
    df.to_parquet(name)
    return df


def main():

    # get the timing information
    df = load_timings("result")

    # get the unique first dimension
    dims = sorted(df.index.unique())
    print(dims)

    # get the timing for normal svd
    # dimensions = set(col.split("__") for col in df.columns if col.startswith("process_times"))
    for col in df.columns:
        # check the rank
        if not col.startswith("process") and ("rnd_rank_5" not in col or "add_rank_2" not in col): continue
        # get the information we need for the plot
        std = [df[col][dim].std()/1_000_000 for dim in dims if df[col][dim] is not None]
        mean = [df[col][dim].mean()/1_000_000 for dim in dims if df[col][dim] is not None]
        dimsx = [dim[0] for dim in dims if df[col][dim] is not None]

        # make the label
        if col == "process_times":
            name = "SVD"
        else:
            name = f"Sbs. Rep.={col.split('__')[2].split('_')[-1]}"

        # make the plot
        plt.errorbar(dimsx, mean, yerr=std, label=name)
    plt.yscale("log")
    plt.ylabel("Time in [ms]")
    plt.xlabel("Dim. 2")
    plt.title("Calculation Time with growing first dimension (second is 100)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
