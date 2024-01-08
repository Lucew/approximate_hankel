import os.path
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_files(path="results_changepoint", combined_name="Changepoint_WindowSizes.parquet"):

    # check whether the file exists
    if os.path.isfile(combined_name):
        data = pd.read_parquet(combined_name)
    else:
        # get all the csv files in the directory
        files = glob(os.path.join(path, "Results_WindowSize_*.csv"))

        # load the files into memory
        files = {file: pd.read_csv(file, header=0) for file in files}

        # concatenate the files into one
        data = pd.concat(files.values(), ignore_index=True)

        # save the file into pandas parquet format
        data.to_parquet(combined_name)
    return data


def main():

    # load the data into memory
    data = load_files()

    # get the method names
    methods = data["method"].unique()

    # get the window sizes
    window_sizes = sorted(data["window lengths"].unique())

    # go over the methods and compute the error
    for method in methods:
        print(f"Method {method} we have a mean error of: \t{data[data['method'] == method]['true-score'].abs().median()}")

    # go over the methods and window sizes and compute the average computation time
    computation_times = {"method": [], "window size": [], "computation time [ms]": []}
    for method in methods:
        for wl in window_sizes:
            computation_times["method"].append(method)
            computation_times["window size"].append(wl)
            computation_times["computation time [ms]"].append(data[(data['method'] == method) & (data["window lengths"] == wl)]["time"].median()/1_000_000)

    # make into a dataframe and plot
    computation_times = pd.DataFrame(computation_times)
    plot = sns.lineplot(data=computation_times, x="window size", y="computation time [ms]", hue="method")
    plot.set(yscale="log")
    plt.show()


if __name__ == "__main__":
    main()
