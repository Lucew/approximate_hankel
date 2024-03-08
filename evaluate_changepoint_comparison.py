import os.path
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_files(path="results", simulated=True):

    # check whether the file exists
    combined_name = f"Changepoint{'_simulated' if simulated else ''}_WindowSizes.parquet"
    if os.path.isfile(combined_name):
        data = pd.read_parquet(combined_name)
    else:
        # get all the csv files in the directory
        # files = glob(os.path.join(path, "Results_simulated_WindowSize_*.csv"))
        files = glob(os.path.join(path, f"Results{'_simulated' if simulated else ''}_WindowSize_*.csv"))

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

    methods = [method for method in methods if method.endswith("rsvd") or method.endswith("svd") or method.endswith("ika") or method.endswith("naive irlb")]

    # go over the methods and window sizes and compute the average computation time
    computation_times = {"method": [], "window size N": [], "computation time [ms]": []}
    for method in methods:
        print(data[data['method'] == method].shape)
        for wl in window_sizes:
            computation_times["method"].append(method)
            computation_times["window size N"].append(wl)
            computation_times["computation time [ms]"].append(data[(data['method'] == method) & (data["window lengths"] == wl) & (data["max. threads"] == 4)]["time"].mean()/1_000_000)

    # make into a dataframe and plot
    plt.rcParams['text.usetex'] = True
    computation_times = pd.DataFrame(computation_times)
    plot = sns.lineplot(data=computation_times, x="window size N", y="computation time [ms]", hue="method")
    plot.set(yscale="log")

    complexity = {"naive svd": "O($N^{3}$)", "naive rsvd": "O($N^{2}$)", "fft rsvd": "O($N*logN$)", "naive ika": "O($N^{3}$)" , "naive irlb": "O($N^{2}$)"}
    for method in methods:
        plot.text(x=computation_times["window size N"].max() + 75, y=computation_times[computation_times['method'] == method]["computation time [ms]"].max(), s=complexity[method], va="center")
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plt.show()
    plt.rcParams['text.usetex'] = False

    # make into a dataframe and plot the histograms
    plot = sns.histplot(data[data['method'].str.endswith("rsvd") | data['method'].str.endswith("ika") | data['method'].str.endswith("naive irlb")], x="true-score", hue="method", multiple="dodge", bins='sturges', log_scale=[False, True])
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plt.show()
    plot = sns.histplot(data[data['method'] == "naive svd"], x="score", hue="method", multiple="dodge", bins='sturges', log_scale=[False, True])
    plt.show()

    # check the scalability of the methods per number of threads
    thread_numbers = data["max. threads"].unique()
    if len(thread_numbers) > 1:

        # find the maximum window size that we ran
        wl = max(window_sizes)
        wl = window_sizes[-2]
        print(wl)

        # go over the methods and window sizes and compute the average computation time
        computation_times = {"method": [], "thread number": [], "computation time [ms]": []}
        for method in methods:

            # compute the time for one thread for comparison
            cmp_val = data[(data['method'] == method) &
                           (data["window lengths"] == wl) & (data["max. threads"] == 1)]["time"].mean() / 1_000_000
            cmp_val2 = data[(data['method'] == method) &
                           (data["window lengths"] == wl) & (data["max. threads"] == 10)]["time"].mean() / 1_000_000
            print(cmp_val, cmp_val2)
            for thr in thread_numbers:
                computation_times["method"].append(method)
                computation_times["thread number"].append(thr)
                computation_times["computation time [ms]"].append(data[
                                                                      (data['method'] == method)
                                                                      & (data["window lengths"] == wl)
                                                                      & (data["max. threads"] == thr)
                                                                      ]
                                                                  ["time"].mean() / 1_000_000 / cmp_val)
        computation_times = pd.DataFrame(computation_times)
        plot = sns.barplot(data=computation_times, x="thread number", y="computation time [ms]", hue="method")
        plt.show()


if __name__ == "__main__":
    main()
