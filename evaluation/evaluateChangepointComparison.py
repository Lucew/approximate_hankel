import os.path
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


SAVE_CONFIG = False
SIMULATION = True

if SAVE_CONFIG:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 22,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    matplotlib.rcParams['figure.figsize'] = 16, 9


def load_files(path="../results", simulated=True):

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


def main(simulated=True):

    # load the data into memory
    data = load_files(simulated=simulated)
    data['Error'] = data['true-score']

    # get the method names
    methods = data["method"].unique()
    data['time'] = data['hankel construction time'] + data['decomposition time']

    # get the window sizes
    window_sizes = sorted(data["window lengths"].unique())

    # go over the methods and compute the error
    for method in methods:
        print(f"Method {method} we have a mean error of: \t{data[data['method'] == method]['Error'].abs().mean()}")

    # make the methods' array and a dict to sort from it
    methods = [method for method in methods if method.endswith("rsvd") or method.endswith("svd") or method.endswith("ika") or method.endswith("naive irlb")]
    methods[0], methods[4] = methods[4], methods[0]
    methods[1], methods[2] = methods[2], methods[1]
    methods_dict = {method: idx for idx, method in enumerate(methods)}

    # sort the dataframe by the methods so the plots keep the same color for all the methods
    data = data.sort_values(by=['method'], key=lambda x: x.apply(methods_dict.get))

    # go over the methods and window sizes and compute the average computation time
    computation_times = {"method": [], "window size N": [], "computation time [ms]": []}
    for method in methods:
        print(data[data['method'] == method].shape)
        for wl in window_sizes:
            computation_times["method"].append(method)
            computation_times["window size N"].append(wl)
            computation_times["computation time [ms]"].append(data[(data['method'] == method) & (data["window lengths"] == wl) & (data["max. threads"] == 6)]["time"].mean(skipna=True)/1_000_000)

    # make into a dataframe and plot
    plt.rcParams['text.usetex'] = True
    computation_times = pd.DataFrame(computation_times)

    # Plot the computation times
    plot = sns.lineplot(data=computation_times, x="window size N", y="computation time [ms]", hue="method")

    # group the methods after the window sizes
    grouped = computation_times.groupby("window size N")
    first_overtake = float('inf')
    for window_size, group in grouped:
        resi = group.loc[group["computation time [ms]"].idxmin(), 'method']
        if resi == "fft rsvd":
            first_overtake = min(first_overtake, window_size)

    # Plot vertical lines where one method becomes faster than the other
    plt.axvline(x=first_overtake, color='black', linestyle='--', linewidth=1)
    plt.text(first_overtake+30, plt.gca().get_ylim()[1], f'{first_overtake}', rotation=0, verticalalignment='bottom')

    # Set labels and title
    plt.xlabel("Window size N")
    plt.ylabel("Computation time [ms]")
    plt.title("Computation Time Comparison")
    plot.set(yscale="log")

    complexity = {"naive svd": "O($N^{3}$)", "naive rsvd": "O($N^{2}$)", "fft rsvd": "O($N*logN$)", "naive ika": "O($N^{3}$)" , "naive irlb": "O($N^{2}$)"}
    for method in methods:
        plot.text(x=computation_times["window size N"].max() + 75, y=computation_times[computation_times['method'] == method]["computation time [ms]"].max(), s=complexity[method], va="center")
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    if SAVE_CONFIG:
        plt.savefig(f'Changepoint_Computation_Time{"_simulated" if simulated else ""}.pgf')
    else:
        plt.show()
    plt.rcParams['text.usetex'] = False

    # make into a dataframe and plot the histograms
    plot = sns.histplot(data[data['method'].str.endswith("rsvd") | data['method'].str.endswith("ika") | data['method'].str.endswith("naive irlb")], x='Error', hue="method", multiple="dodge", bins='sturges', log_scale=[False, True])
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    if SAVE_CONFIG:
        plt.savefig(f'Changepoint_Score_Histogram{"_simulated" if simulated else ""}.pgf')
    else:
        plt.show()
    plt.rcParams['text.usetex'] = False
    plot = sns.histplot(data[data['method'] == "naive svd"], x="score", hue="method", multiple="dodge", bins='sturges', log_scale=[False, True])
    if SAVE_CONFIG:
        plt.savefig(f'Changepoint_Error_Histogram{"_simulated" if simulated else ""}.pgf')
    else:
        plt.show()

    # check the scalability of the methods per number of threads
    thread_numbers = data["max. threads"].unique()
    print(window_sizes)
    if len(thread_numbers) > 1:

        # find the maximum window size that we ran
        wl = sorted(window_sizes)[-1]

        # go over the methods and window sizes and compute the average computation time
        computation_times = {"method": [], "thread number": [], "computation time [%]": []}
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
                computation_times["computation time [%]"].append(data[
                                                                      (data['method'] == method)
                                                                      & (data["window lengths"] == wl)
                                                                      & (data["max. threads"] == thr)
                                                                      ]
                                                                  ["time"].mean() / 1_000_000 / cmp_val * 100)
        computation_times = pd.DataFrame(computation_times)
        plot = sns.barplot(data=computation_times, x="thread number", y="computation time [%]", hue="method")
        if SAVE_CONFIG:
            plt.savefig(f'Changepoint_Parallelization{"_simulated" if simulated else ""}_WindowSize_{wl}.pgf')
        else:
            plt.show()


if __name__ == "__main__":
    main(SIMULATION)
