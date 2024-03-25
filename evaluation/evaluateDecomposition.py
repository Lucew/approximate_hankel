import os

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import tikzplotlib
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 22,
    'text.usetex': True,
    'pgf.rcfonts': False,
})
rcParams['figure.figsize'] = 16, 9


def load_files(path="../results", simulated=True):

    # check whether the file exists
    combined_name = f"Decomposition{'_simulated' if simulated else ''}.parquet"
    if os.path.isfile(combined_name):
        data = pd.read_parquet(combined_name)
    else:
        # get all the csv files in the directory
        files = glob(os.path.join(path, f"Decomposition{'_simulated' if simulated else ''}_Results_WindowSize_*.csv"))

        # load the files into memory
        files = {file: pd.read_csv(file, header=0) for file in files}

        # attach a column with the window length to each dataframe
        for filename, df in files.items():

            # get the window length from the name
            filename = int(os.path.splitext(os.path.split(filename)[-1])[0].split('_')[-1])
            df['Matrix Dim. NxN'] = filename

        # concatenate the files into one
        data = pd.concat(files.values(), ignore_index=True)

        # save the file into pandas parquet format
        data.to_parquet(combined_name)
    return data


def main():

    # load the data into memory
    data = load_files(simulated=True)

    # get the window sizes
    n_name = 'Matrix Dim. NxN'
    window_sizes = sorted(data[n_name].unique())

    # get all the eigenvalues for one window size into an array
    eigcols = [col for col in data.columns if col.startswith("eigenvalue")]

    # compute the participation of each eigenvalue
    data[eigcols] = data[eigcols].div(data[eigcols].sum(axis=1), axis=0)
    eigenvalues = data.melt(id_vars=n_name, value_vars=eigcols, value_name='Participation [%]', var_name='Eigenvalues')

    # get the median eigenvalue from the eigenvalues
    threshold = np.median(data[eigcols].median(axis=1)*2.858)

    # make the color palette
    palette = sns.color_palette()
    fig, ax = plt.subplots()

    # plot the distribution of eigenvalues for every matrix size
    plot = sns.boxplot(eigenvalues, x='Eigenvalues', y='Participation [%]', ax=ax, hue=n_name, orient='v',
                       palette=palette, zorder=10)

    # plot the magical noise threshold
    plt.axhspan(plot.get_ylim()[0], threshold, facecolor='0.4', alpha=0.35, zorder=0)
    # plot.spines["top"].set_visible(False)
    # plot.spines["right"].set_visible(False)

    # make a second axis into the plot with the summation
    ax2 = plot.twinx()
    data[eigcols] = data[eigcols].cumsum(axis=1)*100
    cumsum = data.melt(id_vars=n_name, value_vars=eigcols, value_name='Cumulative Participation [%]', var_name='Eigenvalues')
    plot2 = sns.lineplot(cumsum, x='Eigenvalues', y='Cumulative Participation [%]', hue=n_name, ax=ax2, palette=palette,
                         zorder=-10, errorbar=None, linestyle='--', legend=False)

    # make some formatting
    # https://discourse.matplotlib.org/t/control-twinx-series-zorder-ax2-series-behind-ax1-series-or-place-ax2-on-left-ax1-on-right/15105/2
    plot.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
    plot.patch.set_visible(False)  # hide the 'canvas'
    plot.spines["top"].set_visible(False)
    plot2.spines["top"].set_visible(False)
    plot2.set_ylim([0, 100])
    plot.legend(loc='upper right', ncol=2, title=n_name)
    plot.set_xticklabels([f"{idx}" for idx in range(len(eigcols))])
    plot.set_yscale("log")
    # plt.show()
    plt.tight_layout()
    plt.savefig('histogram.pgf')


if __name__ == "__main__":
    main()