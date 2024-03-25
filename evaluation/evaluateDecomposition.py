import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

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

    # plot the distribution
    plot = sns.boxplot(eigenvalues, x='Eigenvalues', y='Participation [%]', hue=n_name, orient='v')
    plot.set_xticklabels([f"{idx}" for idx in range(len(eigcols))])
    plot.set_yscale("log")
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)
    plot.legend(loc='upper right', ncol=2, title=n_name)
    plt.show()


if __name__ == "__main__":
    main()