import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm

from evaluateChangepointComparison import load_files

SAVE_CONFIG = False

if SAVE_CONFIG:
    print('Saving!')
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 22,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


def print_latex_table(df: pd.DataFrame):
    # create the header we want to have
    file_lines = [f"\\begin{{tabular}}{{{'l'*(len(df.columns)+1)}}}", "\t\\toprule"]

    # create the header of the table
    header = ["\t\\textbf{Threads}"]
    for col in df.columns:
        numerator, denominator = col.split(" - ")
        header.append(f"$\\frac{{\\text{{{numerator.upper()}}}}}{{\\text{{{denominator.upper()}}}}}$")
    header = " & ".join(header)
    file_lines.append(header + "\\\\")
    file_lines.append("\t\\midrule")

    # go through all rows of the dataframe
    firsttup = [0]*len(df.columns)
    for row in df.itertuples():
        row = list(row)

        # save the first values
        if row[0] == 1:
            firsttup = row

        line = [f"\t{row[0]}"]
        for idx, ele in enumerate(row[1:], 1):
            line.append(f"\pcb{{{ele:0.2f}}}{{{firsttup[idx]:0.2f}}}{{}}")
        line = " & ".join(line)
        file_lines.append(line + "\\\\")

    # make the end line and close the table
    file_lines.append("\t\\bottomrule")
    file_lines.append("\\end{tabular}")

    # print table to the console
    for file in file_lines:
        print(file)

def nlogn_speed_factors(datagroup: pd.DataFrame):

    # only keep methods that have the same scaling
    datagroup = datagroup[[cl for cl in datagroup.columns if cl.startswith('fft')]]

    # make the threads to a column again
    datagroup = datagroup.reset_index(1)

    # compute the speed factor by dividing the last two columns by the first
    datagroup['fft rsvd - fft ika'] = datagroup['fft rsvd'] / datagroup['fft ika']
    datagroup['fft irlb - fft ika'] = datagroup['fft irlb'] / datagroup['fft ika']
    datagroup['fft irlb - fft rsvd'] = datagroup['fft irlb'] / datagroup['fft rsvd']

    # only keep the speed up
    datagroup = datagroup[['max. threads', 'fft rsvd - fft ika', 'fft irlb - fft ika', 'fft irlb - fft rsvd']]

    # group by the threadlimits to compute mean speed up
    datagroup = datagroup.groupby(['max. threads']).mean()
    print_latex_table(datagroup)

def efficiency_comparisons(datagroup: pd.DataFrame):

    # keep only the max threadlimit
    datagroup = datagroup.reset_index()
    datagroup = datagroup[datagroup['max. threads'] == 10]
    datagroup = datagroup.drop(columns=['max. threads'])

    # keep only a few window sizes
    window_sizes = {100, 197, 295, 505, 991, 1945, 5000}
    datagroup = datagroup.loc[datagroup['window lengths'].isin(window_sizes)]
    datagroup = datagroup.set_index('window lengths')
    datagroup.index.name = 'Window Size $N$'

    # make the comparisons
    compared_cols = []
    compared_svd_cols = []
    for col in datagroup.columns:
        for col2 in datagroup.columns:
            namespl = col.split(' ')
            namespl2 = col2.split(' ')
            # (namespl[-1] == namespl2[-1] or col2 == 'naive svd')
            if (col2 == 'naive ika' or col2 == 'naive svd') and namespl[0] == 'fft' and namespl2[0] == 'naive':
                if col2 != 'naive svd':
                    new_name = f'{col.upper()}'
                    compared_cols.append(new_name)
                else:
                    new_name = f'{col.upper()} SVD'
                    compared_svd_cols.append(new_name)
                datagroup[new_name] = datagroup[col2] / datagroup[col]
    for col in compared_cols:
        print(datagroup[col])

    vmax = 10**np.floor(np.log10(datagroup.max().max()))

    # avoid 0 with LogNorm; pick something small if your data can hit 0
    datagroup = datagroup.where(datagroup <= 10, datagroup.round(0))
    norm = LogNorm(vmin=0.1, vmax=vmax)

    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(16, 9))
    fig.tight_layout()

    # get the data
    datagroup1 = datagroup[compared_svd_cols].rename(columns={col: col[:-4] for col in compared_svd_cols})


    h0 = sns.heatmap(
        datagroup1.T, annot=labels_round_big(datagroup1.T), norm = norm, ax = axs[0], cbar = False,
        fmt="", annot_kws={"fontsize": 30},
    )
    axs[0].get_xaxis().set_visible(False)

    h1 = sns.heatmap(
        datagroup[compared_cols].T, annot=labels_round_big(datagroup[compared_cols].T), norm=norm, ax=axs[1], cbar=False,
        fmt="", annot_kws={"fontsize": 30},
    )

    # set the axis label
    axs[0].set_title('Baseline: complete SVD', fontsize=35)
    axs[1].set_title('Baseline: naive matrix product (no FFT)', fontsize=35)
    axs[1].set_xlabel('Window Size $N$', fontsize=30)

    # update the labels of the second plot
    # labels = [item.get_text() for item in axs[1].get_xticklabels()]
    # axs[1].set_xticklabels(labels)

    # single colorbar spanning both subplots
    mappable = h0.collections[0]  # both share the same norm/cmap
    fig.colorbar(mappable, ax=axs, orientation='vertical', fraction=0.05, pad=0.02)

    axs[1].tick_params("y", rotation=0, labelsize=30)
    axs[0].tick_params("y", rotation=0, labelsize=30)
    if SAVE_CONFIG:
        plt.savefig(f'SpeedFactors.pgf', bbox_inches='tight')
    else:
        plt.show()

def labels_round_big(df):
    vals = df.to_numpy()
    lab = np.empty(vals.shape, dtype=object)
    mask = vals > 10
    lab[mask] = np.round(vals[mask]).astype(int).astype(str)  # 123
    lab[~mask] = np.vectorize(lambda x: f"{x:.2f}")(vals[~mask])  # 9.87, 0.12, etc.
    return lab

def main():

    # load the data into memory
    data = load_files(simulated=True)
    data['time'] = data['hankel construction time'] + data['decomposition time']

    # group by the method, the window size and the thread limit and compute mean computation time in ms
    datagroup = data.groupby(['method', 'window lengths', 'max. threads'])['time'].mean()/1_000_000

    # unstack the multiindex
    datagroup = datagroup.unstack('method')

    # make the nlogn comparisons
    nlogn_speed_factors(datagroup)

    # make the efficiency comparisons
    efficiency_comparisons(datagroup)


if __name__ == "__main__":
    main()
