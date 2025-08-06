import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from evaluation.evaluateChangepointComparison import load_files

SAVE_CONFIG = False
SIMULATION = True

if SAVE_CONFIG:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 30,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    matplotlib.rcParams['figure.figsize'] = 16, 9

def main(simulated=True):

    # load the data into memory
    data = load_files(simulated=simulated)

    # compute the true score for all methods
    bin_number = 10
    bins = np.linspace(0, 1, bin_number+1)
    bin_width = bins[1]-bins[0]
    data['True Score'] = pd.cut(data['true-score'] + data['score'], bins=bins)
    data['Approx. Score'] = data['score']

    # get the method names
    methods = list(data["method"].unique())
    data['time'] = data['hankel construction time'] + data['decomposition time']
    data[["Mat. Mult.", "norm. Method"]] = data["method"].str.split(" ", expand=True)
    data["Method"] = data["method"]

    # make the methods' array and a dict to sort from it
    # methods[0], methods[-1] = methods[-1], methods[0]
    # methods[1], methods[2] = methods[2], methods[1]
    methods.sort(key=lambda x: (x.split(' ')[1], tuple(-ord(ele) for ele in x.split(' ')[0])))
    methods_dict = {method: idx for idx, method in enumerate(methods)}

    # sort the dataframe by the methods so the plots keep the same color for all the methods
    data = data.sort_values(by=['method'], key=lambda x: x.apply(methods_dict.get))

    # make the plots
    fig, ax = plt.subplots()
    sns.boxplot(data[data['norm. Method'] != 'svd'], x='True Score', y='Approx. Score', hue='Method', showfliers=False)
    plt.plot([0, bin_number-1], [0, 1], linestyle='--', color='black', zorder=-1)
    labels = [f"{float(item.get_text()[1:-1].split(', ')[1]):0.1f}" for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # between the other ticks
    # https://stackoverflow.com/a/24953575
    minor_ticks = ax.get_xticks()
    ax.set_xticks(np.array(minor_ticks[1:])-0.5, minor=True)
    ax.grid(axis='x', which='minor', linestyle='-', alpha=1.0, zorder=-2)
    ax.tick_params(axis='x', which='minor', length=0)
    if SAVE_CONFIG:
        plt.savefig(f'Scoring_Error_Scatter{"_simulated" if simulated else ""}.pgf')
    else:
        plt.show()


if __name__ == '__main__':
    main(SIMULATION)