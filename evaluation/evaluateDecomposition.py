import os

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import matplotlib
import warnings

from scipy.ndimage import rotate

# if this is True, the figure is saved as pgf
# if this is False, the figure is plotted
SAVE_CONFIG = False
SIMULATION = True

if SAVE_CONFIG:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
rcParams['figure.figsize'] = 12, 6.75


def bic_criterion(input_array, method='BIC'):
    if method == 'BIC':
        # only the last one is the window size
        wl = input_array[:, -1]
        other = input_array[:, :-1]

        # make the repeated arrays we need
        wl = np.repeat(wl[:, None], other.shape[1], axis=1)
        d_array = np.repeat(np.arange(1, other.shape[1] + 1)[None, :], other.shape[0], axis=0)

        return np.argmin(wl * np.log(other / wl) + 2 * d_array * np.log(wl), axis=1)
    elif method == 'AIC':
        # only the last one is the window size
        wl = input_array[:, -1]
        other = input_array[:, :-1]

        # make the repeated arrays we need
        wl = np.repeat(wl[:, None], other.shape[1], axis=1)
        d_array = np.repeat(np.arange(1, other.shape[1] + 1)[None, :], other.shape[0], axis=0)

        return np.argmin(wl * np.log(other / wl) + 4 * d_array, axis=1)
    else:
        raise ValueError(f'Method {method} not implemented.')


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


def main(simulated=True):

    # load the data into memory
    data = load_files(simulated=simulated)
    print(f'We have data for {data.shape[0]} different Hankel matrices.')

    # get the window sizes
    n_name = 'Matrix Dim. NxN'
    window_sizes = sorted(data[n_name].unique())
    for ws in window_sizes:
        print(f'For window size {ws} we have: {data[data[n_name] == ws].shape[0]} matrices.')

    # get all the eigenvalues for one window size into an array
    eigcols = [col for col in data.columns if col.startswith("eigenvalue") and not col.endswith("sum")]

    # compute the participation of each eigenvalue
    if 'eigenvalue sum' in data:
        summed = data['eigenvalue sum']
    else:
        warnings.warn('There is no eigenvalue sum in the dataset, constructing one.')
        summed = data[eigcols].sum(axis=1)

    # get the reconstruction columns
    for target_p in [0, 5]:
        svd_recon_cols = [col for col in data.columns if col.startswith("reconstruction svd")]
        rsvd_recon_cols = [col for col in data.columns if col.startswith("reconstruction rsvd naive") and f'p={target_p}' in col]
        fft_rsvd_recon_cols = [col for col in data.columns if col.startswith("reconstruction rsvd fft") and f'p={target_p}' in col]

        # go over different subspace iterations and calculate the difference to the complete reconstruction
        for q in range(4):
            term = f'q={q}'
            tmp = [col for col in rsvd_recon_cols if term in col]
            # data[tmp] = (data[tmp].to_numpy()-data[svd_recon_cols].to_numpy())/((data[n_name].to_numpy()**2)[:, None]*summed.to_numpy()[:, None])
            data[tmp] = np.abs(data[tmp].to_numpy()/data[svd_recon_cols].to_numpy()-1)
            tmp = [col for col in fft_rsvd_recon_cols if term in col]
            # data[tmp] = (data[tmp].to_numpy()-data[svd_recon_cols].to_numpy())/((data[n_name].to_numpy()**2)[:, None]*summed.to_numpy()[:, None])
            data[tmp] = np.abs(data[tmp].to_numpy()/data[svd_recon_cols].to_numpy()-1)

        # melt the data into format to use with seaborn
        recon_cols = rsvd_recon_cols + fft_rsvd_recon_cols
        grouped_data = data[recon_cols + [n_name]].groupby(n_name).mean()
        # reconstruction_df = grouped_data.melt(value_vars=recon_cols, value_name=r'Normalized Frobenius Norm', var_name='Method')
        reconstruction_df = grouped_data.melt(value_vars=recon_cols, value_name=r'Normalized Error', var_name='Method')
        reconstruction_df['Reconstruction Rank'] = reconstruction_df['Method'].apply(lambda x: x.split(' ')[-1].replace('rank=', ''))
        reconstruction_df['Parameter'] = reconstruction_df['Method'].apply(lambda x: x.split(" ")[3][:-1])
        reconstruction_df['Method'] = reconstruction_df['Method'].apply(lambda x: " ".join(x.split(' ')[1:3]))

        # take care of numeric abnormalities
        reconstruction_df.loc[reconstruction_df[r'Normalized Error'] <= np.finfo(np.float64).eps, r'Normalized Error'] = 1e-7

        # plot the reconstruction error
        recon_plot = sns.lineplot(reconstruction_df, x='Reconstruction Rank', y=r'Normalized Error', hue='Parameter', errorbar=None, style='Method', markers=True, markersize=10, estimator=np.median)
        # recon_plot_small = sns.lineplot(reconstruction_df[reconstruction_df["Parameter"] != "q=0"], x='Reconstruction Rank', y=r'Normalized Frobenius Norm', hue='Parameter', errorbar="sd", style='Method', ax=ax)
        # recon_plot = sns.barplot(reconstruction_df, x='Reconstruction Rank', y=r'Normalized Frobenius Norm', hue='Parameter', errorbar="sd")

        # make groups to draw some points
        reconstruction_df_grouped = reconstruction_df[['Reconstruction Rank', 'Normalized Error', 'Parameter', 'Method']].groupby(['Reconstruction Rank', 'Parameter', 'Method'])

        # add some dummy plots so the legend cols are good
        recon_plot.plot([np.NaN, np.NaN], recon_plot.get_xlim(), color='w', alpha=0, label=' ')
        recon_plot.plot([np.NaN, np.NaN], recon_plot.get_xlim(), color='w', alpha=0, label=' ')

        # go through the points to draw text
        for (rank, parameter, method), val in reconstruction_df_grouped:
            # ignore the fft methods
            if 'fft' not in method or int(rank) & 1: continue

            # compute an angle
            angle = 0
            if rank != "20":
                # get value before and after
                val_before = reconstruction_df_grouped.get_group((f"{int(rank) - 1}", parameter, method))[
                    'Normalized Error'].median()
                val_before = max(val_before, reconstruction_df_grouped.get_group((f"{int(rank) - 1}", parameter, method.replace('naive', 'fft')))[
                    'Normalized Error'].median())
                val_after = reconstruction_df_grouped.get_group((f"{int(rank) + 1}", parameter, method))[
                    'Normalized Error'].median()
                val_after = max(val_after, reconstruction_df_grouped.get_group(
                    (f"{int(rank) + 1}", parameter, method.replace('naive', 'fft')))[
                    'Normalized Error'].median())

                # the angle is the tan(alpha) = dy/dx
                dy = abs(np.log(val_after) - np.log(val_before))
                dx = 2

                # get the angle
                angle = np.rad2deg(np.arctan(dy / dx))

            # get the actual value
            val = max(val['Normalized Error'].median(), reconstruction_df_grouped.get_group((rank, parameter, method.replace('naive', 'fft')))['Normalized Error'].median())
            if val < 0.0001: continue

            # get the value in percent with two digits
            perc_val = f"{val * 100:.2f}%"

            # add the text to the plot
            if parameter == 'q=3':
                plt.text(x=rank, y=val/1.8, s=perc_val, ha='center', va='center', fontsize=20, rotation=angle)
            else:
                plt.text(x=rank, y=val*1.4, s=perc_val, ha='center', va='center', fontsize=20, rotation=angle)

        # some formatting for the plot
        recon_plot.set_yscale('log')
        recon_plot.spines["top"].set_visible(False)
        recon_plot.spines["right"].set_visible(False)
        recon_plot.set_ylim([1e-6, 1.5])
        recon_plot.legend(loc="lower right", ncols=2, title=f'Oversampling p={target_p}')
        recon_plot.set_xlabel('Target Rank $k$')
        plt.setp(recon_plot.get_legend().get_texts(), fontsize='22')
        plt.setp(recon_plot.get_legend().get_title(), fontsize='24')
        plt.grid(axis='both', linestyle='-', alpha=0.8)
        recon_plot.xaxis.set_tick_params(labelsize=20)
        recon_plot.yaxis.set_tick_params(labelsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.gca().set_xlabel(plt.gca().get_xlabel(), fontsize=22)
        plt.gca().set_ylabel(plt.gca().get_ylabel(), fontsize=22)
        plt.tight_layout()
        if SAVE_CONFIG:
            plt.savefig(f'Reconstruction{"_simulated" if simulated else ""}_p_{target_p}.pgf')
        else:
            plt.show()
        plt.gcf().clear()

    # compute relative eigenvalues
    data[eigcols] = data[eigcols].div(summed, axis=0)*100
    eigenvalues = data.melt(id_vars=n_name, value_vars=eigcols, value_name='Participation [%]', var_name='Eigenvalues')

    # compute the threshold defined by Gavis
    if 'median eigenvalue' in data:
        # compute the threshold for each row in the dataframe (in percentage)
        threshold = data['median eigenvalue'] / data['eigenvalue sum'] * 2.858 * 100
    else:
        warnings.warn('There is no median eigenvalue sum in the dataset, constructing one.')
        threshold = data[eigcols].median(axis=1)*2.858

    # check how many eigenvalues are above the threshold
    above = data[eigcols].to_numpy()
    thresh = np.repeat(threshold.values[:, None], above.shape[1], axis=1)
    gavis_threshold = (above >= thresh).sum(axis=1)
    above = np.mean(gavis_threshold)
    estimated_rank_gavis = np.median(above)
    threshold = np.mean(threshold)

    # compute the threshold as proposed by Golyndina2020
    reconstruction_cols = [col for col in data.columns if col.startswith("reconstruction svd")]
    other_cols = [n_name]
    aic_threshold = bic_criterion(data[reconstruction_cols+other_cols].to_numpy(), method='AIC')
    estimated_rank_aic = np.mean(aic_threshold)
    bic_threshold = bic_criterion(data[reconstruction_cols+other_cols].to_numpy(), method='BIC')
    estimated_rank_bic = np.mean(bic_threshold)
    print(f'Estimated threshold from BIC criterium is [{estimated_rank_bic}] and from AIC [{estimated_rank_aic}], '
          f'for Gavis it is [{estimated_rank_gavis}].')

    # make the color palette
    palette = sns.color_palette()
    fig, ax = plt.subplots()
    ax.grid(axis='y', linestyle='-', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # plot the distribution of eigenvalues for every matrix size
    plot = sns.boxplot(eigenvalues, x='Eigenvalues', y='Participation [%]', ax=ax, hue=n_name, orient='v',
                       palette=palette, zorder=20, showfliers=False)
    ax.set_xlabel('Singular values')

    # check the singular values and their respective singular value gap
    check_participation_threshold(eigenvalues)

    # plot the magical noise threshold
    plt.axhspan(plot.get_ylim()[0], threshold, facecolor='0.4', alpha=0.35, zorder=0)
    # plot.spines["top"].set_visible(False)
    # plot.spines["right"].set_visible(False)

    # make a second axis into the plot with the summation
    ax2 = plot.twinx()
    data[eigcols] = data[eigcols].cumsum(axis=1)
    cumsum = data[eigcols + [n_name]].groupby(n_name, ).mean()
    cumsum[n_name] = cumsum.index
    cumsum = cumsum.melt(id_vars=n_name, value_vars=eigcols, value_name='Cumulative Participation [%]', var_name='Eigenvalues')
    plot2 = sns.lineplot(cumsum, x='Eigenvalues', y='Cumulative Participation [%]', hue=n_name, ax=ax2, palette=palette,
                         zorder=-10, errorbar=None, linestyle='--', legend=False)

    # make some formatting
    # https://discourse.matplotlib.org/t/control-twinx-series-zorder-ax2-series-behind-ax1-series-or-place-ax2-on-left-ax1-on-right/15105/2
    plot.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
    plot.patch.set_visible(False)  # hide the 'canvas'
    plot.spines["top"].set_visible(False)
    plot2.spines["top"].set_visible(False)
    plot2.set_ylim([0, 100])
    plot.set_ylim([10e-3, 200])
    plot.legend(loc='upper right', ncol=2, title=n_name)
    plot.set_xticklabels([f"{idx}" for idx in range(len(eigcols))])
    plot.set_yscale("log")
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), fontsize=22)
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=22)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=22)
    plt.setp(plot.get_legend().get_texts(), fontsize='18')
    plt.setp(plot.get_legend().get_title(), fontsize='20')
    plt.tight_layout()
    if SAVE_CONFIG:
        plt.savefig(f'EigenvalueSpectrum{"_simulated" if simulated else ""}.pgf')
    else:
        plt.show()

    # make a new plot to check the reconstruction error overall
    fig = plt.figure()

    # get the reconstruction error for only the svd
    print(data.columns)
    groundtruth_df = data[[col for col in data.columns if not ('naive' in col or 'fft' in col)]]
    reconcols = [col for col in groundtruth_df if col.startswith('reconstruction')]

    eigenvalues = data.melt(id_vars=n_name, value_vars=reconcols, value_name='Reconstruction Error',
                            var_name='Rank')
    eigenvalues.loc[:, 'Rank'] = eigenvalues.loc[:, 'Rank'].apply(lambda x: x.split(' ')[-1])
    plot = sns.boxplot(eigenvalues, x='Rank', y='Reconstruction Error', hue=n_name, orient='v',
                       palette=palette, zorder=20, showfliers=False)

    # make violin plots from the thresholds
    fig, ax = plt.subplots()
    ax.grid(axis='y', linestyle='-', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    plot3 = sns.violinplot(pd.DataFrame({'BIC': bic_threshold, 'AIC': aic_threshold, 'Noise estimation': gavis_threshold}))
    plot3.spines["top"].set_visible(False)
    plot3.spines["right"].set_visible(False)
    plt.tight_layout()
    if SAVE_CONFIG:
        plt.savefig(f'Thresholds{"_simulated" if simulated else ""}.pgf')
    else:
        plt.show()


def check_participation_threshold(eigenvalues: pd.DataFrame):

    # create a new figure
    # fig, ax = plt.subplots()

    # get the median value per group
    eigenvalues_grouped_std = eigenvalues.groupby(['Matrix Dim. NxN', 'Eigenvalues']).std()
    # eigenvalues_grouped_std.reset_index(inplace=True)
    eigenvalues_grouped_mean = eigenvalues.groupby(['Matrix Dim. NxN', 'Eigenvalues']).mean()
    # eigenvalues_grouped.reset_index(inplace=True)
    # eigenvalues_grouped = eigenvalues_grouped.sort_values(by='Eigenvalues', key=lambda x: x.str.replace('eigenvalue ', '').astype(int))

    # merge the dataframes
    eigenvalue_participation_info = eigenvalues_grouped_mean.join(eigenvalues_grouped_std, lsuffix='_mean', rsuffix='_std')
    eigenvalue_participation_info.reset_index(inplace=True)
    eigenvalue_participation_info['Eigenvalue number'] = eigenvalue_participation_info['Eigenvalues'].str.replace('eigenvalue ', '').astype(int)

    # get the minimal number where the mean plus std is below 5% (zero indexed singular values, so add one to the result)
    minimal_singular_value_index = eigenvalue_participation_info[eigenvalue_participation_info['Participation [%]_mean'] + eigenvalue_participation_info['Participation [%]_std'] <= 5.0]['Eigenvalue number'].min()
    minimal_singular_value_index_mean = eigenvalue_participation_info[eigenvalue_participation_info['Participation [%]_mean']  <= 5.0]['Eigenvalue number'].min()
    print('The first singular value, where the mean participation + 1 standard deviation is below 5%:', minimal_singular_value_index+1)
    print('The first singular value, where the mean participation is below 5%:', minimal_singular_value_index_mean+1)

if __name__ == "__main__":
    main(simulated=SIMULATION)
