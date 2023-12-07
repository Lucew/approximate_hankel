import numpy as np
import os
import h5py
import pandas as pd
from tqdm import tqdm


def read_csv(path: str, expected_signals: int) -> list[(int, np.ndarray)]:

    # get the array from the file
    array = np.genfromtxt(path, delimiter="\t")
    assert array.shape[0] == expected_signals, f'{path} has {array.shape}/{expected_signals} expected signals'

    # the first column of every file is the class which we don't need for processing
    # see UCR archive description
    array = array[:, 1:]

    # turn the array into a list of rows as signals are the rows of the array
    # and also take care of variable length
    #
    # don't sort out nan by using boolean indexing as we want to make sure to
    # only cut the end of signals. We do not want to cut nan values in between
    # as they would hint for faulty parsing or faulty signals
    arrays = [array[idx, :] for idx in range(array.shape[0])]
    inclusion = []
    with open('excluded_signals_missing_values.txt', 'a') as filet:
        for idx, array in enumerate(arrays):

            # check our assumption that the array is no 1d
            assert array.ndim == 1, f'Signal in {path} and row {idx} is not 1D and has shape {array.shape}.'

            # check how many values we need to cut
            ldx = array.shape[0]-1
            while ldx and np.isnan(array[ldx]):
                ldx -= 1

            # check that not the whole array was nan
            assert ldx >= 0, f'Signal in {path} has only NaN values in row {idx}.'

            # save the pruned array
            arrays[idx] = array[:ldx+1]

            # check for missing values and write to file
            if np.any(np.isnan(arrays[idx])):
                filet.write(f'Signal in {path} has missing NaN values in row {idx}.\n')
            else:
                inclusion.append(idx)

    return [(idx, arrays[idx]) for idx in inclusion]


def find_files(path: str) -> list[str]:
    # specify the unwanted folder
    unwanted = "Missing_value_and_variable_length_datasets_adjusted"

    # get all the tsv files and get rid of the interpolated and extended signals
    files = [os.path.join(root, file) for root, _, files in os.walk(path)
             for file in files if file.endswith('.tsv') and unwanted not in root]
    return files


def create_hdf_file(path: str) -> None:

    # find the files we need to parse
    files = find_files(path)

    # load the info csv for all the files
    info_csv = pd.read_csv("DataSummary.csv", index_col=2)
    info_csv.columns = [col.strip() for col in info_csv.columns]
    available_signals = info_csv["Test"].sum() + info_csv["Train"].sum()
    print(f'We have {available_signals} signals in the dataset.')

    # go through the files, read the content and put into hdf5 file
    saved_signal_counter = 0
    with h5py.File(f'{os.path.split(path)[-1]}.hdf5', 'w') as filet:
        for file in tqdm(files, desc='Processing files'):

            # get the content
            dataset_name = os.path.splitext(os.path.split(file)[-1])[0]
            dataset_name, specifier = dataset_name.split("_")
            contents = read_csv(file, info_csv[specifier.capitalize()].loc[dataset_name])

            # save into the hdf5 file
            for sx, signal in contents:
                # create the names
                name = f'{os.path.splitext(os.path.split(file)[-1])[0]}_signal_{sx}'

                # create the dataset
                filet.create_dataset(name=name, data=signal)
                saved_signal_counter += 1
    print(f"We included {saved_signal_counter}/{available_signals} and "
          f"excluded {available_signals-saved_signal_counter}")


def main():
    create_hdf_file(os.path.join('..', '..', 'Data', 'UCRArchive_2018'))


if __name__ == '__main__':
    main()
