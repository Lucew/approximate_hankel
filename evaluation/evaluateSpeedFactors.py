import evaluateChangepointComparison as ecc
import pandas as pd


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

def main():

    # load the data into memory
    data = ecc.load_files(simulated=True)
    data['time'] = data['hankel construction time'] + data['decomposition time']

    # group by the method, the window size and the thread limit and compute mean computation time in ms
    datagroup = data.groupby(['method', 'window lengths', 'max. threads'])['time'].mean()/1_000_000

    # unstack the multiindex
    datagroup = datagroup.unstack('method')

    # only keep methods that have the same scaling
    datagroup = datagroup[[cl for cl in datagroup.columns if cl.startswith('fft')]]

    # make the threads to a column again
    datagroup = datagroup.reset_index(1)

    # compute the speed factor by dividing the last two columns by the first
    datagroup['fft rsvd - fft ika'] = datagroup['fft rsvd']/datagroup['fft ika']
    datagroup['fft irlb - fft ika'] = datagroup['fft irlb']/datagroup['fft ika']
    datagroup['fft irlb - fft rsvd'] = datagroup['fft irlb']/datagroup['fft rsvd']

    # only keep the speed up
    datagroup = datagroup[['max. threads', 'fft rsvd - fft ika', 'fft irlb - fft ika', 'fft irlb - fft rsvd']]

    # group by the threadlimits to compute mean speed up
    datagroup = datagroup.groupby(['max. threads']).mean()
    print_latex_table(datagroup)


if __name__ == "__main__":
    main()
