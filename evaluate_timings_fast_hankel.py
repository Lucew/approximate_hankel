import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams['figure.figsize'] = (16, 9)

# read the information into memory
df = pd.read_csv("Results_HankelMult.csv", index_col=0)

# get values and check some assumptions
runs = df["Runs"].unique()
assert len(runs) == 1, f"There are different run numbers: {runs}"
runs = runs[0]

machine_precision = df["Machine Precision"].unique()
assert len(machine_precision) == 1, f"There are different run numbers: {machine_precision}"
machine_precision = machine_precision[0]

# get only the runs with one thread
threads = df["Thread Count"].unique()
threads = [np.min(threads), np.max(threads)]
threads[1] = 10
df = df[(df["Thread Count"] == threads[0]) | (df["Thread Count"] == threads[1])]

# define the scaling for the matrices and only keep those values
scaling = 1
df = df[(df["Signal Scaling"] == 1) & (df["Other Matrix Scaling"] == 1)]

# define the other matrix dimension
other_dim = 10
possible_dim = df["Other Matrix Dim."].unique()
assert other_dim in possible_dim, f"We did not measure for other matrix dim. {other_dim}, only for {possible_dim}."
df = df[df["Other Matrix Dim."] == other_dim]

# specify the lag
lag = 1
possible_lag = df["Lag"].unique()
assert lag in possible_lag, f"We did not measure for lag {lag}, only for {possible_lag}."
df = df[df["Lag"] == lag]

# get the possible window lengths and number of windows
window_lengths = df["Window Length"].unique()
window_numbers = df["Window Number"].unique()
df = df[df["Window Length"] == df["Window Number"]]
# "FFT Execution Time (over all Runs) Right Product Hankel@Other"
names = ["Naive Execution Time (over all Runs) Right Product Hankel@Other",
         "Parallel FFT Execution Time (over all Runs) Right Product Hankel@Other",
         "FFT Execution Time (over all Runs) Right Product Hankel@Other"]
for name in names:
    for tc in df["Thread Count"].unique():
        label = f"{name.split(' ')[0]} {name.split(' ')[-1]} with {tc} Threads."
        x = df[df["Thread Count"] == tc]["Window Length"]
        y = df[df["Thread Count"] == tc][name]/runs/1_000_000
        plt.plot(x, y, "x-", label=label)

# find the point where the multithreaded matrix multiplication is surpassed by the fft multithreaded
df = df.sort_values("Window Length")

smaller = np.where(df[df["Thread Count"] == threads[1]][names[0]].values >=
                   df[df["Thread Count"] == threads[1]][names[1]].values)
wd = None
if len(smaller[0]) > 0:
    idx = np.min(smaller)
    wd = df[df["Thread Count"] == threads[1]]["Window Length"].values[idx]
    ymin, ymax = plt.ylim()
    plt.vlines(wd, ymin, ymax)

# make some plot stuff
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Matrix Dimension N")
plt.ylabel("Runtime [ms]")
plt.title(f"Fast NxN Hankel matrix product with Nx{other_dim} random matrix. Parallel FFT Faster at: {wd}.")
plt.legend()
plt.show()

# find some values
for col in names:
    interest = df[df["Thread Count"] == threads[1]][[col, "Window Length"]]
    interest[col] = interest[col]/runs/1_000_000
    print(interest)
