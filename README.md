# Efficient Hankel Decomposition for Change Point Detection
![Computation](images/Changepoint_Computation_Time_simulated.png)
This repository contains the code for the paper Efficient Hankel Decomposition for Change Point Detection.
The repository is aimed at making the research and results from the paper reproducible. While we are not aware of any 
imminent restrictions, we recommend setting up Pythen 3.9 or newer (newest available interpreter is 3.11 at the time).

The code is mainly focused on the paper, for code with a better interface see below:

> If you just want to use the algorithms check out package [changepoynt](https://github.com/Lucew/changepoynt)
> where we will make the methods pip-installable and accessible.

## Data preparation
We are very thankful to Eamonn Keogh and all the contributors to the 
[UCR Time Series Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). In addition to the simulated data,
we use this archive for our empirical measurements. The directory [preprocessing](preprocessing/) contains all the
necessary to preprocess the repository. Adapt the path in the main function of 
[processUcrArchive.py](preprocessing/processUcrArchive.py) to the path where the extracted archive is located. The 
archive is public and downloadable. The resulting hdf5-file will have about 650 mB.

## Decomposition results
The paper evaluates the proposed method in two steps. First checking the decomposition and then computing the change
score in comparison to a baseline. To run the decomposition evaluation use
[decompositionComparison.py](decompositionComparison.py) with the flag `-sim True` for the real signals and `-sim False`
for the simulated signals. To keep as true to the measurements in the paper, we recommend using a conda environment
that also makes sure the optimal array processing libraries are installed. The script will make use of all cores
available, so take care when running on a shared machine.

## Score results
The second step of the paper is measuring the scoring error and computation times. To see similar performance increases
with parallelization as in the paper, your processor should have more than ten cores. The script might run multiple 
days. Run [changepointComparison.py](changepointComparison.py) with the flag `-sim True` for the real signals and 
`-sim False` for the simulated signals.

## Creating the plots
Once both results are available, the directory [evaluation](evaluation/) contains all necessary code to plot the results
similar to the paper. The naming conventions are similar to the computation scripts. Change the global variables
`SAVE_CONFIG = False` at the beginning of the plot scripts to see the plots immediately, otherwise they will be saved 
as .pgf files. `SIMULATION = False` and `SIMULATION = True` toggles between signals from the UCR archive and simulated
signals.

