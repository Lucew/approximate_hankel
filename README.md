# Efficient Hankel Decomposition for Change Point Detection
![Computation](images/Changepoint_Computation_Time_simulated.png)
This repository contains the code for the paper Efficient Hankel Decomposition for Change Point Detection.
The repository aims to make the research and results from the paper reproducible. While we are not aware of any 
imminent restrictions, we recommend setting up Pythen 3.9 or newer (the newest available interpreter is 3.11 
at the time of this publication).

The code is mainly focused on the paper, for code with a better interface see below:

> If you only want to use the algorithms, check out package [changepoynt](https://github.com/Lucew/changepoynt)
> where we will make the methods pip-installable and accessible.

If you only want to use the fast Hankel matrix product, take a look at the functions `fast_numba_hankel_left_matmul`, 
`fast_numba_hankel_matmul`, and `get_fast_hankel_representation` in [fastHankel.py](utils/fastHankel.py). For the
decompositions using these functions look at `randomized_hankel_svd_fft` and `irlb_fft` in 
[changepointComparison](changepointComparison.py). 

⚠ Do not forget to set your thread limit for number or set
`parallel=False` in the decorators if you encounter performance issues or very high CPU utilization!


Our package [changepoynt](https://github.com/Lucew/changepoynt) includes matrix classes that are numpy compatible and
implement all the necessary functions.

> Consider using `scipy.sparse.linalg.LinearOperator` using our `fast_numba_hankel_left_matmul` and 
> `fast_numba_hankel_matmul` if you want to speed up your algorithms working on Hankel matrices

## Data Preparation
We are very thankful to Eamonn Keogh and all the contributors to the 
[UCR Time Series Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/). In addition to the simulated data,
we use this archive for our empirical measurements. The directory [preprocessing](preprocessing/) contains all the
necessary to preprocess the repository. Adapt the path in the main function of 
[processUcrArchive.py](preprocessing/processUcrArchive.py) to the path where the extracted archive is located. The 
archive is public and downloadable. The resulting hdf5-file will have about 650 MB.

## Decomposition Results
The paper evaluates the proposed method in two steps. First checking the decomposition and then computing the change
score in comparison to a baseline. To run the decomposition evaluation use
[decompositionComparison.py](decompositionComparison.py) with the flag `-sim True` for the real signals and `-sim False`
for the simulated signals. To keep as true to the measurements in the paper, we recommend using a conda environment
that also makes sure the optimal array processing libraries are installed. The script will use all cores available, so 
take care when running on a shared machine.

## Score Results
The second step of the paper is measuring the scoring error and computation times. To see similar performance increases
with parallelization as in the paper, your processor should have more than ten cores. The script might run for multiple 
days. Run [changepointComparison.py](changepointComparison.py) with the flag `-sim True` for the real signals and 
`-sim False` for the simulated signals.

## Creating the Plots
Once both results are available, the directory [evaluation](evaluation/) contains all the necessary code to plot the 
results similar to the paper. There are more plots available than we could fit into the paper. The main plots also 
contained in the paper are a result of [evaluateDecomposition.py](evaluation/evaluateDecomposition.py) for the overall 
performance of the decomposition and [evaluateChangepointComparison.py](evaluation/evaluateChangepointComparison.py) 
for the score error and the computation times. [evaluateScoreScatter.py](evaluation/evaluateScoreScatter.py) creates the
scatter plot comparing the ground truth scores with the approximated scores.

Change the global variables at the beginning of the plot scripts 
`SAVE_CONFIG = False` to see the plots immediately, otherwise, they will be saved as .pgf files. `SIMULATION = False` 
and `SIMULATION = True` toggles between signals from the UCR archive and simulated signals.

## Acknowledgement
Projects like this are never solely the outcome of one research paper.
Therefore, we want to thank all the countless contributors and initiators of Python and key libraries like numpy, 
scipy, matplotlib, tqdm, seaborn, numba and pandas.

Other than that, this repository contains modified code from [irlbpy](https://github.com/bwlewis/irlbpy) licensed under
the APL2.0 at the time of publishing this code.
We also took inspiration from the implementation of [SST](https://github.com/statefb/singular-spectrum-transformation)
licensed under the MIT license at the time of publishing this code.
We also modified code from [fbpca](https://github.com/facebookarchive/fbpca) licensed using the BSD licensed at the time
of publishing this code. Additionally, we want to express our gratitude to the developers of
[rocket-fft](https://github.com/styfenschaer/rocket-fft) lincensed with a BSD3 license, which enabled us to use numba
with the FFT and signicantly increase the speed of the algorithms.
[Threadpoolctl](https://github.com/joblib/threadpoolctl) with a BSD3 license was used to make measurements with
thread limitations.
