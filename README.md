# Task-Agnostic Morphology Optimization

This repository contains code for the paper [Task-Agnostic Morphology Evolution](https://openreview.net/pdf?id=CGQ6ENUMX6) by Donald (Joey) Hejna, Pieter Abbeel, and Lerrel Pinto. Currently, the publication is under review at ICLR 2021.

The code has been cleaned up to make it easier to use. An older version of the code was made available with the ICLR submission [here](https://openreview.net/attachment?id=CGQ6ENUMX6&name=supplementary_material). 

## Setup
The code was tested and used on Ubuntu 20.04. Our baseline implementations use `taskset`, an ubuntu program for setting CPU affinity. You *need* taskset to run some of the experiments, and the code will fail without it.

Install the conda environment using the provided file via the command `conda env create -f environment.yml`. Given this project involves only state based RL, the environment does not install CUDA and the code is setup to use CPU. Activate the environment with `conda activate morph_opt`.

Next, make sure to install the `optimal_agents` package by running `pip install -e .` from the github directory. This will use the `setup.py` file.

The code is built on top of Stable Baselines 3, Pytorch, and Pytorch Geometric. The exact specified version of stable baselines 3 is required. 

## Running Experiments
Currently, configs for the 2D experiments have been pushed to the repo. I'm working on pushing more config files that form the basis for the experiments run. To run large scale experiments for the publication, we used additional AWS tools. 

Evolution experiments can be run using the `train_ea.py` script found in the `scripts` directory. Below are example commands for running different morphology evolution algorithms:

```
python scripts/train_ea.py -p configs/locomotion2d/2d_tame.yaml

python scripts/train_ea.py -p configs/locomotion2d/2d_tamr.yaml

python scripts/train_ea.py -p configs/locomotion2d/2d_nge_no_pruning.yaml

python scripts/train_ea.py -p configs/locomotion2d/2d_nge_pruning.yaml
```

After running evolution to discover good morphologies, you can evaluate them using PPO via the provided eval configs.
```
python scripts/train_rl.py -p configs/locomotion2d/2d_eval.yaml
```
Note that you have to edit the config file to include either the path to the optimized morphology or a predefined type like `random2d` or `cheetah`. We evaluate all morphologies across a number of different environments. The provided configuration file runs evaluations for just one.

To better keep track of the experiment names, you can edit the name field in the config files.

By default, experiments are saved to the `data` directory. This can be changed by providing an output location with the `-o` flag.

## Rendering, Testing, and Plotting
See the test scripts for viewing agents after they have been trained.

For plotting results like those in the paper, use the plotting scripts. Note that to use the plotting scripts correctly, a specific directory structure is required. Details for this can be found in `optimal_agents/utils/plotter.py`.

## Citing

If you use this code. Please cite the paper.