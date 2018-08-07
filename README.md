# SkyNEt

This repository houses a collection of functions and scripts used by the Darwin team of the NanoElectronics group (utwente). This README serves to outline the structure of the repo and its files. Additionally it gives some notes about how to contribute. For more details, please refer to the wiki (link)

## Installation notes

To begin with, install Anaconda (link) (python3). For maintaining the code we use GitHub, so please make a GitHub account. To use GitHub on your computer you can use git in the command line, or if this does not ring a bell, we recommend using GitHub Desktop (link).
The code in this repo relies on a couple of packages and uses some drivers from [QCoDeS](https://github.com/QCoDeS/Qcodes). We use their *qcodes* Anaconda environment to run our scripts. To install this environment on a new PC, duplicate the repo and browse to it in an Anaconda prompt. Then run the following command to install the qcodes environment:

```
conda env create -f environment.yml
```

## Repository structure

The repository is structured in a few folders. The *instruments* folder contains instrument drivers. The *modules* folder holds various modules containing function and class definitions used to run experiments. The *experiments* folder is where all the experiments are stored. Each individual experiment has its own measurement script (*.py* file) and configuration class definition.

As for the branching structure, there is a *master* and a *dev* branch. The *master* branch is only for stable versions of the software. Important changes or new features are implemented in the *dev* branch, which at some point will merge into a new version in the *master* branch. Each user at NE uses a *personal* branch to run experiments and optionally change code.

## Practical use

As mentioned above, make sure that there is a *personal* branch for you to run code in and experiment. If this is not the case, contact one of the code maintainers (see the end of this page). Assuming everything is installed correctly, this is the basic workflow for running an experiment.

Open an Anaconda prompt and activate the qcodes environment.
Browse to the directory containing the measurement script you wish to run.
Make sure that you are checked out to your personal branch in git. This can be done using the following commands?
Configure the configuration class definition to your liking.
Run the measurement script by running the following command:

## Configuration class definitions

Some explanation of the basic structure of the configuration class and how to change it.

## Code contribution

If you wish to contribute to the code you are more than welcome to do so. If you see any bugs or come across issues you think need improvement, please raise an issue on this repository here (link). If you want to know more about writing pieces of code to aid development, please refer to the wiki (link).

## Code maintainers
Currently the users with admin rights on this repo are:
