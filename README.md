# SkyNEt

This repository houses a collection of functions and scripts used by the Darwin team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) group at the University of Twente. This README serves to outline the structure of the repo and its files. Additionally it gives some notes about how to contribute. For more details, please refer to the [wiki](https://github.com/BraiNEdarwin/SkyNEt/wiki).

## Installation

To begin with, install [Anaconda](https://www.anaconda.com/download) (python3 version). For maintaining the code we use GitHub, so please make a GitHub account. To use GitHub on your computer you can use git in the command line, or if this does not ring a bell, we recommend using [GitHub Desktop](https://desktop.github.com/).
To run the code in this repo, we make use of an Anaconda environment called *skynet* (based on the [QCoDeS](https://github.com/QCoDeS/Qcodes) environment). To install this environment on a new PC, download the file environment.yml from this repo. Open an Anaconda prompt (or just a normal command prompt) and browse to the directory where you saved environment.yml. Then run the following command to install the skynet environment:

```
conda env create -f environment.yml
```

Now all that is left to do, is add the SkyNEt repository to the skynet environment, such that all script will be able to import it. To do this, go to your Anaconda prompt and activate the skynet environment by running:

```
conda activate skynet
```

Now open up an iPython console

```
ipython
```

And inside the console run the following commands:

```
import sys
sys.path
```

You will now see all directories where python will look for modules if you try to import one. There should be a path that looks something like this:

```
~/anaconda3/envs/skynet/lib/python3.6
```

This is the directory where you should place this repo. So download/clone this full reporitory and place it in the directory you found above. Make sure that you set up git to recognize this location and feel free to ask any of the code maintainers (listed at the bottom of this document) for help.
Now you are done with the installation process and ready to get to work!

## Repository structure

The repository is structured in a few folders. The *instruments* folder contains instrument drivers. The *modules* folder holds various modules containing function and class definitions used to run experiments. The *experiments* folder is where all the experiments are stored. Each individual experiment has its own measurement script (*.py* file) and configuration class definition.

As for the branching structure, there is a *master* and a *dev* branch. The *master* branch is only for stable versions of the software. Important changes or new features are implemented in the *dev* branch, which at some point will merge into a new version in the *master* branch. Each user at NE uses a *personal* branch to run experiments and optionally change code.

## Practical use

As mentioned above, make sure that there is a *personal* branch for you to run code in and experiment. If this is not the case, contact one of the code maintainers (see the end of this page). Assuming everything is installed correctly, this is the basic workflow for running an experiment.

* Open an Anaconda prompt and activate the qcodes environment.
* Browse to the directory containing the measurement script you wish to run.
* Make sure that you are checked out to your personal branch in git!
* Configure the configuration class definition file to your liking.
* Run the measurement script by running the following command:

```
python <your_script>.py
```

## Writing your own experiment scripts

Probably at some point you wish to write your own measurement scripts. There are a few instructions that you should follow in order to keep things coherent with the rest of the repository:

* Follow the file structure conventions (i.e. see https://github.com/BraiNEdarwin/SkyNEt/wiki/File-structure)
* The boolean_logic experiment serves as a template file. Please have a look at both boolean_logic.py and config_boolean_logic.py to see how we would like files to be structured. 

And please make sure you only work on your own branch, but feel free to suggest any experiments you think should go into the dev branch!

## Code contribution

If you wish to contribute to the code you are more than welcome to do so. If you see any bugs or come across issues you think need improvement, please raise an issue on this repository [here](https://github.com/BraiNEdarwin/SkyNEt/issues). If you want to know more about writing pieces of code to aid development, please refer to the [wiki](https://github.com/BraiNEdarwin/SkyNEt/wiki).

## Code maintainers
Currently the users with admin rights on this repo are:
* Bram de Wilde, [@brambozz](https://github.com/brambozz) (b.dewilde-1@student.utwente.nl) 
* Hans-Christian Ruiz Euler [@hcruiz](https://github.com/hcruiz) (h.ruiz@utwente.nl)
* Bram van de Ven, [@bbroo1](https://github.com/bbroo1) (b.vandeven@utwente.nl)
