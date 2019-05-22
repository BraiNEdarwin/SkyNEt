# SkyNEt

This repository houses a collection of functions and scripts used by 
the Darwin team of the 
[NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) group at the 
University of Twente. 
This README serves to outline the structure of the repo and its files. 
Additionally it gives some notes about how to contribute. 
For more details, please refer to the 
[wiki](https://github.com/BraiNEdarwin/SkyNEt/wiki). 
Before you start with your experiments, please read and understand 
[how we handle data](https://github.com/BraiNEdarwin/SkyNEt/wiki/Data-Structures).

## Installation

To begin with, install [Anaconda](https://www.anaconda.com/download) 
(python3 version). 
For maintaining the code we use GitHub, so please make a GitHub account. 
To use GitHub on your computer you can use git in the command line, 
or if this does not ring a bell, we recommend using 
[GitHub Desktop](https://desktop.github.com/).
To run the code in this repo, we make use of an Anaconda environment 
called *skynet* (based on the [QCoDeS](https://github.com/QCoDeS/Qcodes) 
environment). 
To install this environment on a new PC, download the file 
environment.yml from this repo. 
Open an Anaconda prompt (or just a normal command prompt) and browse 
to the directory where you saved environment.yml. 
Then run the following command to install the skynet environment:

```
conda env create -f environment.yml
```

Now all that is left to do, is add the SkyNEt repository to the 
skynet environment, such that all script will be able to import it. 
To do this, go to your Anaconda prompt and activate the skynet 
environment by running:

```
activate skynet
```

Note that for non-windows users this will probably 
be `source activate skynet` instead.

Now open up an iPython console

```
ipython
```

And inside the console run the following commands:

```
import sys
sys.path
```

You will now see all directories where python will look for modules 
if you try to import one. 
There should be a path that looks something like this:

```
~/anaconda3/envs/skynet/lib/python3.6/site-packages/
```

This is the directory where you should place a path configuration 
file named `skynet.pth`. 
This will allow python to find the SkyNEt modules when imported. 
To do this, follow the instructions below:


* Go to the directory you found above, i.e. `~/this/that/site-packages/`
* Make a file named `skynet.pth` containing a line with the 
    absolute path to the directory containing your SkyNEt repo
* Start your python and check sys.path; 
    you should see the path to the repo there
  

Note: by convention, you should import SkyNEt explicitly, 
i.e. in the python sys.path do not include the SkyNEt directory, 
only include the directory a level higher. 
Since Python will look for scripts on that path, 
we recommend you keep it separated from all your other scripts to 
avoid interference.

To finish off the installation process, there is one package left to install.
Activate the skynet environment again and run the following command:

```
pip install nidaqmx
```

Now you are done with the installation process and ready to get to work!
Feel free to ask any of the code maintainers 
(listed at the bottom of this document) for help.

## Neural networks
If you want to work with neural networks, there is an extra dependency
`PyTorch`, which you have to install yourself. 
As this is OS dependent, please have a look at the instructions on the 
[website](https://pytorch.org/get-started/locally/).
Have a look at the various pages at the 
[wiki](https://github.com/BraiNEdarwin/SkyNEt/wiki). 

## Kinetic Monte Carlo
If you want to work with the kinetic monte carlo model, there are 
some other dependencies. The installation instructions can be found
on the separate repository for kmc, 
[here](https://github.com/brambozz/kmc_dn).

TODO: this will be updated with submodules instructions once it is 
setup.

## Repository structure

The repository is structured in a few folders. 
The *instruments* folder contains instrument drivers. 
The *modules* folder holds various modules containing function and 
class definitions used to run experiments. 
The *experiments* folder is where all the experiments are stored. 
Each individual experiment has its own measurement script 
(*.py* file) and configuration class definition.

As for the branching structure, there is a *master* and a *dev* branch. 
The *master* branch is only for stable versions of the software. 
Important changes or new features are implemented in the *dev* branch, 
which at some point will merge into a new version in the *master* branch. 
Each user at NE uses a *personal* branch to run experiments and 
optionally change code.

## Practical use

As mentioned above, make sure that there is a *personal* branch for you 
to run code in and experiment. 
If this is not the case, contact one of the code maintainers 
(see the end of this page). 
Assuming everything is installed correctly, 
this is the basic workflow for running an experiment.

* Open an Anaconda prompt and activate the qcodes environment.
* Browse to the directory containing the measurement script you wish to run.
* Make sure that you are checked out to your personal branch in git!
* Configure the configuration class definition file to your liking.
* Run the measurement script by running the following command:

```
python <your_script>.py
```

## Writing your own experiment scripts

Probably at some point you wish to write your own measurement scripts. 
There are a few instructions that you should follow in order to keep 
things coherent with the rest of the repository:

* Follow the file structure conventions 
(i.e. see https://github.com/BraiNEdarwin/SkyNEt/wiki/File-structure)
* Follow the conventions for the data structures in 
https://github.com/BraiNEdarwin/SkyNEt/wiki/Data-Structures
* The boolean_logic experiment serves as a template file. 
Please have a look at both boolean_logic.py and config_boolean_logic.py 
to see how we would like files to be structured. 
* If you work with NNets, take the TrainNets.py as template to train 
and/or load your networks. 

And please make sure you only work on your own branch, 
but feel free to suggest any experiments you think should go into the 
dev branch! 

## Setting up a new measurement PC

Usually, you will run experiments on a PC that is already setup for 
measurements. This section shows the steps you need to take te setup
a new measurement PC (this is unneccesary for personal PCs).

First follow the installation instructions above.
Then make sure that git can connect to the internet via the utwente 
proxy server:

* Go to C:/users/
* Open .gitconfig
* Add the following 
```
[http]
	proxy = http://proxy.utwente.nl:3128
```

If you want to use the ADwin for measurements, additionally do the 
following steps:

* Install the ADwin software with the CD-rom in the lab
* Install drivers for [Startech USB31000S] (https://www.startech.com/Networking-IO/usb-network-adapters/USB-3-to-Gigabit-Ethernet-NIC-Network-Adapter~USB31000S#dnlds)
* Configure the ADwin via adconfig

## Code contribution

If you wish to contribute to the code you are more than welcome to do so. 
If you see any bugs or come across issues you think need improvement, 
please raise an issue on this repository 
[here](https://github.com/BraiNEdarwin/SkyNEt/issues). 
If you want to know more about writing pieces of code to aid development, 
please refer to the [wiki](https://github.com/BraiNEdarwin/SkyNEt/wiki).

Please be pro-active and discuss with the code maintainers if you think 
some code you wrote can be valuable to improve this repo; 
for example, if you work on an improved sampling procedure, 
an optimized code or found a better way to implement a function, 
do contact us to discuss the adaptation into our code.  

## Code maintainers
Currently the users with admin rights on this repo are:
* Bram de Wilde, [@brambozz](https://github.com/brambozz) 
(b.dewilde-1@student.utwente.nl) 
* Hans-Christian Ruiz Euler [@hcruiz](https://github.com/hcruiz) 
(h.ruiz@utwente.nl)
* Bram van de Ven, [@bbroo1](https://github.com/bbroo1) 
(b.vandeven@utwente.nl)
