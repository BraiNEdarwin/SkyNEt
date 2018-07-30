# SkyNEt

This repository houses a collection of functions and scripts used by the Darwin team of the NanoElectronics group (utwente). This README serves to outline the structure of the repo and its files. Additionally it gives some notes about how to contribute.

## Installation notes

The code in this repo relies on a couple of packages and uses some drivers from [QCoDeS](https://github.com/QCoDeS/Qcodes). We use their *qcodes* Anaconda environment to run our scripts. To install this environment on a new PC, duplicate the repo and browse to it in an Anaconda prompt. Then run the following command to install the qcodes environment:

```
conda env create -f environment.yml
```

## Repository structure

In experiments, we basically use this repo to run measurement scripts, which in turn use predefined functions. The repository structure reflects this: all folders hold function definition files, all separate .py files are measurement scripts. The *instruments* folder contains instrument drivers. The *modules* folder holds various modules containing functions used in the measurement scripts. Any *.py* file is a measurement script. Please refer to *template.py* to see how you can structure your measurement script.

## Practical use



## Implementing new functionality



## Issues
