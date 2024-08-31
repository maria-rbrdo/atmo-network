# Introduction

Welcome to the repository for the Master thesis **"Network-Based Analysis of the Winter Stratospheric Polar Vortex"**. This repository contains the code necessary to run the models and perform the analyses presented in the thesis.

## Repository Structure

The code is organized into the following directories:

- **bob**: Reads output data from the model.
- **netcdf**: Reads observation data.
- **networks**: Contains the analysis code.
  - **s1_create**: Contains code to create the networks.
    - `mcorr.py`: Creates correlation-based networks.
    - `mvort.py`: Creates vorticity-based networks.
    - `mdist.py` and `msim.py`: Create Lagrangian networks.
  - **s2_calculate**: Contains code to calculate network metrics.
  - **s3_plot**: Contains code to plot the network metrics.
