# Introduction

Welcome to the repository for the Master's thesis **"Network-Based Analysis of the Winter Stratospheric Polar Vortex"** conducted at the Mathematical Institute of the University of Oxford. The thesis focuses on exploring how network-based techniques can help improve our understanding of the dynamics of a winter stratospheric polar vortex model. This repository contains the code necessary to perform the analysis presented in the thesis.

# Repository Structure

The code is organized into the following directories:

- **bob**: Reads output data from the model.
- **netcdf**: Reads observation data.
- **networks**: Contains the analysis code.
  - **s1_create**: Contains code to create the networks.
    - `mcorr.py`: Creates correlation-based Eulerian networks.
    - `mvort.py`: Creates vorticity-based Eulerian networks.
    - `mdist.py`: Creates proximity-based Lagrangian networks.
    - `msim.py`: Creates similarity-based Lagrangian networks.
  - **s2_calculate**: Contains code to calculate network metrics.
  - **s3_plot**: Contains code to plot the network metrics.

# Media

Potential Vorticity | In Strenght - Out Strength of the Vorticity-Based Network
:-: | :-:
<video src=https://github.com/user-attachments/assets/46a82d8b-2c0d-468a-97ae-8cff9da69cf2 controls autoplay muted> <\video> | <video src=https://github.com/user-attachments/assets/fdb0b0fe-99fe-4a34-970d-1bb2548778f1 controls autoplay muted> <\video>

Lagrangian Trajectories | Lagrangian Trajectories Clustered According to Proximity
:-: | :-: 
<video src=https://github.com/user-attachments/assets/c53720a9-e451-49fe-9d9e-64ccf1cb399e controls autoplay muted> <\video> | <video src=https://github.com/user-attachments/assets/de03ac9e-e756-4c02-9a7d-2ef0bbab90e2 controls autoplay muted> <\video>
