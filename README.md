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
<video src=https://github.com/user-attachments/assets/4fa87577-a5e3-4b21-9b32-852acee0d75c controls autoplay muted> <\video> | <video src=https://github.com/user-attachments/assets/fdb0b0fe-99fe-4a34-970d-1bb2548778f1 controls autoplay muted> <\video>

Lagrangian Trajectories | Lagrangian Trajectories Clustered According to Proximity
:-: | :-: 
<video src=https://github.com/user-attachments/assets/17d36431-ce27-448a-af05-0f8d5249b8b6 controls autoplay muted> <\video> | <video src=https://github.com/user-attachments/assets/42382279-5254-4979-ab38-09c8c7875432 controls autoplay muted> <\video>
