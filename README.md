# Introduction

Welcome to the repository for the paper **"On complex network techniques for atmospheric ﬂow analysis: a polar vortex case study"** conducted at the Mathematical Institute of the University of Oxford. The paper focuses on exploring how network-based techniques can help improve our understanding of metereological and climatological phenomena. This repository contains the code necessary to perform the analysis presented in the paper.

# Repository Structure

The code is organized into the following directories:

- **buildnet**: Contains code to create the networks.
  - `mcorr.py`: Creates correlation-based Eulerian networks.
  - `mvort.py`: Creates vorticity-based Eulerian networks.
  - `mlag.py`: Creates Lagrangian networks.
- **usenet**: Contains useful code.
  - `prop.py`: Calculates network measures.
  - `findlcs.py`: Clusters trajectories.

# Media

Potential Vorticity | In Strenght - Out Strength of the Vorticity-Based Network
:-: | :-:
<video src=https://github.com/user-attachments/assets/7d27f3c2-c9fe-4cff-9b08-ffacbd536de5 controls autoplay muted> <\video> | <video src=https://github.com/user-attachments/assets/f075d88e-c901-4865-8f0c-be3eac2993f6 controls autoplay muted> <\video>

Lagrangian Trajectories | Lagrangian Trajectories Clustered According to Proximity
:-: | :-: 
<video src=https://github.com/user-attachments/assets/65d5ff59-80c9-44ed-a1d9-ac250e646110 controls autoplay muted> <\video> | <video src=https://github.com/user-attachments/assets/4c9cde14-91a2-40e2-b302-af9b826cff81 controls autoplay muted> <\video>
