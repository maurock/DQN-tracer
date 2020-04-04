# DQN-tracer: Deep Q-Learning for Physically Based Rendering
Simple path tracer that uses Deep Q-Learning for path guiding

## THIS REPOSITORY IS A WORK IN PROGRESS...

### Work
Implementation of a novel Deep Q-Learning approach for Physically-Based Rendering. This work proposes an extension of Dahm and keller's paper "Learning Light Transport the Reinforced Way". A Deep Q-Learning agent learns the distribution of the incident radiance function for every point inside a virtual scene. Monte Carlo importance sampling is then used to sample rays from the learned distribution to efficently scatter rays towards the light source. This reduces noise during rendering.

Three different scenes have been implemented:

![three scenes](https://github.com/maurock/DQN-tracer/blob/master/images/threescenes2.png)

### Implementation
Many of the methods used for path tracing are implemented in C++ and compiled with pybind11. This library generates binaries for python. Instructions on how to generate binaries with CMake can be found in the official pybind11 documentation. The `.cpp` file with these methods can be found in <i>pybind-modules/smallpt_pybind.cpp</i>. I will soon provide further instructions on how to compile the C++ bindings. 
The Deep Q-Learning algorithm is implemented in python 3.6.8.

### Bayesian Optimization
I used Bayesian Optimization to optimize the hyperparameters of the Deep Neural Network. These parameters and additional parameters for training and testing are set in `bayesOpt.py`.

### Current results
Currently, the Deep Q-Learning method improves the SSIM score for the scene Box and Sunrise, while Q-Learning yields a better result for the scene Door. The reference image is rendered at 5120 SPP.
<img src="https://github.com/maurock/DQN-tracer/blob/master/images/result2.png" align="center" width="70%" height="70%">






