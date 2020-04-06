# ENPM_673_Project_3
# Implementation of Gaussian Mixture Models for Image Segmentation

This repository has a report in pdf format with  descriptions of problem 1 and 2. All the solutions of the problem set have been implemented in In different files. There is also a .ipynb notebook for testing.
## Getting Started
Download the entire .zip and extract in a folder. A python environment is required to run the codes.
## Prerequisites
The programs uses numpy, matplotlib and opencv libraries. Moreover, It uses json and scipy libraries .Hence, these libraries should be pre-installed. 
## Running the programs
1. Open Terminal (for Linux and MacOS) or command line (for windows)
2.  Navigate to the $(extracted folder)/code directory using ```cd $(path_to_folder)/code```
3.  Run the following command for the python programs 
	- ```python 1D_GMM.py``` for generating a 1D dataset consisting of 3 gaussian clusters. The GMM algorithm computes the parameters and visualization of the dataset is performed
    - ```python data_collection.py``` is used to obtain frames for data collection
    - ```python GMM_color_segmentation.py``` for generating the dataset from the json file and fitting a GMM to it. It saves the means and covariances in ```means.npy and covar.npy```
    - ```python buoy_detection.py``` is used run the pipeline required for buoy detection that uses the GMM parameters obtained by training on the ROI pixel dataset

## Authors
Vasista Ayyagari
Kumar Sambhav Mahipal
Raghav Agarwal
