Master thesis Appendix A online reference: Matlab and Python code
Matlab functions and code for the Master Thesis of Angelo Dias @ github.com/amsdias/msc\_thesis/

Restricted Boltzmann Machine Based Autoencoders for the Classification of Faults in Rotational Mechanical Systems, Universidade do Algarve, 2022.

Navigation

The RBM code resides in the src/matlab folder.
The script can be started by running the start.m script, which will ask for inputs to select the model (Autoencoder or Classifier), variant (crisp or fuzzy),
and number of layer (1-5).

The code will execute 30 runs by default. This number can be changed in the scripts for each model:
deepauto\_\{v\}\_\{n\}.m and deepclassify\_\{v\}\_\{n\}.m.

At the end the data will be saved to either the saved_features or saved_variables folder.
The Classifier version will output the classification results, and the Autoencoder version will output the weights of the trained models,
which will then be used to generate new reduced dimensionality datasets for use with an external classifier.

The src/python folder contain 3 subfolders:

1. the data_treatment subfolder which contains the scripts used to perform data augmentation and feature generation from the original raw data,
and the script used to scale the output of the first script to values between 0 and 1.

2. the rfc folder contains scripts used to classify the output of the autoencoder model, and also RFC and PCA+RFC scripts used directly on the work dataset,
to obtain results to be used for comparison.

3. the statistics folder, containing scripts used to extract metrics and figures for analysis of the results.