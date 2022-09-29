% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

clear all
close all

for iter = 1:1
    
    addpath(genpath('helpers'))
    addpath(genpath('data'))
    addpath(genpath('rbm'))
    addpath(genpath('converters'))

    maxepoch=30; 
    numhid=400; numpen=200; numpen2=100; numpen3 = 50; numpen4 = 20;

    fprintf(1,'Converting Raw files into Matlab format \n');

    fprintf(1,'Pretraining a deep autoencoder. \n');
    fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

    makebatches_bearings;
    [numcases, numdims, numbatches]=size(batchdata);

    fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
    restart=1;
    rbm;
    hidrecbiases=hidbiases; 
    save('saved_variables/vhclassify_c_5', 'vishid', 'hidrecbiases', 'visbiases');

    fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
    batchdata=batchposhidprobs;
    numhid=numpen;
    restart=1;
    rbm;
    hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
    save('saved_variables/hpclassify_c_5', 'hidpen', 'penrecbiases', 'hidgenbiases');

    fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
    batchdata=batchposhidprobs;
    numhid=numpen2;
    restart=1;
    rbm;
    hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
    save('saved_variables/hp2classify_c_5', 'hidpen2', 'penrecbiases2', 'hidgenbiases2');

    fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numpen3);
    batchdata=batchposhidprobs;
    numhid=numpen3;
    restart=1;
    rbm;
    hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;
    save('saved_variables/hp3classify_c_5', 'hidpen3', 'penrecbiases3', 'hidgenbiases3');

    fprintf(1,'\nPretraining Layer 5 with RBM: %d-%d \n',numpen3,numpen4);
    batchdata=batchposhidprobs;
    numhid=numpen4;
    restart=1;
    rbm;
    hidpen4=vishid; penrecbiases4=hidbiases; hidgenbiases4=visbiases;
    save('saved_variables/hp4classify_c_5', 'hidpen4', 'penrecbiases4', 'hidgenbiases4');

    backpropclassify_c_5;
    classify_c_5;

    clear all
    close all

end

