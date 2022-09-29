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

for iter = 1:30
    
    addpath(genpath('helpers'))
    addpath(genpath('data'))
    addpath(genpath('rbm'))
    addpath(genpath('converters'))

    maxepoch=30; 
    numhid=20; 

    fprintf(1,'Converting Raw files into Matlab format \n');

    fprintf(1,'Pretraining a deep autoencoder. \n');
    fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

    makebatches_bearings;
    [numcases, numdims, numbatches]=size(batchdata);

    fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
    restart=1;
    STFNrbm;
    hidrecbiasesL=hidbiasesL; hidrecbiasesR=hidbiasesR; 
    save('saved_variables/vhclassify_f_1', 'vishidL', 'vishidR', 'hidrecbiasesL', 'hidrecbiasesR', 'visbiasesL', 'visbiasesR');

    backpropclassify_f_1;
    classify_f_1;

    clear all
    close all

end

