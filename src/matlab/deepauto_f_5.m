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
    
    maxepoch = 30; %In the Science paper we use maxepoch=50, but it works just fine. 
    numhid = 400; numpen = 200; numpen2 = 100; numpen3 = 50; numopen = 20;    

    fprintf(1,'Pretraining a deep autoencoder. \n');
    fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

    makebatches_bearings;

    [numcases, numdims, numbatches]=size(batchdata);

    fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
    restart=1;
    STFNrbm;
    hidrecbiasesL=hidbiasesL; hidrecbiasesR=hidbiasesR;    
    save('saved_variables/vh_f_5', 'vishidL', 'vishidR', 'hidrecbiasesL', 'hidrecbiasesR', 'visbiasesL', 'visbiasesR');

    fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
    batchdata=batchposhidprobs;
    numhid=numpen;
    restart=1;
    STFNrbm;
    hidpenL=vishidL; hidpenR = vishidR; penrecbiasesL=hidbiasesL; penrecbiasesR=hidbiasesR; hidgenbiasesL=visbiasesL; hidgenbiasesR=visbiasesR;    
    save('saved_variables/hp_f_5', 'hidpenL', 'hidpenR', 'penrecbiasesL', 'penrecbiasesR', 'hidgenbiasesL', 'hidgenbiasesR');

    fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
    batchdata=batchposhidprobs;
    numhid=numpen2;
    restart=1;
    STFNrbm;
    hidpen2L=vishidL; hidpen2R = vishidR; penrecbiases2L = hidbiasesL; penrecbiases2R = hidbiasesR; hidgenbiases2L=visbiasesL; hidgenbiases2R=visbiasesR;    
    save('saved_variables/hp2_f_5', 'hidpen2L', 'hidpen2R', 'penrecbiases2L', 'penrecbiases2R', 'hidgenbiases2L', 'hidgenbiases2R');

    fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numpen3);
    batchdata=batchposhidprobs;
    numhid=numpen3;
    restart=1;
    STFNrbm;
    hidpen3L=vishidL; hidpen3R = vishidR; penrecbiases3L = hidbiasesL; penrecbiases3R = hidbiasesR; hidgenbiases3L=visbiasesL; hidgenbiases3R=visbiasesR;    
    save('saved_variables/hp3_f_5', 'hidpen3L', 'hidpen3R', 'penrecbiases3L', 'penrecbiases3R', 'hidgenbiases3L', 'hidgenbiases3R');

    fprintf(1,'\nPretraining Layer 5 with RBM: %d-%d \n',numpen3,numopen);
    batchdata=batchposhidprobs;
    numhid=numopen; 
    restart=1;
    STFNrbm;    
    hidtopL=vishidL; hidtopR=vishidR; toprecbiasesL=hidbiasesL; toprecbiasesR=hidbiasesR; topgenbiasesL=visbiasesL; topgenbiasesR=visbiasesR;    
    save('saved_variables/po_f_5', 'hidtopL', 'hidtopR', 'toprecbiasesL', 'toprecbiasesR', 'topgenbiasesL', 'topgenbiasesR');

    backpropae_f_5;
    treat_f_5;    

    clear all
    close all

end

