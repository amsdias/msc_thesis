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

rng('shuffle', 'twister');

%rand('seed', 'reset');
%rand('state', 'reset');

bearingsdata=[]; 
targets=[];
load 'data/dataset';
bearingsdata = features_training(:,:);
targets = labels_training(:,2:8);
totnum=size(bearingsdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

%rand('state',0); %so we know the permutation of the training data

randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(bearingsdata,2);
batchsize = 100;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 7, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = bearingsdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
clear bearingsdata targets;

bearingsdata=[];
targets=[];
bearingsdata = features_test(:,:);
targets = labels_test(:,2:8);
%bearingsdata = bearingsdata/255;

totnum=size(bearingsdata,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

%rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(bearingsdata,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 7, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = bearingsdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end
clear bearingsdata targets features_test features_training labels_test labels_training;


%%% Reset random seeds 
%rand('state',sum(100*clock)); 
%randn('state',sum(100*clock)); 



