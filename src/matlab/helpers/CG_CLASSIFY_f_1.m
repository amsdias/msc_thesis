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


function [f, df] = CG_CLASSIFY_f_1(VVL, VVR, Dim, XX, target)

l1 = Dim(1);
l4 = Dim(2);
l5 = Dim(3);
N = size(XX,1);

% Do decomversion.
w1L = reshape(VVL(1:(l1+1)*l4),l1+1,l4);
w1R = reshape(VVR(1:(l1+1)*l4),l1+1,l4);
xxx = (l1+1)*l4;

w_classL = reshape(VVL(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
w_classR = reshape(VVR(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

XX = [XX ones(N,1)];
w1probsL = 1./(1 + exp(-XX*w1L));
w1probsR = 1./(1 + exp(-XX*w1R));
w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];

targetoutL = exp(w1probs*w_classL);
targetoutR = exp(w1probs*w_classR);
targetout = (targetoutL + targetoutR)/2;
targetout = targetout./repmat(sum(targetout,2),1,7);
f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class = IO; 
dw_class =  w1probs'*Ix_class; 

Ix1L = (Ix_class*w_classL').*w1probs.*(1-w1probs);
Ix1R = (Ix_class*w_classR').*w1probs.*(1-w1probs);
Ix1 = (Ix1L + Ix1R)/2;
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw_class(:)']'; 
