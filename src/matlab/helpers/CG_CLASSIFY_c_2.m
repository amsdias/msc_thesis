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


function [f, df] = CG_CLASSIFY_c_2(VV,Dim,XX,target)

l1 = Dim(1);
l2 = Dim(2);
l4= Dim(3);
l5= Dim(4);
N = size(XX,1);

% Do decomversion.
w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
xxx = (l1+1)*l2;
w2 = reshape(VV(xxx+1:xxx+(l2+1)*l4),l2+1,l4);
xxx = xxx+(l2+1)*l4;

w_class = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

XX = [XX ones(N,1)];
w1probs = 1./(1 + exp(-XX*w1));
w1probs = [w1probs ones(N,1)];
w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs ones(N,1)];

targetout = exp(w2probs*w_class);
targetout = targetout./repmat(sum(targetout,2),1,7);
f = -sum(sum(target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class = IO; 
dw_class = w2probs'*Ix_class; 

Ix2 = (Ix_class*w_class').*w2probs.*(1-w2probs);
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw_class(:)']'; 
