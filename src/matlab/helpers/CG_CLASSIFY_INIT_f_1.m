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


function [f, df] = CG_CLASSIFY_INIT_f_1(VVL, VVR,Dim,w1probs,target)
l1 = Dim(1);
l2 = Dim(2);
N = size(w1probs,1);
% Do decomversion.
  w_classL = reshape(VVL,l1+1,l2);
  w_classR = reshape(VVR,l1+1,l2);
  w1probs = [w1probs ones(N,1)];  

  targetoutL = exp(w1probs*w_classL);
  targetoutR = exp(w1probs*w_classR);
  targetout = (targetoutL + targetoutR)/2;
  targetout = targetout./repmat(sum(targetout,2),1,7);
  f = -sum(sum( target(:,1:end).*log(targetout))) ;
IO = (targetout-target(:,1:end));
Ix_class=IO; 
dw_class =  w1probs'*Ix_class;
%dw_class =  w3probs'*Ix_class;

df = [dw_class(:)']'; 

