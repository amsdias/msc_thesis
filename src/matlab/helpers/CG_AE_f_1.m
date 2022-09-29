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

function [f, df] = CG_AE_f_1(VVL, VVR, Dim, XX)  %VV - Weigths, XX - input

l1 = Dim(1);
l2 = Dim(2);
l3 = Dim(3);
N = size(XX,1);

% Do decomversion.
 w1L = reshape(VVL(1:(l1+1)*l2),l1+1,l2);
 w1R = reshape(VVR(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2L = reshape(VVL(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 w2R = reshape(VVR(xxx+1:xxx+(l2+1)*l3),l2+1,l3);


  XX = [XX ones(N,1)];
  w1probsL = 1./(1 + exp(-XX*w1L)); 
  w1probsR = 1./(1 + exp(-XX*w1R)); 
  w1probs = [(w1probsL+w1probsR)/2 ones(N,1)];

  XXoutL = 1./(1 + exp(-w1probs*w2L));
  XXoutR = 1./(1 + exp(-w1probs*w2R));
  XXout = (XXoutL + XXoutR)/2;

f = -1/N*sum(sum(XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout)));
IO = 1/N*(XXout-XX(:,1:end-1));
Ix2=IO; 
dw2 =  w1probs'*Ix2;

Ix1L = (Ix2*w2L').*w1probs.*(1-w1probs);
Ix1R = (Ix2*w2R').*w1probs.*(1-w1probs);
Ix1 = (Ix1L + Ix1R)/2;
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)']'; 


