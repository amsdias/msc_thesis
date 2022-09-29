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

function [f, df] = CG_AE_f_3l(VVL, VVR, Dim, XX)  %VV - Weigths, XX - input

l1 = Dim(1);
l2 = Dim(2);
l3 = Dim(3);
l4= Dim(4);
l5= Dim(5);
l6= Dim(6);
l7= Dim(7);
N = size(XX,1);

% Do decomversion.
 w1L = reshape(VVL(1:(l1+1)*l2),l1+1,l2);
 w1R = reshape(VVR(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2L = reshape(VVL(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 w2R = reshape(VVR(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 xxx = xxx+(l2+1)*l3;
 w3L = reshape(VVL(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
 w3R = reshape(VVR(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
 xxx = xxx+(l3+1)*l4;
 w4L = reshape(VVL(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
 w4R = reshape(VVR(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
 xxx = xxx+(l4+1)*l5;
 w5L = reshape(VVL(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
 w5R = reshape(VVR(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
 xxx = xxx+(l5+1)*l6;
 w6L = reshape(VVL(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
 w6R = reshape(VVR(xxx+1:xxx+(l6+1)*l7),l6+1,l7);


  XX = [XX ones(N,1)];
  w1probsL = 1./(1 + exp(-XX*w1L)); 
  w1probsR = 1./(1 + exp(-XX*w1R)); 
  w1probs = [(w1probsL+w1probsR)/2 ones(N,1)];

  w2probsL = 1./(1 + exp(-w1probs*w2L));
  w2probsR = 1./(1 + exp(-w1probs*w2R));
  w2probs = [(w2probsL+w2probsR)/2 ones(N,1)];

  w3probsL = w2probs*w3L;
  w3probsR = w2probs*w3R;
  %w3probsL = 1./(1 + exp(-w2probs*w3L));
  %w3probsR = 1./(1 + exp(-w2probs*w3R));
  w3probs = [(w3probsL+w3probsR)/2 ones(N,1)];

  w4probsL = 1./(1 + exp(-w3probs*w4L));
  w4probsR = 1./(1 + exp(-w3probs*w4R));
  w4probs = [(w4probsL+w4probsR)/2 ones(N,1)];

  w5probsL = 1./(1 + exp(-w4probs*w5L));
  w5probsR = 1./(1 + exp(-w4probs*w5R));
  w5probs = [(w5probsL+w5probsR)/2 ones(N,1)];

  XXoutL = 1./(1 + exp(-w5probs*w6L));
  XXoutR = 1./(1 + exp(-w5probs*w6R));
  XXout = (XXoutL + XXoutR)/2;

f = -1/N*sum(sum(XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout)));
IO = 1/N*(XXout-XX(:,1:end-1));
Ix6=IO; 
dw6 =  w5probs'*Ix6;

Ix5L = (Ix6*w6L').*w5probs.*(1-w5probs);
Ix5R = (Ix6*w6R').*w5probs.*(1-w5probs);
Ix5 = (Ix5L + Ix5R)/2;
Ix5 = Ix5(:,1:end-1);
dw5 =  w4probs'*Ix5;

Ix4L = (Ix5*w5L').*w4probs.*(1-w4probs);
Ix4R = (Ix5*w5R').*w4probs.*(1-w4probs);
Ix4 = (Ix4L + Ix4R)/2;
Ix4 = Ix4(:,1:end-1);
dw4 =  w3probs'*Ix4;

Ix3L = (Ix4*w4L');
Ix3R = (Ix4*w4R');
%Ix3L = (Ix4*w4L').*w3probs.*(1-w3probs);
%Ix3R = (Ix4*w4R').*w3probs.*(1-w3probs);
Ix3 = (Ix3L + Ix3R)/2;
Ix3 = Ix3(:,1:end-1);
dw3 =  w2probs'*Ix3;

Ix2L = (Ix3*w3L').*w2probs.*(1-w2probs);
Ix2R = (Ix3*w3R').*w2probs.*(1-w2probs);
Ix2 = (Ix2L + Ix2R)/2;
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1L = (Ix2*w2L').*w1probs.*(1-w1probs);
Ix1R = (Ix2*w2R').*w1probs.*(1-w1probs);
Ix1 = (Ix1L + Ix1R)/2;
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw3(:)' dw4(:)' dw5(:)' dw6(:)']'; 


