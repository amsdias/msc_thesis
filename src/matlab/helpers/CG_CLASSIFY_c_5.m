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


function [f, df] = CG_CLASSIFY_c_5(VV,Dim,XX,target)

l1 = Dim(1);
l2 = Dim(2);
l3= Dim(3);
l4= Dim(4);
l5= Dim(5);
l6= Dim(6);
l7= Dim(7);
N = size(XX,1);

% Do decomversion.
 w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2 = reshape(VV(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 xxx = xxx+(l2+1)*l3;
 w3 = reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
 xxx = xxx+(l3+1)*l4;
 w4 = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
 xxx = xxx+(l4+1)*l5;
 w5 = reshape(VV(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
 xxx = xxx+(l5+1)*l6;
 w_class = reshape(VV(xxx+1:xxx+(l6+1)*l7),l6+1,l7);


  XX = [XX ones(N,1)];
  w1probs = 1./(1 + exp(-XX*w1));
  w1probs = [w1probs ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2));
  w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3));
  w3probs = [w3probs ones(N,1)];
  w4probs = 1./(1 + exp(-w3probs*w4));
  w4probs = [w4probs ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5));
  w5probs = [w5probs ones(N,1)];

  targetout = exp(w5probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,7);
  f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix_class = IO; 
dw_class = w5probs'*Ix_class; 

Ix5 = (Ix_class*w_class').*w5probs.*(1-w5probs);
Ix5 = Ix5(:,1:end-1);
dw5 =  w4probs'*Ix5;

Ix4 = (Ix5*w5').*w4probs.*(1-w4probs); 
Ix4 = Ix4(:,1:end-1);
dw4 =  w3probs'*Ix4;

Ix3 = (Ix4*w4').*w3probs.*(1-w3probs); 
Ix3 = Ix3(:,1:end-1);
dw3 =  w2probs'*Ix3;

Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw3(:)' dw4(:)' dw5(:)' dw_class(:)']'; 
