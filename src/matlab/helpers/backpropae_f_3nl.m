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

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.  

maxepoch=100;
fprintf(1,'\nFine-tuning deep autoencoder by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

load ('saved_variables/vh_f_3.mat')
load ('saved_variables/hp_f_3.mat')
load ('saved_variables/po_f_3.mat')

makebatches_bearings;
[numcases, numdims, numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w1L=[vishidL; hidrecbiasesL];
w1R=[vishidR; hidrecbiasesR];
w2L=[hidpenL; penrecbiasesL];
w2R=[hidpenR; penrecbiasesR];
w3L=[hidtopL; toprecbiasesL];
w3R=[hidtopR; toprecbiasesR];
w4L=[hidtopL'; topgenbiasesL]; 
w4R=[hidtopR'; topgenbiasesR];
w5L=[hidpenL'; hidgenbiasesL];
w5R=[hidpenR'; hidgenbiasesR];
w6L=[vishidL'; visbiasesL];
w6R=[vishidR'; visbiasesR];

%%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1L,1)-1;
l2=size(w2L,1)-1;
l3=size(w3L,1)-1;
l4=size(w4L,1)-1;
l5=size(w5L,1)-1;
l6=size(w6L,1)-1;
l7=l1; 
test_err=[];
train_err=[];


for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
[numcases, numdims, numbatches]=size(batchdata);
N=numcases;
for batch = 1:numbatches
  data = batchdata(:,:,batch);
  data = [data ones(N,1)];
  w1probsL = 1./(1 + exp(-data*w1L));
  w1probsR = 1./(1 + exp(-data*w1R));
  w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];

  w2probsL = 1./(1 + exp(-w1probs*w2L));
  w2probsR = 1./(1 + exp(-w1probs*w2R));
  w2probs = [((w2probsL + w2probsR)/2) ones(N,1)];

  %w4probsL = w3probs*w4L;
  %w4probsR = w3probs*w4R;
  w3probsL = 1./(1 + exp(-w2probs*w3L));
  w3probsR = 1./(1 + exp(-w2probs*w3R));
  w3probs = [((w3probsL + w3probsR)/2) ones(N,1)];

  w4probsL = 1./(1 + exp(-w3probs*w4L));
  w4probsR = 1./(1 + exp(-w3probs*w4R));
  w4probs = [((w4probsL + w4probsR)/2) ones(N,1)];

  w5probsL = 1./(1 + exp(-w4probs*w5L));
  w5probsR = 1./(1 + exp(-w4probs*w5R));
  w5probs = [((w5probsL + w5probsR)/2) ones(N,1)];

  dataoutL = 1./(1 + exp(-w5probs*w6L));
  dataoutR = 1./(1 + exp(-w5probs*w6R));
  dataout = (dataoutL + dataoutR)/2;

  err = err + 1/N*sum(sum((data(:,1:end-1)-dataout).^2 )); 
end
train_err(epoch) = err/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% DISPLAY FIGURE TOP ROW REAL DATA BOTTOM ROW RECONSTRUCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf(1,'Displaying in figure 1: Top row - real data, Bottom row -- reconstructions \n');
%y1 = 1:size(data,2);
%y2 = 1:size(dataout,2);
%hFig = figure(1);
%set(hFig, 'Position', [50 50 1000 600]);
%subplot(4,1,1);
%plot(y2,data(:,1:size(data,2)-1))
%axis tight %([0 size(data,2) 0 1])
%subplot(4,1,2);
%plot(y2, dataout)
%axis tight %([0 size(data,2) 0 1])
%subplot(4,1,3);
%plot(y2, dataoutL)
%axis tight %([0 size(data,2) 0 1])
%subplot(4,1,4);
%plot(y2, dataoutR)
%axis tight %([0 size(data,2) 0 1])


%%%%TODO: DISPLAY data, dataout, dataoutL, dataoutR%%%%%%%%%
%output=[];
% for ii=1:15
%  output = [output data(ii,1:end-1)' dataout(ii,:)'];
% end
%   if epoch==1 
%   close all 
%   figure('Position',[100,600,1000,200]);
%   else 
%   figure(1)
%   end 
%   mnistdisp(output);
%   drawnow;
if (mod(batch, 60) == 0)
  y = 1:size(data(:,1:end-1),2);
  hFig = figure(1);
  set(hFig, 'Position', [50 240 1000 600]);
  subplot(2,1,1);
  plot(y,data(:,1:end-1))
  axis tight %([0 size(data,2) 0 1])
  subplot(2,1,2);
  plot(y, dataout)
  axis tight %([0 size(data,2) 0 1])
 end

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[testnumcases, testnumdims, testnumbatches]=size(testbatchdata);
N=testnumcases;
err=0;
for batch = 1:testnumbatches
  data = testbatchdata(:,:,batch);
  data = [data ones(N,1)];
  w1probsL = 1./(1 + exp(-data*w1L));
  w1probsR = 1./(1 + exp(-data*w1R));
  w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];

  w2probsL = 1./(1 + exp(-w1probs*w2L));
  w2probsR = 1./(1 + exp(-w1probs*w2R));
  w2probs = [((w2probsL + w2probsR)/2) ones(N,1)];

  %w4probsL = w3probs*w4L;
  %w4probsR = w3probs*w4R;
  w3probsL = 1./(1 + exp(-w2probs*w3L));
  w3probsR = 1./(1 + exp(-w2probs*w3R));
  w3probs = [((w3probsL + w3probsR)/2) ones(N,1)];

  w4probsL = 1./(1 + exp(-w3probs*w4L));
  w4probsR = 1./(1 + exp(-w3probs*w4R));
  w4probs = [((w4probsL + w4probsR)/2) ones(N,1)];

  w5probsL = 1./(1 + exp(-w4probs*w5L));
  w5probsR = 1./(1 + exp(-w4probs*w5R));
  w5probs = [((w5probsL + w5probsR)/2) ones(N,1)];

  dataoutL = 1./(1 + exp(-w5probs*w6L));
  dataoutR = 1./(1 + exp(-w5probs*w6R));
  dataout = (dataoutL + dataoutR)/2;

  err = err + 1/N*sum(sum( (data(:,1:end-1)-dataout).^2 ));
end
test_err(epoch)=err/testnumbatches;
fprintf(1,'Before epoch %d Train squared error: %6.3f Test squared error: %6.3f \t \t \n',epoch,train_err(epoch),test_err(epoch));
if(epoch > 20 && mod(epoch-1, 20) == 0)
  fprintf(1,'%6.3f %6.3f \t \t \n',train_err(epoch)/train_err(epoch-20),test_err(epoch)/test_err(epoch-20));
  %if(train_err(epoch)/train_err(epoch-20) > 0.95 && test_err(epoch)/test_err(epoch-20) > 0.95)
  %  break
  %end
end
%%%%%%%%%%%%%% END OF COMPUTING TEST RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tt=0;
for batch = 1:numbatches/10
  fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  tt=tt+1; 
  data=[];
  for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  max_iter=3;
  VVL = [w1L(:)' w2L(:)' w3L(:)' w4L(:)' w5L(:)' w6L(:)']';  %%do one or both sets at once?
  VVR = [w1R(:)' w2R(:)' w3R(:)' w4R(:)' w5R(:)' w6R(:)']';  %%do one or both sets at once?
  Dim = [l1; l2; l3; l4; l5; l6; l7];

  %[XL, fXL] = minimize(VVL,'CG_MNIST',max_iter,Dim,data);
  %[XR, fXR] = minimize(VVR,'CG_MNIST',max_iter,Dim,data);
  [XL, XR, fX] = minimizef(VVL, VVR, 'CG_AE_f_3', max_iter, Dim, data);

  w1L = reshape(XL(1:(l1+1)*l2),l1+1,l2);
  w1R = reshape(XR(1:(l1+1)*l2),l1+1,l2);
  xxx = (l1+1)*l2;
  w2L = reshape(XL(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
  w2R = reshape(XR(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
  xxx = xxx+(l2+1)*l3;
  w3L = reshape(XL(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
  w3R = reshape(XR(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
  xxx = xxx+(l3+1)*l4;
  w4L = reshape(XL(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
  w4R = reshape(XR(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
  xxx = xxx+(l4+1)*l5;
  w5L = reshape(XL(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
  w5R = reshape(XR(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
  xxx = xxx+(l5+1)*l6;
  w6L = reshape(XL(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
  w6R = reshape(XR(xxx+1:xxx+(l6+1)*l7),l6+1,l7);

%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  end

  %save mnist_weights3f w1L w1R w2L w2R w3L w3R w4L w4R w5L w5R w6L w6R 
  %save mnist_error3f test_err train_err;
  save(['saved_variables/weights_f_3_' num2str(iter)], 'w1L', 'w1R', 'w2L', 'w2R', 'w3L', 'w3R', 'w4L', 'w4R', 'w5L', 'w5R', 'w6L', 'w6R');  
  save(['saved_variables/error_f_3_' num2str(iter)], 'test_err', 'train_err');

end
