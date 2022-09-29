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
fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

load ('saved_variables/vhclassify_f_1')

makebatches_bearings;
[numcases, numdims, numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1L=[vishidL; hidrecbiasesL];
w1R=[vishidR; hidrecbiasesR];

w_classL = -0.01*rand(size(w1L,2)+1,7);
w_classR = 0.01*rand(size(w1R,2)+1,7);
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1L,1)-1;

l4=size(w_classL,1)-1;
l5=7; 
test_err=[];
train_err=[];


for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
err_cr=0;
counter=0;
[numcases, numdims, numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches
  data = batchdata(:,:,batch);
  target = batchtargets(:,:,batch);
  data = [data ones(N,1)];
  w1probsL = 1./(1 + exp(-data*w1L));
  w1probsR = 1./(1 + exp(-data*w1R));
  w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];
  
  targetoutL = exp(w1probs*w_classL);
  targetoutR = exp(w1probs*w_classR);
  targetout = (targetoutL + targetoutR)/2;
  targetout = targetout./repmat(sum(targetout,2),1,7);

  [I, J]=max(targetout,[],2);
  [I1, J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
 end
 train_err(epoch)=(numcases*numbatches-counter);
 train_crerr(epoch)=err_cr/numbatches;

%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0;
err_cr=0;
counter=0;
[testnumcases, testnumdims, testnumbatches]=size(testbatchdata);
N=testnumcases;
for batch = 1:testnumbatches
  data = testbatchdata(:,:,batch);
  target = testbatchtargets(:,:,batch);
  data = [data ones(N,1)];
  w1probsL = 1./(1 + exp(-data*w1L)); 
  w1probsR = 1./(1 + exp(-data*w1R));
  w1probs = [((w1probsL + w1probsR)/2) ones(N,1)];
  
  targetoutL = exp(w1probs*w_classL);
  targetoutR = exp(w1probs*w_classR);
  targetout = (targetoutL + targetoutR)/2;
  targetout = targetout./repmat(sum(targetout,2),1,7);

  [I, J]=max(targetout,[],2);
  [I1, J1]=max(target,[],2);
  counter=counter+length(find(J==J1));
  err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
end
 test_err(epoch)=(testnumcases*testnumbatches-counter);
 test_crerr(epoch)=err_cr/testnumbatches;
 fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);

%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/10
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 targets=[]; 
 for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
  targets=[targets
        batchtargets(:,:,(tt-1)*10+kk)];
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;

  if epoch<6  % First update top-level weights holding other weights fixed. 
    N = size(data,1);
    XX = [data ones(N,1)];
    w1probsL = 1./(1 + exp(-XX*w1L)); 
    w1probsR = 1./(1 + exp(-XX*w1R));

    w1probs = (w1probsL + w1probsR)/2;
    %w3probs = [w3probs  ones(N,1)];

    VVL = [w_classL(:)']';
    VVR = [w_classR(:)']';
    Dim = [l4; l5];
    [XL, XR, fX] = minimizef(VVL, VVR, 'CG_CLASSIFY_INIT_f_1', max_iter, Dim, w1probs, targets);
    w_classL = reshape(XL,l4+1,l5);
    w_classR = reshape(XR,l4+1,l5);

  else
    VVL = [w1L(:)' w_classL(:)']';
    VVR = [w1R(:)' w_classR(:)']';
    Dim = [l1; l4; l5];
    [XL, XR, fX] = minimizef(VVL, VVR, 'CG_CLASSIFY_f_1', max_iter, Dim, data, targets);

    w1L = reshape(XL(1:(l1+1)*l4),l1+1,l4);
    w1R = reshape(XR(1:(l1+1)*l4),l1+1,l4);
    xxx = (l1+1)*l4;

    w_classL = reshape(XL(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
    w_classR = reshape(XR(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 %save mnistclassify_weights w1L w1R w_classL w_classR
 %save mnistclassify_error test_err test_crerr train_err train_crerr;
 save(['saved_variables/classify_weights_f_1_' num2str(iter)], 'w1L', 'w1R', 'w_classL', 'w_classR');
 save(['saved_variables/classify_error_f_1_' num2str(iter)], 'test_err', 'test_crerr', 'train_err', 'train_crerr');

end



