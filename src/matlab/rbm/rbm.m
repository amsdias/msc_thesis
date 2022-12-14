% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
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

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases, numdims, numbatches]=size(batchdata);

if restart ==1
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
  
  errors    = zeros(1,0);
  errorsums = zeros(1,0);
end

for epoch = epoch:maxepoch
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches
    if (mod(batch, 100) == 0) 
        fprintf(1,'epoch %d batch %d\r',epoch,batch);
    end  

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1))); 
  %poshidprobs = log(1 + exp(data*vishid + repmat(hidbiases,numcases,1)));
  
  %poshidprobs = max(0, data*vishid + repmat(hidbiases,numcases,1)); 
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  %negdata = max(0, poshidstates*vishid' + repmat(visbiases,numcases,1) + normrnd(0,1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1))),[numcases,numdims]));
  
  %negdata = max(0, poshidstates*vishid' + repmat(visbiases,numcases,1));
  
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
  %neghidprobs = log(1 + exp(negdata*vishid + repmat(hidbiases,numcases,1)));
  
  %neghidprobs = max(0, negdata*vishid + repmat(hidbiases,numcases,1));
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 )/batchsize);
  errsum = err + errsum;
  
  %if (mod(batch, 700) == 0)
  %  hFig = figure(1);
  %  set(hFig, 'Position', [200 300 1000 600]);
  %  colormap(gray)
  % for i = 1:10                                    % preview first 25 samples
  %     subplot(2,10,i)                              % plot them in 6 x 6 grid
  %     digit = reshape(data(i, 1:end), [28,28])';    % row = 28 x 28 image
  %     imshow(digit)
  %     subplot(2,10,i+10)
  %     digit2 = reshape(negdata(i, 1:end), [28,28])';
  %     imshow(digit2)
  % end
  %end
  if (mod(batch, 600) == 0)
  y = 1:size(data,2);
  hFig = figure(1);
  set(hFig, 'Position', [50 240 1000 600]);
  subplot(2,1,1);
  plot(y,data)
  axis tight %([0 size(data,2) 0 1])
  subplot(2,1,2);
  plot(y, negdata)
  axis tight %([0 size(data,2) 0 1])
 end

   if epoch>5
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  end
  %fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
  fprintf(1, '### Epoch %4i ###\n', epoch);
  fprintf(1, '### Error Sum (Average) %6.10f, Epoch Error (Average) %6.10f ###\n', errsum, err);
  errors    = [errors err];
  errorsums = [errorsums errsum];
  %w = waitforbuttonpress;
end
save(['saved_variables/errors_c5_' num2str(numdims) '-' num2str(numhid)], 'errors');

