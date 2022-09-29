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

epsilonw      = 0.001;   % Learning rate for weights 
epsilonvb     = 0.001;   % Learning rate for biases of visible units 
epsilonhb     = 0.001;   % Learning rate for biases of hidden units 
weightcost  = 0.0003;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases, numdims, numbatches]=size(batchdata);

if restart ==1
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishidL = -0.1*rand(numdims, numhid);
  vishidR = 0.1*rand(numdims, numhid);
  
  hidbiasesL = -0.1*rand(1,numhid);
  hidbiasesR = 0.1*rand(1,numhid);
  
  visbiasesL = -0.1*rand(1,numdims);
  visbiasesR = 0.1*rand(1,numdims);

  poshidprobsL = zeros(numcases,numhid);
  poshidprobsR = zeros(numcases,numhid);
  
  neghidprobsL = zeros(numcases,numhid);
  neghidprobsR = zeros(numcases,numhid);
  
  posprods    = zeros(numdims,numhid);
  
  negprods    = zeros(numdims,numhid);
  
  vishidinc    = zeros(numdims,numhid);
  vishidincL   = zeros(numdims,numhid); 
  vishidincR   = zeros(numdims,numhid); 
  
  hidbiasinc  = zeros(1,numhid);
  
  visbiasinc  = zeros(1,numdims);
  
  batchposhidprobs=zeros(numcases,numhid,numbatches);
  
  errors=zeros(3,0);
  errorsums=zeros(3,0);
end

for epoch = epoch:maxepoch
 fprintf(1,'epoch %d\r',epoch); 
 errsumD=0;
 errsumL=0;
 errsumR=0;
    %if (mod(epoch, 10) == 0) 
    %    epsilonw      = epsilonw * 0.9;  
    %    epsilonvb     = epsilonvb * 0.9;  
    %    epsilonhb     = epsilonhb * 0.9;
    %end
 for batch = 1:numbatches
    if (mod(batch, 100) == 0) 
        fprintf(1,'epoch %d batch %d\r',epoch,batch);
    end

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  
  poshidprobsL = data*vishidL + repmat(hidbiasesL,numcases,1); %STEP 2
  poshidprobsR = data*vishidR + repmat(hidbiasesR,numcases,1); %STEP 2
  
  poshidprobsD = (poshidprobsL + poshidprobsR)/2;
  
  batchposhidprobs(:,:,batch) = poshidprobsD; %WHAT GOES HERE? PASS PROBS OR STATES?
  
  posprods = data' * poshidprobsD; %USED FOR WEIGHT UPDATES
  poshidact = sum(poshidprobsD);   %USED FOR HIDDEN BIAS UPDATES
  posvisact = sum(data);          %USED FOR VISIBLEBIAS UPDATES

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  poshidstatesL = poshidprobsL + randn(numcases,numhid); %STEP 3
  poshidstatesR = poshidprobsR + randn(numcases,numhid); %STEP 5

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdataL = 1./(1 + exp(-poshidstatesL*vishidL' - repmat(visbiasesL,numcases,1))); %STEP 8
  negdataR = 1./(1 + exp(-poshidstatesR*vishidR' - repmat(visbiasesR,numcases,1))); %STEP 8
  %negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  negdataD = (negdataL + negdataR)/2;
  
  neghidprobsL = negdataD*vishidL + repmat(hidbiasesL,numcases,1);   %STEP 14
  neghidprobsR = negdataD*vishidR + repmat(hidbiasesR,numcases,1);   %STEP 14
  
  neghidprobsD = (neghidprobsL + neghidprobsR)/2;
  
  negprods  = negdataD'*neghidprobsD; %USED FOR WEIGHT UPDATES
  neghidact = sum(neghidprobsD);     %USED FOR HIDDEN BIAS UPDATES
  negvisact = sum(negdataD);         %USED FOR VISIBLEBIAS UPDATES

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  errD = sum(sum((data-negdataD).^2 )/batchsize);
  errsumD = errD + errsumD;
  errL = sum(sum((data-negdataL).^2 )/batchsize);
  errsumL = errL + errsumL;
  errR = sum(sum((data-negdataR).^2 )/batchsize);
  errsumR = errR + errsumR;
  
  error=[errD;errL;errR];
  errorsum=[errsumD;errsumL;errsumR];
  errors=[errors error];
  errorsums=[errorsums errorsum];
  
 % if (mod(batch, 600) == 0)
 %   hFig = figure(1);
 %   set(hFig, 'Position', [50 50 1000 600]);
 %   colormap(gray)
 %  for i = 1:10                                    % preview first 25 samples
 %      subplot(4,10,i)                              % plot them in 6 x 6 grid
 %      digit = reshape(data(i, 1:end), [28,28])';    % row = 28 x 28 image
 %      imshow(digit)
 %      subplot(4,10,i+10)
 %      digit2 = reshape(negdataD(i, 1:end), [28,28])';
 %      imshow(digit2)
 %      subplot(4,10,i+20)
 %      digit3 = reshape(negdataL(i, 1:end), [28,28])';
 %      imshow(digit3)
 %      subplot(4,10,i+30)
 %      digit4 = reshape(negdataR(i, 1:end), [28,28])';
 %      imshow(digit4)       
 %  end
 % end
 %if (mod(batch, 600) == 0)
 % y = 1:size(data,2);
 % hFig = figure(1);
 % set(hFig, 'Position', [50 240 1000 600]);
 % subplot(4,1,1);
 % plot(y,data)
 % axis tight %([0 size(data,2) 0 1])
 % subplot(4,1,2);
 % plot(y, negdataD)
 % axis tight %([0 size(data,2) 0 1])
 % subplot(4,1,3);
 % plot(y, negdataL)
 % axis tight %([0 size(data,2) 0 1])
 % subplot(4,1,4);
 % plot(y, negdataR)
 % axis tight %([0 size(data,2) 0 1])
 %end

   if epoch>5
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end
   
   %if(mod(epoch, 100) == 0)
   % epsilonw = epsilonw * 0.5;   % Learning rate for weights 
   % epsilonvb = epsilonvb * 0.5;   % Learning rate for biases of visible units 
   % epsilonhb = epsilonhb * 0.5;
   %end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    %vishidincL = momentum*vishidincL + epsilonw*( (posprods-negprods)/numcases - weightcost*vishidL);
    %vishidincR = momentum*vishidincR + epsilonw*( (posprods-negprods)/numcases - weightcost*vishidR);
    deltaW = (posprods-negprods)/2;
    deltab = (posvisact-negvisact)/2;
    deltac = (poshidact-neghidact)/2;
    
    %vishidinc = momentum*vishidinc + epsilonw*(posprods - negprods) / 6;
    %vishidM = (vishidL + vishidR)/2;
    vishidinc = momentum*vishidinc + epsilonw*(deltaW/numcases); %- weightcost*vishidM);
    
    %vishidincL = momentum*vishidincL + epsilonw*((deltaW)/numcases - weightcost*vishidL);
    %vishidincR = momentum*vishidincR + epsilonw*((deltaW)/numcases - weightcost*vishidR);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*deltab;
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*deltac;

    vishidL = vishidL + vishidinc; %(epsilonw * deltaW);
    visbiasesL = visbiasesL + visbiasinc;
    hidbiasesL = hidbiasesL + hidbiasinc;
    
    vishidR = vishidR + vishidinc;
    visbiasesR = visbiasesR + visbiasinc;
    hidbiasesR = hidbiasesR + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

end  
  fprintf(1, '###epoch %4i errorsumD %6.10f, errorD %6.10f  ###\n', epoch, errsumD, errD);
  fprintf(1, 'epoch %4i errorsumL %6.10f, errorL %6.10f  \n', epoch, errsumL, errL);
  fprintf(1, 'epoch %4i errorsumR %6.10f, errorR %6.10f  \n', epoch, errsumR, errR);  
  %w = waitforbuttonpress;
end
