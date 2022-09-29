use_data=[]; 
use_label=[];
load('../data/test.mat');

[numcases, numbatches]=size(features);
N=numcases;

data = [features ones(N,1)];
w1probsL = data*w1L;
w1probsR = data*w1R;
w1probs = (w1probsL + w1probsR)/2;

test_features = w1probs(:,:);
test_labels = str2num(labels(:,4));

load('../data/training.mat');
[numcases, numbatches]=size(features);
N=numcases;

data = [features ones(N,1)];
w1probsL = data*w1L;
w1probsR = data*w1R;
w1probs = (w1probsL + w1probsR)/2;

training_features = w1probs(:,:);
training_labels = str2num(labels(:,4));

save(['saved_features/feat_f_1L_' num2str(iter)], 'training_features', 'training_labels', 'test_features', 'test_labels')
