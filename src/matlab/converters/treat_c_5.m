use_data=[]; 
use_label=[];
load('../data/test.mat');

[numcases, numbatches]=size(features);
N=numcases;

data = [features ones(N,1)];
w1probs = [1./(1 + exp(-data*w1)) ones(N,1)];
w2probs = [1./(1 + exp(-w1probs*w2)) ones(N,1)];
w3probs = [1./(1 + exp(-w2probs*w3)) ones(N,1)];
w4probs = [1./(1 + exp(-w3probs*w4)) ones(N,1)];
w5probs = w4probs*w5;

test_features = w5probs(:,:);
test_labels = str2num(labels(:,4));

load('../data/training.mat');
[numcases, numbatches]=size(features);
N=numcases;

data = [features ones(N,1)];
w1probs = [1./(1 + exp(-data*w1)) ones(N,1)];
w2probs = [1./(1 + exp(-w1probs*w2)) ones(N,1)];
w3probs = [1./(1 + exp(-w2probs*w3)) ones(N,1)];
w4probs = [1./(1 + exp(-w3probs*w4)) ones(N,1)];
w5probs = w4probs*w5;

training_features = w5probs(:,:);
training_labels = str2num(labels(:,4));

save(['saved_features/feat_c_5NL_' num2str(iter)], 'training_features', 'training_labels', 'test_features', 'test_labels')