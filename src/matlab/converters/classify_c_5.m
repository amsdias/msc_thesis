clear
for iter = 1:30
    %load 'data/dataset';'    
    rng('shuffle', 'twister');
    load '../data/test1';
    load (['../saved_variables/classify_weights_c_5_' num2str(iter)])

    %targets = labels_test(:,2:8);
    N = 10080;
    randomorder=randperm(N);
    
    data = features_test(randomorder, :);
    %targets_temp = str2num(labels_test(:,4));
    targets = labels_test(randomorder, :);

    data = [data ones(N,1)];
    w1probs = 1./(1 + exp(-data*w1));
    w1probs = [w1probs ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2));
    w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*w3));
    w3probs = [w3probs ones(N,1)];
    w4probs = 1./(1 + exp(-w3probs*w4));
    w4probs = [w4probs ones(N,1)];
    w5probs = 1./(1 + exp(-w4probs*w5));
    w5probs = [w5probs ones(N,1)];

    targetout = exp(w5probs * w_class);
    targetout = targetout./repmat(sum(targetout,2),1,7);

    [I, J]=max(targetout,[],2);
    [I1, J1]=max(targets,[],2);
    counter=length(find(I1==J));
    correct = zeros(N, 1);
    correct(1) = counter;
    T = table(I1, J, correct);
    T.Properties.VariableNames = {'Real' 'Predicted' 'Correct'};
    writetable(T, ['../classification_results/c_5_' num2str(iter) '.csv']);
    clear;
end