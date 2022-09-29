for iter = 1:30 
    %load 'data/dataset';
    rng('shuffle', 'twister');
    load '../data/test';
    load (['../saved_variables/classify_weightsc1_' num2str(iter)])

    %targets = labels_test(:,2:8);
    N = 10080;
    randomorder=randperm(N);
    
    data = features(randomorder, :);
    targets_temp = str2num(labels(:,4));
    targets = targets_temp(randomorder, :);

    data = [data ones(N,1)];
    w1probs = 1./(1 + exp(-data*w1));
    w1probs = [w1probs ones(N,1)];

    targetout = exp(w1probs * w_class);
    targetout = targetout./repmat(sum(targetout,2),1,7);

    [I, J]=max(targetout,[],2);
    [I1, J1]=max(targets,[],2);
    counter=length(find(J==J1));
    correct = zeros(N, 1);
    correct(1) = counter;
    T = table(J1, J, correct);
    T.Properties.VariableNames = {'Real' 'Predicted' 'Correct'};
    writetable(T, ['classification_results/c1_' num2str(iter) '.csv']);
    clear;
end