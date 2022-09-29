from __future__ import print_function, division

import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
import seaborn as sns
import pandas as pd

def plot_multiclass_roc(y_score, X_test, y_test, n_classes, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    total_auc = 0
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        total_auc += roc_auc[i]
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %i' % (roc_auc[i], i + 1))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    average_auc = total_auc / n_classes
    print("Average AUC: ", average_auc)
    return average_auc

data = hdf5storage.loadmat('dataset.mat')
X_train = np.array(data['features_training'])
train_labels = np.array(data['labels_training'], dtype=np.int8)
y_train = np.where(train_labels == 1)[1]
X_test = np.array(data['features_test'])
test_labels = np.array(data['labels_test'], dtype=np.int8)
y_test = np.where(test_labels == 1)[1]

if __name__ == '__main__':
    pca = PCA(n_components=20)
    results = []
    auc_results = []
    for x in range(30):
        X_train_new = pca.fit_transform(X_train)
        X_test_new = pca.transform(X_test)
        model = RandomForestClassifier(n_estimators = 100, verbose=True)
        model.fit(X_train_new, y_train)
        results.append(model.score(X_test_new, y_test))
        predictions = model.predict(X_test_new)
        predictions_score = model.predict_proba(X_test_new)
        print(y_test.ravel())
        average_auc = plot_multiclass_roc(predictions_score, X_test_new, y_test, n_classes=7, figsize=(16, 10))
        auc_results.append(average_auc)
    print(sum(results)/len(results))
    with open('accuracy.txt', 'w') as file_handler:
        file_handler.write("\n".join(str(item) for item in results))
    with open('auc.txt', 'w') as file_handler:
        file_handler.write("\n".join(str(item) for item in auc_results))