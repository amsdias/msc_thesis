from __future__ import print_function, division

import hdf5storage
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from joblib import parallel_backend
import matplotlib.pyplot as plt
import pandas as pd
import confusion_matrix_print as cfp
import os
import seaborn as sns


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


def plot_multiclass_roc(y_score, X_test, y_test, n_classes, model, count, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    with open('fpr_tpr_' + str(model) + '_' + str(count) + '.txt', 'w') as f:
        f.write("FPR values\n")
        for key, value in fpr.items():
            f.write('\n%s: ' % (key + 1))
            f.write(to_str(value))

        f.write("\n\nTPR values\n")
        for key, value in tpr.items():
            f.write('\n%s: ' % (key + 1))
            f.write(to_str(value))

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %i' % (roc_auc[i], i + 1))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


def plot_multiclass_roc_simple(y_score, X_test, y_test, n_classes, figsize=(17, 6), model=''):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_array = np.empty((7))

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve for class %i (AUC = %0.3f)' % (i + 1, roc_auc[i]))
        auc_array[i] = roc_auc[i]
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

    print(auc_array)
    return auc_array

if __name__ == '__main__':
    model = RandomForestClassifier(n_estimators=18, verbose=False)
    configuration = input('c1-c5 or f1-f5:\n')
    type = input('L or NL:\n')
    if type == "NL":
        linear = "_NL"
    else:
        linear = ""
    confmat = np.zeros((7, 7), dtype=int)

    results_complete = []
    roc_auc_complete = []    
    # structures
    fpr_all = np.zeros((7, 30, 30))
    tpr_all = np.zeros((7, 30, 30))
    roc_auc_all = np.zeros((30, 7))
    count = 0
    print("feat" + "_" + configuration + linear)
    with parallel_backend('loky', n_jobs=-1):
        for file in [f for f in os.listdir('final_saved_features') if f.startswith("feat" + '_' + configuration + linear + '_')]:
            print('File: ' + file)
            trdata = hdf5storage.loadmat(
                os.path.join("final_saved_features", file))

            X_train = np.array(trdata['training_features'])
            y_train = np.array(trdata['training_labels'], dtype=np.int8)

            X_test = np.array(trdata['test_features'])
            y_test = np.array(trdata['test_labels'], dtype=np.int8)
            count += 1
            results = []

            for x in range(1):
                model.fit(X_train, y_train.ravel())
                score = model.score(X_test, y_test.ravel())

                results.append(score)
                predictions = model.predict(X_test)
                predictions_score = model.predict_proba(X_test)

                # structures
                fpr_temp = dict()
                tpr_temp = dict()
                roc_auc_temp = dict()

                # calculate dummies once
                y_test_dummies = pd.get_dummies(y_test.ravel(), drop_first=False).values
                for i in range(7):
                    fpr_temp[i], tpr_temp[i], _ = roc_curve(y_test_dummies[:, i], predictions_score[:, i])
                    fpr1 = np.array(fpr_temp[i])
                    tpr1 = np.array(tpr_temp[i])
                    fpr_all[i, count - 1, :fpr1.shape[0]] = fpr1
                    tpr_all[i, count - 1, :tpr1.shape[0]] = tpr1
                    roc_auc_all[count - 1, i] = auc(np.trim_zeros(fpr_all[i, count - 1], 'b'), np.trim_zeros(tpr_all[i, count - 1], 'b'))

                roc_auc_complete.append(roc_auc_score(y_test.ravel(), predictions_score, multi_class='ovr'))

                CM = confusion_matrix(y_test.ravel(), predictions)
                recall = np.diag(CM) / np.sum(CM, axis=1)
                precision = np.diag(CM) / np.sum(CM, axis=0)
                confmat += CM

            results_complete.append(sum(results) / len(results))
        results_complete.append('#ea#\nAverage Accuracy: ' + str(sum(results_complete) / len(results_complete)))
        roc_auc_complete.append('#er#\nAverage ROC_AUC: ' + str(sum(roc_auc_complete) / len(roc_auc_complete)))
        average_cm = pd.DataFrame(confmat / count, columns=['1', '2', '3', '4', '5', '6', '7'], index=['1', '2', '3', '4', '5', '6', '7'])
        average_recall = np.diag(average_cm) / np.sum(average_cm, axis=1)
        average_precision = np.diag(average_cm) / np.sum(average_cm, axis=0)
        rounded_average_cm = average_cm.round(2)

        with open(os.path.join('reports', 'result_rfc_' + configuration + linear + '.txt'), 'w') as file_handler:
            file_handler.write('Accuracy:\n#sa#\n')
            file_handler.write("\n".join(str(item) for item in results_complete))
            file_handler.write('\n\nROC_AUC:\n#sr#\n')
            file_handler.write("\n".join(str(item) for item in roc_auc_complete))
            file_handler.write("\n\nAverage Precision: " + str(np.average(average_precision)))
            file_handler.write("\n\nAverage Recall: " + str(np.average(average_recall)))

        print('Count: ' + str(count))

        cfp.pretty_plot_confusion_matrix(rounded_average_cm, cmap='YlGnBu', pred_val_axis='x', filename='rfc_' + configuration)
        print("Average Precision: " + str(np.average(average_precision)))
        print("Average Recall: " + str(np.average(average_recall)))
