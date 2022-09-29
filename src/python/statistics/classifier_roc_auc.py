from __future__ import print_function, division

import hdf5storage
import numpy as np
from sklearn import preprocessing #if necessary
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from joblib import parallel_backend
import matplotlib.pyplot as plt
import pandas as pd
import confusion_matrix_print as cfp
import os
import class_report as cr
# from yellowbrick.classifier import ROCAUC
import seaborn as sns


# def plot_multiclass_roc(y_score, X_test, y_test, n_classes, figsize=(17, 6)):
def plot_multiclass_roc(y_score, y_test, n_classes, figsize=(17, 6)):
    # y_score = model.score(X_test, y_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], thresh = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(thresh)
    print(fpr[1])

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


if __name__ == '__main__':
    configuration = input('c1-c5 or f1-f5:\n')
    confmat = np.zeros((7, 7), dtype=int)
    n_classes = 7
    results_complete = []
    roc_auc_complete = []
    accuracy = []
    fpr = []
    tpr = []
    count = 0
    print("full" + configuration + '_')
    with parallel_backend('loky', n_jobs=-1):
        for file in [f for f in os.listdir('final_classification_results') if f.startswith("full_" + configuration + '_')]:
            print('File: ' + file)
            data = pd.read_csv(os.path.join('final_classification_results', file))
            # print(data.at[0, 'Correct'])
            y_test = np.array(data['Real'], dtype=np.int8)
            y_predicted = np.array(data['Predicted'], dtype=np.int8)
            y_score = np.array(data.loc[:, 'P1':'P7'], dtype=np.float32)
            accuracy.append(data.at[0, 'Correct'])
            results_complete.append(data.at[0, 'Correct'] / y_test.shape[0])
            roc_auc_complete.append(roc_auc_score(y_test, y_score, multi_class='ovr'))
            # print(y_test)
            # print(y_score)
            # print(accuracy)

            # plot_multiclass_roc(y_score, y_test, n_classes=n_classes, figsize=(16, 10))
            CM = confusion_matrix(y_test, y_predicted)
            confmat += CM

            count += 1
            # results = []
            # for x in range(1):
                # print(predictions_score)
                # plot_ROC_curve(model, X_train, y_train.ravel(), X_test, y_test.ravel())

                # plot_multiclass_roc(predictions_score, X_test, y_test.ravel(), n_classes=7, figsize=(16, 10))
                # print(roc_auc_score(y_test.ravel(), predictions_score, multi_class='ovr'))
                # cr1 = cr.class_report(y_true=y_test.ravel(),y_pred=predictions,y_score=predictions_score)
                # print(cr1)
                # CM = confusion_matrix(y_test.ravel(), predictions)
                # recall = np.diag(CM) / np.sum(CM, axis=1)
                # precision = np.diag(CM) / np.sum(CM, axis=0)
                # confmat += CM

                # print(classification_report(y_test.ravel(),
                #    predictions, digits=3))

            # print(sum(results) / len(results))

            # results_complete.append(sum(results) / len(results))        
        results_complete.append('#ea#\nAverage: ' + str(sum(results_complete) / len(results_complete)))
        roc_auc_complete.append('#er#\nAverage ROC_AUC: ' + str(sum(roc_auc_complete) / len(roc_auc_complete)))
        average_cm = pd.DataFrame(confmat / count, columns=['1', '2', '3', '4', '5', '6', '7'], index=['1', '2', '3', '4', '5', '6', '7'])
        average_recall = np.diag(average_cm) / np.sum(average_cm, axis=1)
        average_precision = np.diag(average_cm) / np.sum(average_cm, axis=0)
        rounded_average_cm = average_cm.round(2)
        with open(os.path.join('reports2', 'result_classifier_' + configuration + '.txt'), 'w') as file_handler:
            file_handler.write('Accuracy:\n#sa#\n')
            file_handler.write("\n".join(str(item) for item in results_complete))
            file_handler.write('\n\nROC_AUC:\n#sr#\n')
            file_handler.write("\n".join(str(item) for item in roc_auc_complete))
            file_handler.write("\n\nAverage Precision: " + str(np.average(average_precision)))
            file_handler.write("\n\nAverage Recall: " + str(np.average(average_recall)))
            file_handler.write("\n\novr")

        # print(results_complete)
        print('Count: ' + str(count))

        cfp.pretty_plot_confusion_matrix(rounded_average_cm, cmap='YlGnBu', pred_val_axis='x', filename='classifier_' + configuration)
        print("Average Precision: " + str(np.average(average_precision)))
        print("Average Recall: " + str(np.average(average_recall)))
