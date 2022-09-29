from __future__ import print_function, division

import hdf5storage
import numpy as np
from sklearn import preprocessing #if necessary
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support, classification_report
#from sklearn.metrics import precision_recall_fscore_support
from joblib import parallel_backend
import matplotlib.pyplot as plt
import pandas as pd
import confusion_matrix_print as cfp
import os

if __name__ == '__main__':
    configuration = input('c0-c5 or f1-f5:\n')
    count = 0
    confmat = np.zeros((7, 7), dtype=int)
    #for file in os.listdir("../saved_features"):
    #    if file.startswith("feat" + configuration + "L"):
    #        print(os.path.join("../saved_features", file))
    results_complete = []
    with parallel_backend('loky', n_jobs=-1):
        for file in [f for f in os.listdir('../classification_results2') if f.startswith(configuration)]:
            print(file)            
            data = np.loadtxt(os.path.join("../classification_results2", file),delimiter=',', skiprows=1, dtype=int)
            print(data)

            results = []

            count+=1
            results.append(data[0, 2]/10000)
            print(str(data[0, 2]/10000))
            CM = confusion_matrix(data[:, 0], data[:, 1])
            recall = np.diag(CM) / np.sum(CM, axis = 1)
            precision = np.diag(CM) / np.sum(CM, axis = 0)
            confmat += CM
            #print(precision)
            #print(recall)                

            #print(CM)

            print(classification_report(data[:, 0], data[:, 1], digits=3, zero_division=0))
            #print(precision_recall_fscore_support(y_test.ravel(), predictions, average='macro'))
            #print(precision_recall_fscore_support(y_test.ravel(), predictions, average='micro'))
            #print(precision_recall_fscore_support(y_test.ravel(), predictions, average='weighted'))

            #disp = plot_confusion_matrix(model, X_test, y_test.ravel(), cmap=plt.cm.Blues, normalize=None)
            #plt.show() 

            print(sum(results)/len(results))
            #print(confmat)
            results_complete.append(sum(results)/len(results))
            #with open('features002V1.txt', 'w') as file_handler:
            #with open('result_rfc_' + str(y+1) + '.txt', 'w') as file_handler:
            #    file_handler.write("\n".join(str(item) for item in results))
        print(results_complete)
        print(count)
        average_cm = pd.DataFrame(confmat/count, columns=['1', '2', '3', '4', '5', '6', '7'], index=['1', '2', '3', '4', '5', '6', '7'])
        average_recall = np.diag(average_cm) / np.sum(average_cm, axis = 1)
        average_precision = np.diag(average_cm) / np.sum(average_cm, axis = 0)
        rounded_asd = average_cm.round(2)
        #print(asd)
        cfp.pretty_plot_confusion_matrix(rounded_asd, cmap='YlGnBu', pred_val_axis='x')
        print(np.average(average_precision))
        print(np.average(average_recall))
        