import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time

from interface import Interface

if __name__=='__main__':
    interface=Interface()

    #accuracy = interface.predict('./train_list.txt', assigned_modelName=None)
    #print('Weighted F1 score: %.6lf' % accuracy)

    #tot_correct_cnt, tot_cnt_table, f1_score=interface.predict('./train_list.txt', assigned_modelName=None)
    accuracy=interface.predict('./dev_list.txt', assigned_modelName=None)
    print('Accuracy: %.6lf'%accuracy)

    #accuracy=interface.predict('./test_list.txt', assigned_modelName=None)
    #print('Accuracy: %.6lf'%accuracy)
    