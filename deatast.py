from __future__ import division
import sample_operator
import time
import random
from imblearn.datasets import fetch_datasets
from sklearn import datasets
import numpy as np
import pandas as pd
def label_convert(label_set):
    a = -1
    for i in range(len(label_set)):
        if label_set[i] == 1:
            label_set[i] = a
        elif label_set[i] == 0:
            label_set[i] = 1
    return label_set
ecoli1, ecoli1_label = sample_operator.load_data_mat('Data/ecoli1.data',
                                                     'Data/ecoli1_label.data')
ecoli2, ecoli2_label = sample_operator.load_data_mat('Data/ecoli2.data',
                                                     'Data/ecoli2_label.data')
dermatology_6, dermatology_6_label = sample_operator.load_data_mat('Data/dermatology_6.data',
                                                     'Data/dermatology_6_label.data')

data = {'ecoli1': ecoli1, 'ecoli2': ecoli2, 'dermatology_6':dermatology_6}

data_label = {'ecoli1': ecoli1_label, 'ecoli2': ecoli2_label, 'dermatology_6': dermatology_6_label}
def get_dataset():
    return data, data_label
