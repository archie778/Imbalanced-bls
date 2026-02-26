# coding=utf-8
'''
Created on 11/04/2017

@author: ykx
'''

from numpy import *
from sklearn import tree
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

def new_load_data_mat(inputfile, inputfile_label):
    arr = np.loadtxt(inputfile)
    lab = np.loadtxt(inputfile_label)
    """
    for i in range(len(lab)):     
        if lab[i] ==1:
            lab[i] = -1
        if lab[i] ==0:
            lab[i] = 1
    print "modified lab",lab
    np.savetxt("fertility_label.data", lab)
    """
    return array(arr), array(lab)

def get_selcted_sample(initial_data, selected_index):
    new_sample = []
    data = array(initial_data)
    #print "the length of initial_data:",len(initial_data)
    #print "the length of selected_index:", len(selected_index)
    #print "selected index:",selected_index
    for i in range(len(selected_index)):
        #print i
        if selected_index[i] == 1:
            new_sample.append(data[i].tolist())
    return new_sample

def get_index_sample(initial_data, selected_index):
    Newdata_List = []
    Newdata_Set = []
    data = array(initial_data)
    #print selected_index
    #initial_majority_data = array(initial_majority_data).T
    for i in selected_index:
        Newdata_List.append(data[i])
    Newdata_Set = np.array(Newdata_List)
    #print Newdata_Set
    return Newdata_Set

def get_feature_index_sample(data, index):
    newdata = []
    data = array(data).T
    for i in index:
        newdata.append(data[i].tolist())
    return mat(newdata).T

def get_balanced_training_data(marjority_data, minority_data):
    new_balanced_list = []
    new_balanced_label_list = []
    new_balanced_set = []
    new_balanced_label_set = []
    mar_len = len(marjority_data)
    min_len = len(minority_data)
    marjority_data = marjority_data.tolist()
    minority_data = minority_data.tolist() 
    for i in range(mar_len):
        new_balanced_list.append(marjority_data[i])
        new_balanced_label_list.append(1)

    for j in range(min_len):
        new_balanced_list.append(minority_data[j])
        new_balanced_label_list.append(-1)

    new_balanced_set = np.array(new_balanced_list)
    new_balanced_label_set = np.array(new_balanced_label_list)
    return new_balanced_set,new_balanced_label_set

def get_balanced_training_set(marjority_data, minority_data):
    new_balanced_list = []
    new_balanced_label_list = []
    new_balanced_set = []
    new_balanced_label_set = []
    mar_len = len(marjority_data)
    min_len = len(minority_data)
    minority_data = minority_data.tolist() 
    for i in range(mar_len):
        new_balanced_list.append(marjority_data[i])
        new_balanced_label_list.append(1)

    for j in range(min_len):
        new_balanced_list.append(minority_data[j])
        new_balanced_label_list.append(-1)

    new_balanced_set = np.array(new_balanced_list)
    new_balanced_label_set = np.array(new_balanced_label_list)
    clf = KNN(n_neighbors=3).fit(new_balanced_set, new_balanced_label_set)
    return clf

def generate_cross_valitation_dataset(sample, label, cv_num):
    N, D = sample.shape
    postive_index = []
    negative_index = []
    for i in range(N):
        if label[i] == 1:
            postive_index.append(i)
        else:
            negative_index.append(i)
    postive_sample_num = len(postive_index)
    negative_sample_num = len(negative_index)
    postive_cv_num = int(postive_sample_num / cv_num)
    negative_cv_num = int(negative_sample_num / cv_num)
    dataset = []
    dataset_label = []
    random.shuffle(postive_index)
    random.shuffle(negative_index)
    for j in range(cv_num):
        x = []
        if j == cv_num - 1:
            for k in postive_index[j * postive_cv_num: postive_sample_num]:
                x.append(k)
            for k in negative_index[j * negative_cv_num: negative_sample_num]:
                x.append(k)
        else:
            x = [k for k in postive_index[j * postive_cv_num: (j+1) * postive_cv_num]]
            x.extend([k for k in negative_index[j * negative_cv_num: (j+1) * negative_cv_num]])
        data_tmp = []
        data_label_tmp = []
        for i in x:
            data_tmp.append(sample[i])
            data_label_tmp.append(label[i])
        dataset.append(array(data_tmp))
        dataset_label.append(array(data_label_tmp))
    return dataset, dataset_label

    

def generate_train_test_dataset(set, label, i, cv_num):
    train_data = []
    train_data_label = []
    test_data = []
    test_data_label = []
    for j in range(cv_num):
        set_j_list = set[j].tolist()
        label_j_list = label[j].tolist()
        if j == i:
            for k in range(len(set[j])):
                test_data.append(set_j_list[k])
                test_data_label.append(label_j_list[k])
        else:
            for k in range(len(set[j])):
                train_data.append(set_j_list[k])
                train_data_label.append(label_j_list[k])
    return array(train_data), array(train_data_label), \
           array(test_data), array(test_data_label)

def load_data_mat(inputfile, inputfile_label):
    fr = open(inputfile)
    fr_label = open(inputfile_label)
    datalines = fr.readlines()
    arr = []
    lab = []
    for line in datalines:
        line = line.strip()
        listfromline = line.split(',')
        newline = []
        for i in range(0, len(listfromline)):
            newline.append(float(listfromline[i]))
        arr.append(newline)

    datalines_label = fr_label.readlines()
    for line in datalines_label:
        listfromline = line.strip()
        lab.append(int(listfromline))

    return array(arr), array(lab)


def load_img_mat(inputfile, inputfile_label):
    fr = open(inputfile)
    fr_label = open(inputfile_label)
    datalines = fr.readlines()
    arr = []
    lab = []
    for line in datalines:
        line = line.strip()
        listfromline = line.strip( ).split( )
        newline = []
        for i in range(0, len(listfromline)):
            newline.append(float(listfromline[i]))
        arr.append(newline)

    datalines_label = fr_label.readlines()
    for line in datalines_label:
        listfromline = line.strip()
        lab.append((listfromline))

    return array(arr), array(lab)

def get_dataset(data):
    fr = open(data)
    fw_1 = open('%s.data' % data.replace('.dat', ''), 'w')
    fw_2 = open('%s_label.data' % data.replace('.dat', ''), 'w')
    lines = fr.readlines()
    fr.close()
    for line in lines:
        line = line.strip().split(', ')
        str = ",".join(line[0: -1])
        fw_1.write(str+'\n')
        if line[len(line)-1] == 'negative':
            fw_2.write('1\n')
        else:
            fw_2.write('-1\n')
    fw_1.close()
    fw_2.close()


def autoNorm(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(data))
    m = data.shape[0]
    normDataSet = data - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet

def dataNorm(list):
    value = 0
    newlist = []
    for i in list:
        value += i
    for i in list:
        try:
            newlist.append(round(float(i) / float(value), 3))
        except Exception:
            newlist.append(0)

    return newlist

def mat2Norm(data):
    N, D = data.shape
    data_copy = data.T.tolist()
    for i in range(D):
        max_num = max(data_copy[i])
        for j in range(N):
            data_copy[i][j] = float(data_copy[i][j]) / max_num
    return mat(data_copy).T
