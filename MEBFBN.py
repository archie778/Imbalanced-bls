#-*- coding: utf-8 -*
from __future__ import division

import csv
from numpy import *
import numpy as np
from sklearn import preprocessing, metrics
from numpy import random
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import skimage.io as io
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import datasets_0
import time
import evaluation
import sample_operator
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class FuzzyInferenceFeatureGenerator:

    
    def __init__(self, num_fuzzy_rules=15, cluster_method='kmeans', std=1.0, random_state=42):
        self.num_fuzzy_rules = num_fuzzy_rules
        self.cluster_method = cluster_method
        self.std = std
        self.random_state = random_state

        self.centers = None
        self.alpha = None
        self.is_fitted = False
        
    def _generate_fuzzy_centers(self, X):
        np.random.seed(self.random_state)
        
        if self.cluster_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.num_fuzzy_rules, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            centers = kmeans.cluster_centers_
        elif self.cluster_method == 'random':
            indices = np.random.choice(X.shape[0], self.num_fuzzy_rules, replace=False)
            centers = X[indices, :]
        else:
            raise ValueError(f"Unsupported cluster method: {self.cluster_method}")
            
        return centers
    
    def _calculate_fuzzy_membership(self, X, centers):
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.num_fuzzy_rules))
        
        for i in range(n_samples):
            x_sample = X[i, :]
            
            distances = np.sum((x_sample - centers) ** 2, axis=1)
            mf = np.exp(-distances / self.std)
            
            mf_sum = np.sum(mf)
            if mf_sum > 1e-10:
                mf = mf / mf_sum
            else:
                mf = np.ones(self.num_fuzzy_rules) / self.num_fuzzy_rules
                
            membership[i, :] = mf
            
        return membership
    
    def _generate_fuzzy_features(self, X, membership, alpha):
        n_samples = X.shape[0]
        
        F_features = np.zeros((n_samples, self.num_fuzzy_rules))
        
        for i in range(n_samples):
            x_alpha = np.dot(X[i, :], alpha)
            F_features[i, :] = membership[i, :] * x_alpha
        
        X_alpha = np.dot(X, alpha)
        G_features = membership * X_alpha
        
        return F_features, G_features
    
    def fit(self, X):

        self.centers = self._generate_fuzzy_centers(X)
        
        np.random.seed(self.random_state)
        self.alpha = np.random.rand(X.shape[1], self.num_fuzzy_rules)
        
        self.is_fitted = True
        
        print(f"Fuzzy inference feature generator training completed: number of rules={self.num_fuzzy_rules}, cluster method={self.cluster_method}")
        
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Please call the fit method to train the fuzzy inference feature generator")
        
        membership = self._calculate_fuzzy_membership(X, self.centers)
        
        F_features, G_features = self._generate_fuzzy_features(X, membership, self.alpha)
        
        fuzzy_features = np.concatenate([F_features, G_features], axis=1)
        
        return fuzzy_features
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def show_accuracy(predictLabel,Label):
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    return (round(count/len(Label),5))

class node_generator:
    def __init__(self,whiten = False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten
    
    def sigmoid(self,data):
        return 1.0/(1+np.exp(-data))
    
    def linear(self,data):
        return data
    
    def tanh(self,data):
        return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    
    def relu(self,data):
        return np.maximum(data,0)
    
    def orth(self,W):
        for i in range(0,W.shape[1]):
            w = np.mat(W[:,i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:,j].copy()).T
                w_sum += (w.T.dot(wj))[0,0]*wj 
            w -= w_sum
            w = w/np.sqrt(w.T.dot(w))
            W[:,i] = np.ravel(w)
        return W
        
    def generator(self,shape,times):
        for i in range(times):
            W = 2*random.random(size=shape)-1
            if self.whiten == True:
                W = self.orth(W)
            b = 2*random.random()-1
            yield (W,b)
    
    def generator_nodes(self, data, times, batchsize, nonlinear):
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1],batchsize),times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1],batchsize),times)]
        self.nonlinear = {'linear':self.linear,
                          'sigmoid':self.sigmoid,
                          'tanh':self.tanh,
                          'relu':self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i])+self.blist[i])))
        return nodes
        
    def transform(self,testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0])+self.blist[0])
        for i in range(1,len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i])+self.blist[i])))
        return testnodes   

    def update(self,otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb
        
class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0
    
    def fit_transform(self,traindata):
        self._mean = traindata.mean(axis = 0)
        self._std = traindata.std(axis = 0)
        return (traindata-self._mean)/(self._std+0.001)
    
    def transform(self,testdata):
        return (testdata-self._mean)/(self._std+0.001)
        

class broadnet:
    def __init__(self, 
                 maptimes = 10, 
                 enhencetimes = 10,
                 map_function = 'linear',
                 enhence_function = 'linear',
                 batchsize = 'auto', 
                 reg = 0.001,
                 use_fuzzy_features = True,
                 num_fuzzy_rules = 15,
                 fuzzy_cluster_method = 'kmeans',
                 num_layers = 7,  
                 num_nodes = 800):  
        
        self._maptimes = maptimes
        self._enhencetimes = enhencetimes
        self._batchsize = batchsize
        self._reg = reg
        self._map_function = map_function
        self._enhence_function = enhence_function
        
        self._use_fuzzy_features = use_fuzzy_features
        self._num_fuzzy_rules = num_fuzzy_rules
        self._fuzzy_cluster_method = fuzzy_cluster_method
        
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        
        self.W_layers = []  # 每层的最终权重
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse = False,categories='auto')
        
        self.mapping_generators = []
        self.enhence_generators = []
        
        for layer in range(self._num_layers):
            self.mapping_generators.append(node_generator())
            self.enhence_generators.append(node_generator(whiten = True))
        
        if self._use_fuzzy_features:
            self.fuzzy_generator = FuzzyInferenceFeatureGenerator(
                num_fuzzy_rules=self._num_fuzzy_rules,
                cluster_method=self._fuzzy_cluster_method,
                random_state=42
            )

    def fit(self,data,label):
        data = self.normalscaler.fit_transform(data)
        n_samples, n_features = data.shape
        
        unique_labels = np.unique(label)
        class_counts = {l: np.sum(label == l) for l in unique_labels}
        total_samples = len(label)
        self._class_weights = {l: total_samples / (len(unique_labels) * count) for l, count in class_counts.items()}
        self._sample_class_weights = np.array([self._class_weights[l] for l in label])
        print(f"Class weights: {self._class_weights}")
        
        if self._use_fuzzy_features:
            fuzzy_features = self.fuzzy_generator.fit_transform(data)
            original_features = data.shape[1]
            fuzzy_features_count = fuzzy_features.shape[1]
            print(f"Fuzzy inference feature enhancement: original features{original_features} + fuzzy features{fuzzy_features_count} = total features{data.shape[1] + fuzzy_features_count}")
        else:
            fuzzy_features = np.array([]).reshape(n_samples, 0)
        

        if self._batchsize == 'auto':
            self._batchsize = data.shape[1] + fuzzy_features.shape[1]
        
        label = self.onehotencoder.fit_transform(np.mat(label).T)
        
        self.W_layers = []
        predictions = []
        
        current_input = np.hstack([data, fuzzy_features]) if self._use_fuzzy_features else data
        
        print(f"Start training {self._num_layers} layer integrated network...")
        
        for layer in range(self._num_layers):
            mappingdata = self.mapping_generators[layer].generator_nodes(
                current_input, self._maptimes, self._batchsize, self._map_function)
            
            enhencedata = self.enhence_generators[layer].generator_nodes(
                mappingdata, self._enhencetimes, self._batchsize, self._enhence_function)
            
            inputdata = np.column_stack((mappingdata, enhencedata))
            
            pesuedoinverse = self.pinv(inputdata, self._reg)
            W_initial = pesuedoinverse.dot(label)
            predictions_initial = inputdata.dot(W_initial)
            
            num_samples = inputdata.shape[0]
            weights = np.zeros(num_samples)
            epsilon = 1e-8
            
            for i in range(num_samples):
                y_i = label[i, :]
                y_hat_i = predictions_initial[i, :]
                
                error_norm = np.linalg.norm(y_i - y_hat_i, ord=1)
                class_weight = self._sample_class_weights[i]
                error_weight = 1.0 + error_norm
                
                weights[i] = class_weight * error_weight
            
            W_matrix = np.diag(weights)

            A_T = inputdata.T
            term_to_invert_mat = np.mat(self._reg * np.eye(inputdata.shape[1]) + A_T.dot(W_matrix).dot(inputdata))
            A_T_W_Y = A_T.dot(W_matrix).dot(label)
            W_final = term_to_invert_mat.I.dot(A_T_W_Y)
            
            self.W_layers.append(W_final)
            
            layer_prediction = inputdata.dot(W_final)
            layer_pred_class = self.decode(layer_prediction)
            predictions.append(layer_pred_class)
            if layer < self._num_layers - 1:
                if self._use_fuzzy_features:
                    current_input = np.hstack([data, fuzzy_features, mappingdata])
                else:
                    current_input = np.hstack([data, mappingdata])
        
        self.train_predictions = np.array(predictions).T

    def pinv(self,A,reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

    def decode(self,Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i,:]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)
    
    def accuracy(self,predictlabel,label):
        label = np.ravel(label).tolist()
        predictlabel = predictlabel.tolist()
        count = 0
        for i in range(len(label)):
            if label[i] == predictlabel[i]:
                count += 1
        return (round(count/len(label),5))
        
    def predict(self,testdata):
        testdata = self.normalscaler.transform(testdata)
        n_samples = testdata.shape[0]
        
        if self._use_fuzzy_features:
            fuzzy_features = self.fuzzy_generator.transform(testdata)
            print(f"Fuzzy feature enhancement of test data: {testdata.shape[1]} + {fuzzy_features.shape[1]} = {testdata.shape[1] + fuzzy_features.shape[1]}")
        else:
            fuzzy_features = np.array([]).reshape(n_samples, 0)
        
        predictions = []
        
        current_input = np.hstack([testdata, fuzzy_features]) if self._use_fuzzy_features else testdata
        
        for layer in range(self._num_layers):
            test_mappingdata = self.mapping_generators[layer].transform(current_input)
            
            test_enhencedata = self.enhence_generators[layer].transform(test_mappingdata)
            
            test_inputdata = np.column_stack((test_mappingdata, test_enhencedata))
            
            layer_prediction = test_inputdata.dot(self.W_layers[layer])
            layer_pred_class = self.decode(layer_prediction)
            predictions.append(layer_pred_class)
            
            if layer < self._num_layers - 1:
                if self._use_fuzzy_features:
                    current_input = np.hstack([testdata, fuzzy_features, test_mappingdata])
                else:
                    current_input = np.hstack([testdata, test_mappingdata])
        
        predictions = np.array(predictions).T
        final_predictions = []
        
        for i in range(n_samples):
            unique, counts = np.unique(predictions[i], return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        return np.array(final_predictions)

def main():

    dataset = ["dermatology_6"]
    data, data_label = datasets_0.get_dataset()

    run_times = 5
    cv_num = 5 

    for dataname in dataset:
        print ('%s:  %s' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           '==============================Dataset-%s-begin============================' % dataname))

        result_auc_set_1 = []
        result_f1_set_1 = []
        result_gmean_set_1 = []
        for i in range(run_times):
            cv_data, cv_data_label = sample_operator.generate_cross_valitation_dataset(data[dataname],
                                                                                       data_label[dataname], cv_num)
            bestscore=0
            auc_set_1 = []
            f1_set_1 = []
            gmean_set_1 = []
            for j in range(cv_num):
                train_data, train_label, test_data, test_label = sample_operator.generate_train_test_dataset(cv_data, cv_data_label, j, cv_num)

                sc = StandardScaler()
                train_data = sc.fit_transform(train_data)
                test_data = sc.transform(test_data)

                print ("train_data",train_data.shape)
                bls = broadnet(maptimes=15, enhencetimes=15, map_function='linear', enhence_function='sigmoid', batchsize=80,
                               reg=0.001, use_fuzzy_features=True, num_fuzzy_rules=20, fuzzy_cluster_method='kmeans',
                               num_layers=3, num_nodes=810)
                print('train_label.shape',train_label.shape)

                bls.fit(train_data, train_label)
                predict_label = bls.predict(test_data)

                print("predict_label", predict_label)
                for i in range(len(test_label)):
                    if predict_label[i] == 0:
                        predict_label[i] = -1
                auc_1 = roc_auc_score(test_label, predict_label)
                f1_value = metrics.f1_score(test_label, predict_label)
                gmean = evaluation.evaluate_prediction(test_label, predict_label)
                f1_set_1.append(f1_value)
                gmean_set_1.append(gmean)
                auc_set_1.append(auc_1)

            result_auc_set_1.append(mean(auc_set_1))
            result_f1_set_1.append(mean(f1_set_1))
            result_gmean_set_1.append(mean(gmean_set_1))
        print ("the mean auc testing of ME-BFBN:", mean(result_auc_set_1))
        print("the mean gmean testing of ME-BFBN:", mean(result_gmean_set_1))
        print("the mean f1 testing of ME-BFBN:", mean(result_f1_set_1))

if __name__ =="__main__":
    main()
  
