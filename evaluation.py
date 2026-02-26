#-*- coding: utf-8 -*
from numpy import *
import numpy as np
from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
def evaluate_prediction(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	#precision = float(tp)/float(tp+fp)

	recall = float(tp)/float(tp+fn)
	TNR = float(tn)/float(tn+fp)
	G_mean = sqrt(recall * TNR)
	return G_mean

def g_mean(test_label, predict_label):
	g_mean = 1.0
	for label in np.unique(test_label):
		idx = (test_label == label)
		g_mean *= metrics.accuracy_score(test_label[idx], predict_label[idx])

	g_mean = np.sqrt(g_mean)
	print ("g_mean", g_mean)
