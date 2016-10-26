import numpy as np
from numpy.linalg import inv
import math


#get mean and std for each feature
def get_mean_std(dataset, feature_id):
    mean = np.mean(dataset[:, feature_id], axis=0)
    std = np.std(dataset[:, feature_id], axis=0)
    return mean, std

#probability for each feature for each observation
def prob_gaussian(x, mean, std):
    return np.exp(-((x - mean)**2) / (2 * (std**2))) * (1 / (np.sqrt(2 * math.pi) * std))
    
def Naive_bayes_likelyhood(X_features, means, stds):
    prob = 1.
    for i in range(len(X_features)):
        prob *= prob_gaussian(X_features[i], means[i], stds[i])
    return prob
    
def Naive_bayes_prediction(X_features, means1, stds1, means2, stds2, prior1, prior2):
    prob_class1 = Naive_bayes_likelyhood(X_features, means1, stds1) * prior1
    prob_class2 = Naive_bayes_likelyhood(X_features, means2, stds2) * prior2
    if prob_class1 > prob_class2:
        return 0
    elif prob_class1 < prob_class2:
        return 1
    else:
        return None


data = np.loadtxt('hw2.txt', delimiter=',')
accuracys = []
for trial in range(10):
    #Re-order the dataset
    data = np.random.permutation(data)
    #Split train and test data sets
    train_data = data[0:len(data)/2,:]
    test_data = data[len(data)/2:,:]
    
    #Choose feature 2 to 4
    active_feat = [1, 2, 3]
    
    #Split train dataset with corresponding class lable
    train_data_1 = train_data[train_data[:,8]==0, :]
    train_data_2 = train_data[train_data[:,8]==1, :]
    
    #calcuate the prior for each feature
    prior1temp = float(len(train_data[train_data[:,8]==0, :]))
    prior2temp = float(len(train_data[train_data[:,8]==1, :]))
    prior1 = prior1temp/(prior1temp + prior2temp)
    prior2 = prior2temp/(prior1temp + prior2temp)
    
    #training to get means and std for each class
    mean1_NB, std1_NB = get_mean_std(train_data_1, active_feat)
    mean2_NB, std2_NB = get_mean_std(train_data_2, active_feat)
        
    test_features = test_data[:, active_feat]
    correct = 0
    wrong = 0
    for i in range(len(test_data)):
        predict_label = Naive_bayes_prediction(test_features[i,:], mean1_NB, std1_NB, mean2_NB, std2_NB, prior1, prior2)
        if predict_label == test_data[i, 8]:
            correct += 1
        else:
            wrong += 1
    NB_accuracy = float(correct) / len(test_data)
    accuracys.append(NB_accuracy)
    print trial+1, NB_accuracy
print 'Average accuracy:', np.mean(accuracys)
print 'std for accuracy:', np.std(accuracys)