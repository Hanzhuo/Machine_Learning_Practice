import numpy as np
from numpy.linalg import inv

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
    
    #training
    #calcuate the mean for each feature
    mean1 = np.mean(train_data[train_data[:,8]==0, :][:, active_feat], axis=0)
    mean2 = np.mean(train_data[train_data[:,8]==1, :][:, active_feat], axis=0)
    
    #calcuate the covariance for each feature
    cov1 = np.cov(train_data[train_data[:,8]==0, :][:, active_feat].T)
    cov2 = np.cov(train_data[train_data[:,8]==1, :][:, active_feat].T)
    
    #calcuate the prior for each feature
    prior1temp = float(len(train_data[train_data[:,8]==0, :]))
    prior2temp = float(len(train_data[train_data[:,8]==1, :]))
    prior1 = prior1temp/(prior1temp + prior2temp)
    prior2 = prior2temp/(prior1temp + prior2temp)
    
    #testing
    correct = 0
    wrong = 0
    for i in range(len(test_data)):
        likelyhood1 = np.exp(np.linalg.det(- np.matrix(test_data[i, active_feat] - mean1) * inv(np.matrix(cov1)) 
                                           * np.matrix(test_data[i, active_feat] - mean1).T) / 2) / np.sqrt(np.linalg.det(cov1))
        likelyhood2 = np.exp(np.linalg.det(- np.matrix(test_data[i, active_feat] - mean2) * inv(np.matrix(cov2)) 
                                           * np.matrix(test_data[i, active_feat] - mean2).T) / 2) / np.sqrt(np.linalg.det(cov2))
        post1 = likelyhood1 * prior1
        post2 = likelyhood2 * prior2
    
        if post1 > post2 and test_data[i, 8] == 0:
            correct += 1
        elif post1 < post2 and test_data[i, 8] == 1:
            correct += 1
        else:
            wrong += 1
    accuracy = float(correct) / len(test_data)
    accuracys.append(accuracy)
    print trial+1, accuracy
print 'Average accuracy:', np.mean(accuracys)
print 'std for accuracy:', np.std(accuracys)