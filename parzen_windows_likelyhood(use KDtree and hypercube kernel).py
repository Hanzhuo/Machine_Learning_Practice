import numpy as np
from sklearn.neighbors import KDTree

data = np.loadtxt('Diabetes.txt', delimiter=',')
accuracys = []

#Define parzen windows estimation function with hypercube kernel
def parzen_windows_likelyhood(train_data, h, observation, tree):
    dimensions = len(observation)
    #since |u| = |(x-x_i)/h| < 1/2 when window function = 1, so |x - x_i| < 1/2 * h
    inside_index = tree.query_radius(observation, r= h/2)
    k = len(inside_index[0])
    return (float(k) / len(train_data)) / float(h**dimensions)

#use random training and testing datasets to run ten times
for trial in range(10):
    #Re-order the dataset
    data = np.random.permutation(data)
    
    #Split train and test data sets
    train_data = data[0:len(data)/2,:]
    test_data = data[len(data)/2:,:]
    
    #select features
    active_feat = [1, 2, 3]
    
    #calcuate the prior for each feature
    prior1temp = float(len(train_data[train_data[:,8]==0, :]))
    prior2temp = float(len(train_data[train_data[:,8]==1, :]))
    prior1 = prior1temp/(prior1temp + prior2temp)
    prior2 = prior2temp/(prior1temp + prior2temp)
    
    #just select train dataset with features with class label
    train_data_features1 = train_data[train_data[:,8]==0, :][:, active_feat]
    train_data_features2 = train_data[train_data[:,8]==1, :][:, active_feat]
    
    #select test dataset with features varibales
    test_data_features = test_data[:, active_feat]
    
    #train KDtree
    #the defsult metric of distance is chebyshev distance which is |x - x_i|
    tree1 = KDTree(train_data_features1, leaf_size=40, metric='chebyshev')
    tree2 = KDTree(train_data_features2, leaf_size=40, metric='chebyshev')
    
    correct = 0
    wrong = 0
    for i in range(len(test_data)):
        post1 = parzen_windows_likelyhood(train_data_features1, 20, test_data_features[i], tree1) * prior1
        post2 = parzen_windows_likelyhood(train_data_features2, 20, test_data_features[i], tree2) * prior2
        if post1 > post2 and test_data[i, 8] == 0:
            correct += 1
        elif post1 < post2  and test_data[i, 8] == 1:
            correct += 1
        else:
            wrong += 1
    accuracy = float(correct) / len(test_data)
    accuracys.append(accuracy)
    print trial+1, accuracy
print 'Average accuracy:', np.mean(accuracys)
print 'std for accuracy:', np.std(accuracys)