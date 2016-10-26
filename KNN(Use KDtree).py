import numpy as np
from sklearn.neighbors import KDTree

data = np.loadtxt('Diabetes.txt', delimiter=',')

def KNN_KDtree_predict(k, data):
    predict_data = []
    for i in range(len(data)):
        dist, index = tree.query(data[i], k=k)
        num_pred_0 = 0
        num_pred_1 = 0
        for x in index[0]:
            if train_data[x, 8] == 0:
                num_pred_0 += 1
            if train_data[x, 8] == 1:
                num_pred_1 += 1

        if num_pred_0 > num_pred_1:
            prediction = 0
        else:
            prediction = 1
        predict_data.append(prediction)
    return predict_data



accuracys_k_1 = []
accuracys_k_5 = []
accuracys_k_11 = []


for trial in range(10):
    #Re-order the dataset
    data = np.random.permutation(data)
    
    #Split train and test data sets
    train_data = data[0:len(data)/2,:]
    test_data = data[len(data)/2:,:]
    
    #select features
    active_feat = [1, 2, 3]
    
    #just select train dataset with features
    train_data_features = train_data[:, active_feat]
    test_data_features = test_data[:, active_feat]
    
    #train KDtree
    #the defsult metric of distance is Euclidean distance
    tree = KDTree(train_data_features, leaf_size=40)
    
    #K=1
    correct = 0
    wrong = 0
    prediction_K_1 = KNN_KDtree_predict(1, test_data_features)
    for i in range(len(prediction_K_1)):
        if prediction_K_1[i] == test_data[i, 8]:
            correct += 1
        else:
            wrong += 1
    accuracy_k_1 = float(correct)/len(prediction_K_1)
    accuracys_k_1.append(accuracy_k_1)
    print trial+1, accuracy_k_1

    #K=5
    correct = 0
    wrong = 0
    prediction_K_5 = KNN_KDtree_predict(5, test_data_features)
    for i in range(len(prediction_K_5)):
        if prediction_K_5[i] == test_data[i, 8]:
            correct += 1
        else:
            wrong += 1
    accuracy_k_5 = float(correct)/len(prediction_K_5)
    accuracys_k_5.append(accuracy_k_5)
    print trial+1, accuracy_k_5

    #K=11
    correct = 0
    wrong = 0
    prediction_K_11 = KNN_KDtree_predict(11, test_data_features)
    for i in range(len(prediction_K_11)):
        if prediction_K_11[i] == test_data[i, 8]:
            correct += 1
        else:
            wrong += 1
    accuracy_k_11 = float(correct)/len(prediction_K_11)
    accuracys_k_11.append(accuracy_k_11)
    print trial+1, accuracy_k_11
    
print 'Average accuracy k=1:', np.mean(accuracy_k_1)
print 'std for accuracy k=1:', np.std(accuracy_k_1)
print 'Average accuracy k=5:', np.mean(accuracy_k_5)
print 'std for accuracy k=5:', np.std(accuracy_k_5)
print 'Average accuracy k=11:', np.mean(accuracy_k_11)
print 'std for accuracy k=11:', np.std(accuracy_k_11)