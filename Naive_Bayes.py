# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:02:25 2020

@author: Kenkun Liu
"""

import torch
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np

def load_data():
    print("-->loading data...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(mnist_trainset, shuffle=True)
    testloader = torch.utils.data.DataLoader(mnist_testset, shuffle=True)
    
    # load data and transform it into numpy values
    train_data = []
    train_label = []
    for image, label in trainloader:
        train_data.append(image.squeeze().numpy())
        train_label.append(label.item())
    
    test_data = []
    test_label = []
    for image, label in testloader:
        test_data.append(image.squeeze().numpy())
        test_label.append(label.item())

    print("-->finish loading")
    return train_data, train_label, test_data, test_label


def count(ls, elem):
    n = 0
    for e in ls:
        if elem == e:
            n = n + 1
    return n


def training(train_data, train_label):
    print("-->start training")
    classes = list(set(train_label))
    # count the number of samples in the training set of each class
    # then divide it by the total number of samples in the training set
    priors = np.array([count(train_label, label) for label in classes])/len(train_label)
    num_features = train_data[0].size
    counter = np.zeros([len(classes), num_features])
    
    for i in range(len(train_data)):
        im_vector = (train_data[i].flatten() > 0.5).astype(np.int_)
        counter[train_label[i]] = counter[train_label[i]] + im_vector
    cond_prob = np.zeros_like(counter)
    for i in range(len(classes)):
        cond_prob[i] = (counter[i]+1)/(priors[i]*len(train_label)+2)  # laplacian smoothing
    print("-->finish training")
    model = {'priors':priors, 'cond_prob':cond_prob, 'classes':classes}
    return model
    
    
def testing(model, test_data, test_label):
    print("-->testing")
    priors = model['priors']
    cond_prob = model['cond_prob']
    classes = model['classes']
    num_of_True = 0
    for i in range(len(test_data)):
        label = -1
        im_vector = (test_data[i].flatten() > 0.5).astype(np.int_)
        max_log_prob = -10000000
        for k in range(len(classes)):
            log_prob = np.log2(priors[k]) + np.sum(im_vector*np.log2(cond_prob[k])+(1-im_vector)*np.log2(1-cond_prob[k]))
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                label = k
        if label == test_label[i]:
            num_of_True = num_of_True + 1
    accuracy = num_of_True/len(test_label)
    return accuracy


def main():
    train_data, train_label, test_data, test_label = load_data()
    model = training(train_data, train_label)
    print("Accuracy = {}".format(testing(model, test_data, test_label)))


#if __name__ == '__main__':
#    main()
    