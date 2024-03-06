# -pairwise_classifiers- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:02:21 2023

@author: nguyenli_admin
"""

from base_classifiers import base_classifiers


class pairwise_classifiers:
    def __init__(self,
                 base_learner: str):
        self.base_learner = base_learner
   
    def _pairwise_2classifier(self, 
                             n_labels, 
                             X, 
                             Y):
        n_instances, _ = Y.shape
        calibrated_classifiers = []
        for k in range(n_labels):
            MCC_y = []
            for n in range(n_instances):
                if Y[n,k] == 1:
                    MCC_y.append(0)
                else: 
                    MCC_y.append(1)
            calibrated_classifiers.append(base_classifiers._2classifier(self, X, MCC_y))
        classifiers = []
        for k_1 in range(n_labels-1):
            local_classifier = []
            for k_2 in range(k_1+1, n_labels):
                MCC_X = []
                MCC_y = []
                for n in range(n_instances):
                    if Y[n,k_1] == 1:
                       MCC_X.append(X[n]) 
                       MCC_y.append(0)
                    elif Y[n,k_2] == 1: 
                        MCC_X.append(X[n]) 
                        MCC_y.append(1)
                # if len(transformed_classes) == 0:
                #     print(labels)
                #     print(k_1,k_2)
                #     print(labels[:, k_1])
                #     print(labels[:, k_2])
                local_classifier.append(base_classifiers._2classifier(self, MCC_X, MCC_y))
            classifiers.append(local_classifier) 
        return classifiers, calibrated_classifiers
    
    def _pairwise_3classifier(self, 
                              n_labels,
                              X, 
                              Y):
        n_instances, _ = Y.shape
        classifiers = []
        for k_1 in range(n_labels-1):
            local_classifier = []
            for k_2 in range(k_1+1, n_labels):
                MCC_y = []
                for n in range(n_instances):
                    if Y[n,k_1] == Y[n,k_2]:
                        MCC_y.append(2)
                    elif Y[n,k_1] == 1:
                       MCC_y.append(0)
                    elif Y[n,k_2] == 1: 
                        MCC_y.append(1)
                local_classifier.append(base_classifiers._3classifier(self, X, MCC_y))
            classifiers.append(local_classifier) 
        return classifiers
    
    def _pairwise_4classifier(self, 
                              n_labels,
                              X, 
                              Y):
        n_instances, _ = Y.shape
        classifiers = []
        for k_1 in range(n_labels-1):
            local_classifier = []
            for k_2 in range(k_1+1, n_labels):
                MCC_y = []
                for n in range(n_instances):
                    if Y[n,k_1] == 0 and Y[n,k_2] ==0 :
                        MCC_y.append(2)
                    elif Y[n,k_1] == 1 and Y[n,k_2] ==1 :
                        MCC_y.append(3)
                    elif Y[n,k_1] == 1:
                       MCC_y.append(0)
                    elif Y[n,k_2] == 1: 
                        MCC_y.append(1)
                local_classifier.append(base_classifiers._4classifier(self, X, MCC_y))
            classifiers.append(local_classifier) 
        return classifiers
    
    def _BR(self, 
            n_labels,
            X, 
            Y):
        classifiers = []
        for k in range(n_labels):
            classifiers.append(base_classifiers._2classifier(self, X, Y[:,k]))
        return classifiers
    
