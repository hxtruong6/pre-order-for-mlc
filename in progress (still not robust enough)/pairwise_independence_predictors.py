# -pairwise_independence_predictors- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:54:40 2023

@author: nguyenli_admin
"""

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import bernoulli


from pairwise_classifiers import pairwise_classifiers

class predictors:
    
    def __init__(self,
                 n_labels,
                 base_learner: str):
        self.base_learner = base_learner
        
    def save_model(model,filename):
        import pickle
        with open(filename,'wb') as f:
            pickle.dump(model,f)
            
    def _CLR(self, 
            X_test,
            pairwise_2classifier, 
            calibrated_2classifier):
        n_instances, _ = X_test.shape
        calibrated_scores = np.zeros((n_instances))
        for k in range(n_labels):
            clf = calibrated_2classifier[k]
            probabilistic_predictions = clf.predict_proba(X_test)
            _, n_classes =  probabilistic_predictions.shape
            if n_classes == 1:
               predicted_class = clf.predict(X_test[:2]) 
               if predicted_class[0] == 1:
                  calibrated_scores += probabilistic_predictions
            else:
                calibrated_scores += probabilistic_predictions[:,1]
        voting_scores = np.zeros((n_labels,n_instances))
        for k_1 in range(n_labels - 1):
            local_classifier = pairwise_2classifier[k_1]
            for k_2 in range(n_labels - k_1 -1):
                clf = local_classifier[k_2]
                probabilistic_predictions = clf.predict_proba(X_test)
                _, n_classes =  probabilistic_predictions.shape
                if n_classes == 1:
                    predicted_class = clf.predict(X_test[:2]) 
                    if predicted_class[0] == 0:
                        voting_scores[k_1, :] += [1 for n in range(n_instances)]
                    else:
                        voting_scores[k_1 + k_2 +1, :] += [1 for n in range(n_instances)]
                else:
                    voting_scores[k_1, :] += probabilistic_predictions[:,0]
                    voting_scores[k_1 + k_2 +1, :] += probabilistic_predictions[:,1]
        predicted_Y = []
        predicted_ranks = []
        for index in range(n_instances):
            prediction = [1 if voting_scores[k, index] >= calibrated_scores[index] else 0 for k in range(n_labels)]
            rank = [n_labels - sorted(voting_scores[:,index]).index(x) for x in voting_scores[:,index]]
            predicted_Y.append(prediction)
            predicted_ranks.append(rank)        
        return predicted_Y, predicted_ranks    
    
    
    def _partialorders(self, 
                  X_test,
                  Y_pred_chains, n_chains):
        indices_vector = {}
        indVec = 0
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                for l in range(3):
                    key = "%i_%i_%i"%(i,j,l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = predictors._partialorders_compute_parameters(base_learner, indices_vector, n_labels)
        predicted_Y = []
        predicted_preorders = []
        n_instances, _ = X_test.shape
        for index in range(n_instances):
#            print(index, n_instances)
            vector = []
 #           indexEmpty = []
            for i in range(n_labels-1):
                for j in range(j,n_labels):
                 current_predictions = Y_pred_chains[:, index, [i,j]] 
                 pairInfor = [0, 0, 0]
                 for index_clf in range(n_chains):                 
                     pairInfor[0] += current_predictions[index_clf,0]*(1-current_predictions[index_clf,1])
                     pairInfor[1] += (1-current_predictions[index_clf,0])*current_predictions[index_clf,1]
                     pairInfor[2] += 1 - current_predictions[index_clf,0]*(1-current_predictions[index_clf,1]) -(1-current_predictions[index_clf,0])*current_predictions[index_clf,1]   
                pairInfor = [x/n_chains for x in pairInfor]  
                # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic   
                if max(pairInfor) == 1:
                    pairInfor = [x - 10**-10 if x == 1 else (10**-10)/2 for x in pairInfor]
                if min(pairInfor) == 0:
                    zero_indices = [ind for ind in range(3) if pairInfor[ind] == 0]               
                    pairInfor = [(10**-10)/len(zero_indices) if x ==0 else x - (10**-10)/(3-len(zero_indices) ) for x in pairInfor]
                pairInfor = [- np.log(pairInfor[l]) for l in range(3)]
                vector += pairInfor
#                    empty = [indices_vector["%i_%i_%i"%(i,j,l)] for l in range(3) if pairInfor[l] == 0]
#                    indexEmpty += empty
            Gtest = np.array(G)
            Atest = np.array(A)
#            for indCol in indexEmpty:
#                Gtest[:, indCol] = 0
#                Atest[:, indCol] = 0
            hard_prediction, predicted_preorder= predictors._partialorders_reasoning_procedure(base_learner, vector, indices_vector, n_labels, Gtest, h, Atest, b, I, B)
            # , indexEmpty)
            predicted_Y.append(hard_prediction)
            predicted_preorders.append(predicted_preorder)
        return predicted_Y, predicted_preorders   
    
    def _partialorders_compute_parameters(self, 
                           indices_vector,
                           n_labels):    
        G = np.zeros((int(n_labels*(n_labels-1)*(n_labels-2)), int(n_labels*(n_labels-1)*1.5)))
        rowG = 0
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                for k in range(i):
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,0), "%i_%i_%i"%(k,i,1),  "%i_%i_%i"%(k,j,0)]] 
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1,3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,1), "%i_%i_%i"%(k,i,0), "%i_%i_%i"%(k,j,1)]] 
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1,3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(i+1,j):
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,0),  "%i_%i_%i"%(i,k,0), "%i_%i_%i"%(k,j,0)]] 
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1,3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,1), "%i_%i_%i"%(i,k,1), "%i_%i_%i"%(k,j,1)]] 
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1,3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(j+1,n_labels):
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,0), "%i_%i_%i"%(i,k,0), "%i_%i_%i"%(j,k,1)]] 
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1,3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,1), "%i_%i_%i"%(i,k,1), "%i_%i_%i"%(j,k,0)]] 
                    for ind in range(1):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(1,3):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
        h = np.ones((n_labels*(n_labels-1)*(n_labels-2),1))
        A = np.zeros((int(n_labels*(n_labels-1)*0.5), int(n_labels*(n_labels-1)*1.5)))
        rowA = 0
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                # we can inject the information of partial labels at test time here
                for l in range(3):
                    indVec = indices_vector["%i_%i_%i"%(i,j,l)] 
                    A[rowA, indVec] = 1
                rowA += 1
        b = np.ones((int(n_labels*(n_labels-1)*0.5), 1))
        I=set()
        B=set(range(int(n_labels*(n_labels-1)*1.5))) 
        return G, h, A, b, I, B
    
    def _partialorders_reasoning_procedure(self,
                            vector, 
                            indices_vector, 
                            n_labels, 
                            G, 
                            h, 
                            A, 
                            b, 
                            I, 
                            B):
#                            , 
#                            indexEmpty):
        from cvxopt.glpk import ilp
        from numpy import array
        from cvxopt import matrix
        c = np.zeros((n_labels*(n_labels-1)*2,1))
        for ind in range(len(vector)):
            c[ind,0] = vector[ind]   
        (_,x) = ilp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b), I, B)
        optX = array(x)
#        for indX in indexEmpty:
#            optX[indX,0] = 0

# Let both partial and preorder make the hard predictions in similar ways ...         
        scores_d = [0 for x in range(n_labels)] # label i-th dominates at least one label
        scores_n = [0 for x in range(n_labels)] # no label dominates label i-th 
        for i in range(n_labels):
            for k in range(0,i):
                scores_d[i] += optX[indices_vector["%i_%i_%i"%(k,i,1)],0]
                scores_n[i] += optX[indices_vector["%i_%i_%i"%(k,i,0)],0] 
            for j in range(i+1,n_labels):
                scores_d[i] += optX[indices_vector["%i_%i_%i"%(i,j,0)],0] 
                scores_n[i] += optX[indices_vector["%i_%i_%i"%(i,j,1)],0] 
#                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0] 
#                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]             
        hard_prediction = [1 if scores_d[ind] > 0 or scores_n[ind] == 0 else 0 for ind in range(n_labels)]
        
        predicted_partialorder = []
        return hard_prediction, predicted_partialorder    
    
    def _preorders(self, 
                  X_test,
                  Y_pred_chains, n_chains):
        indices_vector = {}
        indVec = 0
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                for l in range(4):
                    key = "%i_%i_%i"%(i,j,l)
                    indices_vector[key] = indVec
                    indVec += 1
        G, h, A, b, I, B = predictors._preorders_compute_parameters(base_learner, indices_vector, n_labels)
        predicted_Y = []
        predicted_preorders = []
        n_instances, _ = X_test.shape
        for index in range(n_instances):
#            print(index, n_instances)
            vector = []
 #           indexEmpty = []
            for i in range(n_labels-1):
                for j in range(j,n_labels):
                 current_predictions = Y_pred_chains[:, index, [i,j]] 
                 pairInfor = [0, 0, 0, 0]
                 for index_clf in range(n_chains):                 
                     pairInfor[0] += current_predictions[index_clf,0]*(1-current_predictions[index_clf,1])
                     pairInfor[1] += (1-current_predictions[index_clf,0])*current_predictions[index_clf,1]
                     pairInfor[2] += (1 - current_predictions[index_clf,0])*(1-current_predictions[index_clf,1]) 
                     pairInfor[3] += current_predictions[index_clf,0]*current_predictions[index_clf,1]
                pairInfor = [x/n_chains for x in pairInfor]
                # add a small regularization term if the probabilistic prediction is deterministic instead of probabilistic   
                if max(pairInfor) == 1:
                    pairInfor = [x - 10**-10 if x == 1 else (10**-10)/3 for x in pairInfor]
                if min(pairInfor) == 0:
                    zero_indices = [ind for ind in range(4) if pairInfor[ind] == 0]
                    pairInfor = [(10**-10)/len(zero_indices) if x ==0 else x- (10**-10)/(4-len(zero_indices)) for x in pairInfor]
                pairInfor = [- np.log(pairInfor[l]) for l in range(4)]
                vector += pairInfor
 #                   empty = [indices_vector["%i_%i_%i"%(i,j,l)] for l in range(4) if pairInfor[l] == 0]
 #                   indexEmpty += empty
            Gtest = np.array(G)
            Atest = np.array(A)
#            for indCol in indexEmpty:
#                Gtest[:, indCol] = 0
#                Atest[:, indCol] = 0
            hard_prediction, predicted_preorder= predictors._preorders_reasoning_procedure(base_learner, vector, indices_vector, n_labels, Gtest, h, Atest, b, I, B)
            #, indexEmpty)
            predicted_Y.append(hard_prediction)
            predicted_preorders.append(predicted_preorder)
        return predicted_Y, predicted_preorders   
    
    def _preorders_compute_parameters(self, 
                           indices_vector,
                           n_labels):    
        G = np.zeros((n_labels*(n_labels-1)*(n_labels-2), n_labels*(n_labels-1)*2))
        rowG = 0
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                for k in range(i):
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,0), "%i_%i_%i"%(i,j,3), "%i_%i_%i"%(k,i,1), "%i_%i_%i"%(k,i,3),  "%i_%i_%i"%(k,j,0), "%i_%i_%i"%(k,j,3)]] 
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2,6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,1), "%i_%i_%i"%(i,j,3), "%i_%i_%i"%(k,i,0), "%i_%i_%i"%(k,i,3), "%i_%i_%i"%(k,j,1), "%i_%i_%i"%(k,j,3)]] 
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2,6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(i+1,j):
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,0), "%i_%i_%i"%(i,j,3),  "%i_%i_%i"%(i,k,0), "%i_%i_%i"%(i,k,3), "%i_%i_%i"%(k,j,0), "%i_%i_%i"%(k,j,3)]] 
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2,6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,1), "%i_%i_%i"%(i,j,3), "%i_%i_%i"%(i,k,1), "%i_%i_%i"%(i,k,3), "%i_%i_%i"%(k,j,1), "%i_%i_%i"%(k,j,3)]] 
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2,6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                for k in range(j+1,n_labels):
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,0), "%i_%i_%i"%(i,j,3), "%i_%i_%i"%(i,k,0), "%i_%i_%i"%(i,k,3),  "%i_%i_%i"%(j,k,1), "%i_%i_%i"%(j,k,3)]] 
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2,6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
                    indVecs = [indices_vector[val] for val in ["%i_%i_%i"%(i,j,1), "%i_%i_%i"%(i,j,3), "%i_%i_%i"%(i,k,1), "%i_%i_%i"%(i,k,3), "%i_%i_%i"%(j,k,0), "%i_%i_%i"%(j,k,3)]] 
                    for ind in range(2):
                        G[rowG, indVecs[ind]] = -1
                    for ind in range(2,6):
                        G[rowG, indVecs[ind]] = 1
                    rowG += 1
        h = np.ones((n_labels*(n_labels-1)*(n_labels-2),1))
        A = np.zeros((int(n_labels*(n_labels-1)*0.5), int(n_labels*(n_labels-1)*2)))
        rowA = 0
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                # we can inject the information of partial labels at test time here
                for l in range(4):
                    indVec = indices_vector["%i_%i_%i"%(i,j,l)] 
                    A[rowA, indVec] = 1
                rowA += 1
        b = np.ones((int(n_labels*(n_labels-1)*0.5), 1))
        I=set()
        B=set(range(n_labels*(n_labels-1)*2)) 
        return G, h, A, b, I, B
    
    def _preorders_reasoning_procedure(self,
                            vector, 
                            indices_vector, 
                            n_labels, 
                            G, 
                            h, 
                            A, 
                            b, 
                            I, 
                            B):
#                            , 
#                            indexEmpty):
        from cvxopt.glpk import ilp
        from numpy import array
        from cvxopt import matrix
        c = np.zeros((n_labels*(n_labels-1)*2,1))
        for ind in range(len(vector)):
            c[ind,0] = vector[ind]   
        (_,x) = ilp(matrix(c), matrix(G), matrix(h), matrix(A), matrix(b), I, B)
        optX = array(x)
#        for indX in indexEmpty:
#            optX[indX,0] = 0
            
#        epist_00 = 0
#        aleat_11 = 0
        scores_d = [0 for x in range(n_labels)] # label i-th dominates at least one label
        scores_n = [0 for x in range(n_labels)] # no label dominates label i-th 
        for i in range(n_labels):
            for k in range(0,i):
                scores_d[i] += optX[indices_vector["%i_%i_%i"%(k,i,1)],0]
                scores_n[i] += optX[indices_vector["%i_%i_%i"%(k,i,0)],0] 
            for j in range(i+1,n_labels):
                scores_d[i] += optX[indices_vector["%i_%i_%i"%(i,j,0)],0] 
                scores_n[i] += optX[indices_vector["%i_%i_%i"%(i,j,1)],0] 
#                epist_00 += optX[indicesVector["%i_%i_%i"%(i,j,2)],0] 
#                aleat_11 += optX[indicesVector["%i_%i_%i"%(i,j,3)],0]             
        hard_prediction = [1 if scores_d[ind] > 0 or scores_n[ind] == 0 else 0 for ind in range(n_labels)]
        predicted_preorder = []
        return hard_prediction, predicted_preorder
        
    def _hamming(self,
                 predicted_Y, 
                 true_Y):
        from sklearn.metrics import hamming_loss
        return 1 - hamming_loss(predicted_Y, true_Y)
    
    def _f1(self,
            predicted_Y, 
            true_Y):    
#        from sklearn.metrics import f1_score
#        return np.mean(f1_score(hard_predictions, true_label, average=None))
        f1 = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if max(predicted_Y[index]) == 0 and max(true_Y[index]) == 0:
                f1 += 1
            else:
                f1 += (2*np.dot(predicted_Y[index], true_Y[index]))/(np.sum(predicted_Y[index]) + np.sum(true_Y[index]))
        return f1/n_instances 
    
    def _jaccard(self,
                 predicted_Y, 
                 true_Y):
#        from sklearn.metrics import jaccard_score
#        return np.mean(jaccard_score(hard_predictions, true_label, average=None))
        jaccard = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if max(predicted_Y[index]) == 0 and max(true_Y[index]) == 0:
                jaccard += 1
            else:
                jaccard += (np.dot(predicted_Y[index], true_Y[index]))/(np.sum(predicted_Y[index]) + np.sum(true_Y[index]) - np.dot(predicted_Y[index], true_Y[index]))
        return jaccard/n_instances     

    def _subset0_1(self,
                  predicted_Y, 
                  true_Y):
#        from sklearn.metrics import jaccard_score
#        return np.mean(jaccard_score(hard_predictions, true_label, average=None))
        subset0_1 = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if list(predicted_Y[index]) == list(true_Y[index]):
                subset0_1 += 1
        return subset0_1/n_instances 
    
    def _subset_exact_match(self,
                  predicted_Y, 
                  true_Y):
        n_instances = len(predicted_Y)
        n_labels = len(predicted_Y[0])
        subset_exact_match = [0 for x in range(n_labels)]
        for index in range(n_instances):
            matched_positions = np.sum([1 if predicted_Y[index][x] == true_Y[index][x] else 0 for x in range(n_labels) ])
            for pos in range(matched_positions):
                subset_exact_match[pos] += 1
        return [x/n_instances for x in subset_exact_match]
    
    def _recall(self,
                  predicted_Y, 
                  true_Y):
#        from sklearn.metrics import jaccard_score
#        return np.mean(jaccard_score(hard_predictions, true_label, average=None))
        recall = 0
        n_instances = len(predicted_Y)
        for index in range(n_instances):
            if np.dot(predicted_Y[index], true_Y[index]) == np.sum(true_Y[index]):
                recall += 1
        return recall/n_instances     
    
#for a quick test 
if __name__ == '__main__':
    dataPath = './data/'
#    dataFile = 'emotions.arff'
#    dataFile = 'scene.arff'
#    dataFile = 'CHD_49.arff'
    # https://cometa.ujaen.es/datasets/VirusGO
    # https://www.uco.es/kdis/mllresources/#ImageDesc
#    for dataFile in ['emotions.arff','CHD_49.arff', 'scene.arff']: 
#    n_labels_set = [14]
    ind = 0
#    for dataFile in ['Water-quality.arff']: 
    n_labels_set = [6, 6, 6, 14, 14]
    for dataFile in ['emotions.arff','CHD_49.arff', 'scene.arff', 'Yeast.arff', 'Water-quality.arff']: 
#    n_labels_set = [19]
#    for dataFile in ['birds.arff']: 
        n_labels = n_labels_set[ind]
        ind += 1
#        n_labels = 6
        total_repeat = 1
        folds = 10
        n_chains = 50
        for noisy_rate in [0.0, 0.2, 0.4]:
#        for noisy_rate in [0.2, 0.4]:
            for base_learner in ["RF", "ET"]:
    #        for base_learner in ["RF", "ETC", "XGBoost", "LightGBM"]:
    
                print(dataFile, base_learner)
            #    base_learner = "LightGBM"
                data = arff.loadarff(dataPath+dataFile)  
                df = pd.DataFrame(data[0]).to_numpy()
                n_cols = len(df[0])
                if dataFile in ['birds.arff', 'emotions.arff', 'scene.arff']:
                    X = df[:, : n_cols - n_labels]
                    Y = df[:, n_cols - n_labels :].astype(int) 
                else:
                    X = df[:, n_labels :] 
                    Y = df[:, : n_labels].astype(int) 
                Y = np.where(Y < 0, 0, Y)
                # from skmultilearn.dataset import load_from_arff                
                # features, labels = load_from_arff(dataPath+dataFile, 
                #     # number of labels
                #     label_count=6, 
                #     # MULAN format, labels at the end of rows in arff data, using 'end' for label_location
                #     # 'start' is also available for MEKA format
                #     label_location='end', 
                #     # bag of words
                #     input_feature_type='int', encode_nominal=False, 
                #     # sometimes the sparse ARFF loader is borked, like in delicious,
                #     # scikit-multilearn converts the loaded data to sparse representations, 
                #     # so disabling the liac-arff sparse loader
                #     # but you may set load_sparse to True if this fails
                #     load_sparse=False, 
                #     # this decides whether to return attribute names or not, usually 
                #     # you don't need this
                #     return_attribute_definitions=False)
            
                for repeat in range(total_repeat):
                    average_hamming_loss_pairwise_2classifier = []
                    average_hamming_loss_pairwise_3classifier = []
                    average_hamming_loss_pairwise_4classifier = []
                    average_hamming_loss_ECC = []
    
                    average_f1_pairwise_2classifier = []
                    average_f1_pairwise_3classifier = []
                    average_f1_pairwise_4classifier = []
                    average_f1_ECC = []
                    
                    average_jaccard_pairwise_2classifier = []
                    average_jaccard_pairwise_3classifier = []
                    average_jaccard_pairwise_4classifier = []
                    average_jaccard_ECC = []
                    
                    average_subset0_1_pairwise_2classifier = []
                    average_subset0_1_pairwise_3classifier = []
                    average_subset0_1_pairwise_4classifier = []  
                    average_subset0_1_ECC = [] 
                    
                    average_recall_pairwise_2classifier = []
                    average_recall_pairwise_3classifier = []
                    average_recall_pairwise_4classifier = [] 
                    average_recall_ECC = [] 
    #                average_subset_exact_match_pairwise_2classifier = []
    #                average_subset_exact_match_pairwise_3classifier = []
    #                average_subset_exact_match_pairwise_4classifier = []
                    
                    fold = 0

                    Kf = KFold(n_splits=folds, random_state=42, shuffle=True)
                    
                    for train_index, test_index in Kf.split(Y):
                        for index in range(len(train_index)):
                            known_index = []
                            for k in range(n_labels):
                                if bernoulli.rvs(size = 1, p = noisy_rate)[0] == 1:
                                   if Y[train_index[index], k] == 1:
                                      Y[train_index[index], k] = 0
                                   else:
                                      Y[train_index[index], k] = 1
                        print(["repeat", "fold", repeat, fold])
                        print("====================== pairwise_2classifier ======================")
                        pairwise_2classifier, calibrated_2classifier = pairwise_classifiers._pairwise_2classifier(base_learner, n_labels, X[train_index], Y[train_index])
                        predicted_Y, predicted_ranks = predictors._CLR(n_labels, X[test_index], pairwise_2classifier, calibrated_2classifier)
                        hamming_loss_pairwise_2classifier = predictors._hamming(base_learner, predicted_Y, Y[test_index])
                        average_hamming_loss_pairwise_2classifier.append(hamming_loss_pairwise_2classifier)
                        print(hamming_loss_pairwise_2classifier)
                        f1_pairwise_2classifier = predictors._f1(base_learner, predicted_Y, Y[test_index])
                        average_f1_pairwise_2classifier.append(f1_pairwise_2classifier)
                        print(f1_pairwise_2classifier)
                        jaccard_pairwise_2classifier = predictors._jaccard(base_learner, predicted_Y, Y[test_index])
                        average_jaccard_pairwise_2classifier.append(jaccard_pairwise_2classifier)
                        print(jaccard_pairwise_2classifier)   
                        subset0_1_pairwise_2classifier = predictors._subset0_1(base_learner, predicted_Y, Y[test_index])
                        average_subset0_1_pairwise_2classifier.append(subset0_1_pairwise_2classifier)
                        print(subset0_1_pairwise_2classifier)   
                        recall_pairwise_2classifier = predictors._recall(base_learner, predicted_Y, Y[test_index])
                        average_recall_pairwise_2classifier.append(recall_pairwise_2classifier)
                        print(recall_pairwise_2classifier) 
                        # subset_exact_match_pairwise_2classifier = predictors._subset_exact_match(base_learner, predicted_Y, Y[test_index])
                        # average_subset_exact_match_pairwise_2classifier.append(subset_exact_match_pairwise_2classifier)
                        # print(subset_exact_match_pairwise_2classifier)
    #                    print(np.mean([np.mean(x) for x in hard_predictions]))
    
                        print("====================== Train ECC ======================")
                        from sklearn.multioutput import ClassifierChain
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.ensemble import ExtraTreesClassifier
                        from sklearn.ensemble import GradientBoostingClassifier
                        import lightgbm as lgb
                        
                        if base_learner == "RF":
                            clf = RandomForestClassifier(random_state=42)
                        if base_learner == "ET":
                            clf = ExtraTreesClassifier(random_state=42)
                        if base_learner == "XGBoost":
                            clf = GradientBoostingClassifier(random_state=42)
                        if base_learner == "LightGBM":
                            clf = lgb.LGBMClassifier(random_state=42) 
                        chains = [ClassifierChain(clf, order="random", random_state = i*10) for i in range(n_chains)] 
                        n_clf = 0
                        for chain in chains:
                            print(n_clf, n_chains)
                            n_clf += 1
                            chain.fit(X[train_index], Y[train_index])
                        Y_pred_chains = np.array([chain.predict_proba(X[test_index]) for chain in chains])
    
                        print("====================== pairwise_3classifier ======================")
                        predicted_Y, predicted_ranks = predictors._partialorders(n_labels, X[test_index], Y_pred_chains, n_chains)
            #            print(hard_predictions)
                        hamming_loss_pairwise_3classifier = predictors._hamming(base_learner, predicted_Y, Y[test_index])
                        average_hamming_loss_pairwise_3classifier.append(hamming_loss_pairwise_3classifier)
                        print(hamming_loss_pairwise_3classifier)
                        f1_pairwise_3classifier = predictors._f1(base_learner, predicted_Y, Y[test_index])
                        average_f1_pairwise_3classifier.append(f1_pairwise_3classifier)
                        print(f1_pairwise_3classifier)
                        jaccard_pairwise_3classifier = predictors._jaccard(base_learner, predicted_Y, Y[test_index])
                        average_jaccard_pairwise_3classifier.append(jaccard_pairwise_3classifier)
                        print(jaccard_pairwise_3classifier)
                        subset0_1_pairwise_3classifier = predictors._subset0_1(base_learner, predicted_Y, Y[test_index])
                        average_subset0_1_pairwise_3classifier.append(subset0_1_pairwise_3classifier)
                        print(subset0_1_pairwise_3classifier)
                        recall_pairwise_3classifier = predictors._recall(base_learner, predicted_Y, Y[test_index])
                        average_recall_pairwise_3classifier.append(recall_pairwise_3classifier)
                        print(recall_pairwise_3classifier) 
                        # subset_exact_match_pairwise_3classifier = predictors._subset_exact_match(base_learner, predicted_Y, Y[test_index])
                        # average_subset_exact_match_pairwise_3classifier.append(subset_exact_match_pairwise_3classifier)
                        # print(subset_exact_match_pairwise_3classifier)
                        
    #                    print(np.mean([np.mean(x) for x in hard_predictions]))
                        
                        print("====================== pairwise_4classifier ======================")
                                                                                
                        predicted_Y, predicted_ranks = predictors._preorders(n_labels, X[test_index], Y_pred_chains, n_chains)
            #            print(hard_predictions)
                        hamming_loss_pairwise_4classifier = predictors._hamming(base_learner, predicted_Y, Y[test_index])
                        average_hamming_loss_pairwise_4classifier.append(hamming_loss_pairwise_4classifier)
                        print(hamming_loss_pairwise_4classifier)
                        f1_pairwise_4classifier = predictors._f1(base_learner, predicted_Y, Y[test_index])
                        average_f1_pairwise_4classifier.append(f1_pairwise_4classifier)
                        print(f1_pairwise_4classifier)
                        jaccard_pairwise_4classifier = predictors._jaccard(base_learner, predicted_Y, Y[test_index])
                        average_jaccard_pairwise_4classifier.append(jaccard_pairwise_4classifier)
                        print(jaccard_pairwise_4classifier)
                        subset0_1_pairwise_4classifier = predictors._subset0_1(base_learner, predicted_Y, Y[test_index])
                        average_subset0_1_pairwise_4classifier.append(subset0_1_pairwise_4classifier)
                        print(subset0_1_pairwise_4classifier)
                        recall_pairwise_4classifier = predictors._recall(base_learner, predicted_Y, Y[test_index])
                        average_recall_pairwise_4classifier.append(recall_pairwise_4classifier)
                        print(recall_pairwise_4classifier) 
                        # subset_exact_match_pairwise_4classifier = predictors._subset_exact_match(base_learner, predicted_Y, Y[test_index])
                        # average_subset_exact_match_pairwise_4classifier.append(subset_exact_match_pairwise_4classifier)
                        # print(subset_exact_match_pairwise_4classifier)
    #                    print(np.mean([np.mean(x) for x in hard_predictions]))
    
    #                    ECC.fit(X[train_index].astype(float), Y[train_index].astype(float))
    #                    predicted_Y = ECC.predict(X[test_index].astype(float))
                        print("====================== ECC ======================")
                        Y_pred_members = np.array([chain.predict(X[test_index]) for chain in chains])

                        Y_pred_ensemble = Y_pred_members.mean(axis=0)
                        predicted_Y = np.where(Y_pred_ensemble > 0.5, 1, 0)
                        
                        hamming_loss_ECC = predictors._hamming(base_learner, predicted_Y, Y[test_index])
                        average_hamming_loss_ECC.append(hamming_loss_ECC)
                        print(hamming_loss_ECC)
                        f1_ECC = predictors._f1(base_learner, predicted_Y, Y[test_index])
                        average_f1_ECC.append(f1_ECC)
                        print(f1_ECC)
                        jaccard_ECC = predictors._jaccard(base_learner, predicted_Y, Y[test_index])
                        average_jaccard_ECC.append(jaccard_ECC)
                        print(jaccard_ECC)
                        subset0_1_ECC = predictors._subset0_1(base_learner, predicted_Y, Y[test_index])
                        average_subset0_1_ECC.append(subset0_1_ECC)
                        print(subset0_1_ECC)
                        recall_ECC = predictors._recall(base_learner, predicted_Y, Y[test_index])
                        average_recall_ECC.append(recall_ECC)
                        print(recall_ECC) 
                       
                        fold += 1      
                    print("====================== Average results: ======================")
                    print("====================== Haming ======================")               
                    print(np.mean(average_hamming_loss_pairwise_2classifier))
                    print(np.mean(average_hamming_loss_pairwise_3classifier))
                    print(np.mean(average_hamming_loss_pairwise_4classifier))
                    print(np.mean(average_hamming_loss_ECC))
                    print("====================== F1 ======================")               
                    print(np.mean(average_f1_pairwise_2classifier))
                    print(np.mean(average_f1_pairwise_3classifier))
                    print(np.mean(average_f1_pairwise_4classifier))
                    print(np.mean(average_f1_ECC))
                    print("====================== Jaccard ======================")                               
                    print(np.mean(average_jaccard_pairwise_2classifier))
                    print(np.mean(average_jaccard_pairwise_3classifier))
                    print(np.mean(average_jaccard_pairwise_4classifier))
                    print(np.mean(average_jaccard_ECC))
                    print("====================== Subset 0/1 ======================")                              
                    print(np.mean(average_subset0_1_pairwise_2classifier))
                    print(np.mean(average_subset0_1_pairwise_3classifier))
                    print(np.mean(average_subset0_1_pairwise_4classifier))
                    print(np.mean(average_subset0_1_ECC))
                    print("====================== Recall ======================")                
                    print(np.mean(average_recall_pairwise_2classifier))
                    print(np.mean(average_recall_pairwise_3classifier))
                    print(np.mean(average_recall_pairwise_4classifier))
                    print(np.mean(average_recall_ECC))
                    
                    # print(list(np.mean(average_subset_exact_match_pairwise_2classifier, axis=0)))
                    # print(list(np.mean(average_subset_exact_match_pairwise_3classifier, axis=0)))
                    # print(list(np.mean(average_subset_exact_match_pairwise_4classifier, axis=0)))
    
                    
                final_results = [[np.mean(average_hamming_loss_pairwise_2classifier), np.std(average_hamming_loss_pairwise_2classifier), np.mean(average_hamming_loss_pairwise_3classifier), np.std(average_hamming_loss_pairwise_3classifier), np.mean(average_hamming_loss_pairwise_4classifier), np.std(average_hamming_loss_pairwise_4classifier), np.mean(average_hamming_loss_ECC), np.std(average_hamming_loss_ECC)],
                                 [np.mean(average_f1_pairwise_2classifier), np.std(average_f1_pairwise_2classifier), np.mean(average_f1_pairwise_3classifier), np.std(average_f1_pairwise_3classifier), np.mean(average_f1_pairwise_4classifier), np.std(average_f1_pairwise_4classifier), np.mean(average_f1_ECC), np.std(average_f1_ECC)],
                                 [np.mean(average_jaccard_pairwise_2classifier), np.std(average_jaccard_pairwise_2classifier), np.mean(average_jaccard_pairwise_3classifier), np.std(average_jaccard_pairwise_3classifier), np.mean(average_jaccard_pairwise_4classifier), np.std(average_jaccard_pairwise_4classifier), np.mean(average_jaccard_ECC), np.std(average_jaccard_ECC)],
                                 [np.mean(average_subset0_1_pairwise_2classifier), np.std(average_subset0_1_pairwise_2classifier), np.mean(average_subset0_1_pairwise_3classifier), np.std(average_subset0_1_pairwise_3classifier), np.mean(average_subset0_1_pairwise_4classifier), np.std(average_subset0_1_pairwise_4classifier), np.mean(average_subset0_1_ECC), np.std(average_subset0_1_ECC)],
                                 [np.mean(average_recall_pairwise_2classifier), np.std(average_recall_pairwise_2classifier), np.mean(average_recall_pairwise_3classifier), np.std(average_recall_pairwise_3classifier), np.mean(average_recall_pairwise_4classifier), np.std(average_recall_pairwise_4classifier), np.mean(average_recall_ECC), np.std(average_recall_ECC)]]
    
                res_file = "pairwise_independence_noisy_4_w_try_compareAcc_%i_%i_%i_%s_%s" %(int(noisy_rate*10), total_repeat, folds, dataFile, base_learner)
                file = open(res_file, "w")
                file.writelines("%s\n" %line for line in final_results)
                file.close()  
                
                # final_results = [list(np.mean(average_subset_exact_match_pairwise_2classifier, axis=0)),
                #                  list(np.mean(average_subset_exact_match_pairwise_3classifier, axis=0)),
                #                  list(np.mean(average_subset_exact_match_pairwise_4classifier, axis=0))]
                # res_file = "noisy_w_s_try_compareAcc_%i_%i_%s_%s" %(total_repeat, folds, dataFile, base_learner)
                # file = open(res_file, "w")
                # file.writelines("%s\n" %line for line in final_results)
                # file.close()                                    