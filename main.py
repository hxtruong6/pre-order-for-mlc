# -predictors- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:54:40 2023

@author: nguyenli_admin
"""

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import bernoulli
from evaluation_metric import EvaluationMetric


from pairwise_classifiers import pairwise_classifiers
from predictor import Predictor

# for a quick test
if __name__ == "__main__":
    dataPath = "./data/"
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

    eval_metric = EvaluationMetric()

    for dataFile in [
        "emotions.arff",
        # "CHD_49.arff",
        # "scene.arff",
        # "Yeast.arff",
        # "Water-quality.arff",
    ]:
        #    n_labels_set = [19]
        #   for dataFile in ['birds.arff']:
        n_labels = n_labels_set[ind]

        ind += 1
        #        n_labels = 6
        total_repeat = 1
        folds = 2
        for noisy_rate in [0.0]:
            #        for noisy_rate in [0.0, 0.2, 0.4]:
            for base_learner in ["XGBoost"]:
                #        for base_learner in ["RF", "ETC", "XGBoost", "LightGBM"]:

                print(dataFile, base_learner)
                predictors = Predictor(n_labels, base_learner)

                #    base_learner = "LightGBM"
                data = arff.loadarff(dataPath + dataFile)
                df = pd.DataFrame(data[0]).to_numpy()
                n_cols = len(df[0])
                if dataFile in ["birds.arff", "emotions.arff", "scene.arff"]:
                    X = df[:, : n_cols - n_labels]
                    Y = df[:, n_cols - n_labels :].astype(int)
                else:
                    X = df[:, n_labels:]
                    Y = df[:, :n_labels].astype(int)
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
                                if bernoulli.rvs(size=1, p=noisy_rate)[0] == 1:
                                    if Y[train_index[index], k] == 1:
                                        Y[train_index[index], k] = 0
                                    else:
                                        Y[train_index[index], k] = 1
                        print(["repeat", "fold", repeat, fold])
                        print(
                            "====================== pairwise_2classifier ======================"
                        )
                        pairwise_2classifier, calibrated_2classifier = (
                            pairwise_classifiers._pairwise_2classifier(
                                base_learner, n_labels, X[train_index], Y[train_index]
                            )
                        )
                        predicted_Y, predicted_ranks = predictors._CLR(
                            X[test_index],
                            pairwise_2classifier,
                            calibrated_2classifier,
                        )
                        # hamming_loss_pairwise_2classifier = predictors._hamming(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        hamming_loss_pairwise_2classifier = eval_metric.hamming_loss(
                            predicted_Y, Y[test_index]
                        )

                        average_hamming_loss_pairwise_2classifier.append(
                            hamming_loss_pairwise_2classifier
                        )
                        print(hamming_loss_pairwise_2classifier)
                        # f1_pairwise_2classifier = predictors._f1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        f1_pairwise_2classifier = eval_metric.f1(
                            predicted_Y, Y[test_index]
                        )
                        average_f1_pairwise_2classifier.append(f1_pairwise_2classifier)
                        print(f1_pairwise_2classifier)

                        # jaccard_pairwise_2classifier = predictors._jaccard(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        jaccard_pairwise_2classifier = eval_metric.jaccard(
                            predicted_Y, Y[test_index]
                        )
                        average_jaccard_pairwise_2classifier.append(
                            jaccard_pairwise_2classifier
                        )
                        print(jaccard_pairwise_2classifier)
                        # subset0_1_pairwise_2classifier = predictors._subset0_1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        subset0_1_pairwise_2classifier = eval_metric.subset0_1(
                            predicted_Y, Y[test_index]
                        )
                        average_subset0_1_pairwise_2classifier.append(
                            subset0_1_pairwise_2classifier
                        )
                        print(subset0_1_pairwise_2classifier)
                        # recall_pairwise_2classifier = predictors._recall(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        recall_pairwise_2classifier = eval_metric.recall(
                            predicted_Y, Y[test_index]
                        )
                        average_recall_pairwise_2classifier.append(
                            recall_pairwise_2classifier
                        )
                        print(recall_pairwise_2classifier)
                        # subset_exact_match_pairwise_2classifier = predictors._subset_exact_match(base_learner, predicted_Y, Y[test_index])
                        # average_subset_exact_match_pairwise_2classifier.append(subset_exact_match_pairwise_2classifier)
                        # print(subset_exact_match_pairwise_2classifier)
                        #                    print(np.mean([np.mean(x) for x in hard_predictions]))

                        print(
                            "====================== pairwise_3classifier ======================"
                        )
                        pairwise_3classifier = (
                            pairwise_classifiers._pairwise_3classifier(
                                base_learner, n_labels, X[train_index], Y[train_index]
                            )
                        )
                        predicted_Y, predicted_ranks = predictors._partialorders(
                            X[test_index],
                            pairwise_3classifier,
                            calibrated_2classifier,
                        )
                        #            print(hard_predictions)
                        # hamming_loss_pairwise_3classifier = predictors._hamming(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        hamming_loss_pairwise_3classifier = eval_metric.hamming_loss(
                            predicted_Y, Y[test_index]
                        )
                        average_hamming_loss_pairwise_3classifier.append(
                            hamming_loss_pairwise_3classifier
                        )
                        print(hamming_loss_pairwise_3classifier)
                        # f1_pairwise_3classifier = predictors._f1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        f1_pairwise_3classifier = eval_metric.f1(
                            predicted_Y, Y[test_index]
                        )
                        average_f1_pairwise_3classifier.append(f1_pairwise_3classifier)
                        print(f1_pairwise_3classifier)
                        # jaccard_pairwise_3classifier = predictors._jaccard(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        jaccard_pairwise_3classifier = eval_metric.jaccard(
                            predicted_Y, Y[test_index]
                        )
                        average_jaccard_pairwise_3classifier.append(
                            jaccard_pairwise_3classifier
                        )
                        print(jaccard_pairwise_3classifier)
                        # subset0_1_pairwise_3classifier = predictors._subset0_1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        subset0_1_pairwise_3classifier = eval_metric.subset0_1(
                            predicted_Y, Y[test_index]
                        )

                        subset0_1_pairwise_3classifier = eval_metric.subset0_1(
                            predicted_Y, Y[test_index]
                        )

                        average_subset0_1_pairwise_3classifier.append(
                            subset0_1_pairwise_3classifier
                        )
                        print(subset0_1_pairwise_3classifier)
                        # recall_pairwise_3classifier = predictors._recall(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        recall_pairwise_3classifier = eval_metric.recall(
                            predicted_Y, Y[test_index]
                        )
                        average_recall_pairwise_3classifier.append(
                            recall_pairwise_3classifier
                        )
                        print(recall_pairwise_3classifier)
                        # subset_exact_match_pairwise_3classifier = predictors._subset_exact_match(base_learner, predicted_Y, Y[test_index])
                        # average_subset_exact_match_pairwise_3classifier.append(subset_exact_match_pairwise_3classifier)
                        # print(subset_exact_match_pairwise_3classifier)

                        #                    print(np.mean([np.mean(x) for x in hard_predictions]))

                        print(
                            "====================== pairwise_4classifier ======================"
                        )
                        pairwise_4classifier = (
                            pairwise_classifiers._pairwise_4classifier(
                                base_learner, n_labels, X[train_index], Y[train_index]
                            )
                        )
                        predicted_Y, predicted_ranks = predictors._preorders(
                            X[test_index],
                            pairwise_4classifier,
                            calibrated_2classifier,
                        )
                        #            print(hard_predictions)
                        # hamming_loss_pairwise_4classifier = predictors._hamming(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        hamming_loss_pairwise_4classifier = eval_metric.hamming_loss(
                            predicted_Y, Y[test_index]
                        )

                        average_hamming_loss_pairwise_4classifier.append(
                            hamming_loss_pairwise_4classifier
                        )
                        print(hamming_loss_pairwise_4classifier)
                        # f1_pairwise_4classifier = predictors._f1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        f1_pairwise_4classifier = eval_metric.f1(
                            predicted_Y, Y[test_index]
                        )
                        average_f1_pairwise_4classifier.append(f1_pairwise_4classifier)
                        print(f1_pairwise_4classifier)

                        # jaccard_pairwise_4classifier = predictors._jaccard(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        jaccard_pairwise_4classifier = eval_metric.jaccard(
                            predicted_Y, Y[test_index]
                        )

                        average_jaccard_pairwise_4classifier.append(
                            jaccard_pairwise_4classifier
                        )
                        print(jaccard_pairwise_4classifier)
                        # subset0_1_pairwise_4classifier = predictors._subset0_1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        subset0_1_pairwise_4classifier = eval_metric.subset0_1(
                            predicted_Y, Y[test_index]
                        )

                        average_subset0_1_pairwise_4classifier.append(
                            subset0_1_pairwise_4classifier
                        )
                        print(subset0_1_pairwise_4classifier)
                        # recall_pairwise_4classifier = predictors._recall(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        recall_pairwise_4classifier = eval_metric.recall(
                            predicted_Y, Y[test_index]
                        )

                        average_recall_pairwise_4classifier.append(
                            recall_pairwise_4classifier
                        )
                        print(recall_pairwise_4classifier)
                        # subset_exact_match_pairwise_4classifier = predictors._subset_exact_match(base_learner, predicted_Y, Y[test_index])
                        # average_subset_exact_match_pairwise_4classifier.append(subset_exact_match_pairwise_4classifier)
                        # print(subset_exact_match_pairwise_4classifier)
                        #                    print(np.mean([np.mean(x) for x in hard_predictions]))
                        print("====================== ECC ======================")
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
                        chains = [
                            ClassifierChain(clf, order="random", random_state=i)
                            for i in range(10)
                        ]
                        for chain in chains:
                            chain.fit(X[train_index], Y[train_index])
                        Y_pred_chains = np.array(
                            [chain.predict(X[test_index]) for chain in chains]
                        )
                        Y_pred_ensemble = Y_pred_chains.mean(axis=0)
                        predicted_Y = np.where(Y_pred_ensemble > 0.5, 1, 0)

                        #                    ECC.fit(X[train_index].astype(float), Y[train_index].astype(float))
                        #                    predicted_Y = ECC.predict(X[test_index].astype(float))
                        print("====================== ECC ======================")
                        # hamming_loss_ECC = predictors._hamming(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        hamming_loss_ECC = eval_metric.hamming_loss(
                            predicted_Y, Y[test_index]
                        )

                        average_hamming_loss_ECC.append(hamming_loss_ECC)
                        print(hamming_loss_ECC)
                        # f1_ECC = predictors._f1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        f1_ECC = eval_metric.f1(predicted_Y, Y[test_index])
                        average_f1_ECC.append(f1_ECC)
                        print(f1_ECC)
                        # jaccard_ECC = predictors._jaccard(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        jaccard_ECC = eval_metric.jaccard(predicted_Y, Y[test_index])
                        average_jaccard_ECC.append(jaccard_ECC)
                        print(jaccard_ECC)
                        # subset0_1_ECC = predictors._subset0_1(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        subset0_1_ECC = eval_metric.subset0_1(
                            predicted_Y, Y[test_index]
                        )
                        average_subset0_1_ECC.append(subset0_1_ECC)
                        print(subset0_1_ECC)
                        # recall_ECC = predictors._recall(
                        #     base_learner, predicted_Y, Y[test_index]
                        # )
                        recall_ECC = eval_metric.recall(predicted_Y, Y[test_index])
                        average_recall_ECC.append(recall_ECC)
                        print(recall_ECC)

                        fold += 1
                    print(
                        "====================== Average results: ======================"
                    )
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

                final_results = [
                    [
                        np.mean(average_hamming_loss_pairwise_2classifier),
                        np.std(average_hamming_loss_pairwise_2classifier),
                        np.mean(average_hamming_loss_pairwise_3classifier),
                        np.std(average_hamming_loss_pairwise_3classifier),
                        np.mean(average_hamming_loss_pairwise_4classifier),
                        np.std(average_hamming_loss_pairwise_4classifier),
                        np.mean(average_hamming_loss_ECC),
                        np.std(average_hamming_loss_ECC),
                    ],
                    [
                        np.mean(average_f1_pairwise_2classifier),
                        np.std(average_f1_pairwise_2classifier),
                        np.mean(average_f1_pairwise_3classifier),
                        np.std(average_f1_pairwise_3classifier),
                        np.mean(average_f1_pairwise_4classifier),
                        np.std(average_f1_pairwise_4classifier),
                        np.mean(average_f1_ECC),
                        np.std(average_f1_ECC),
                    ],
                    [
                        np.mean(average_jaccard_pairwise_2classifier),
                        np.std(average_jaccard_pairwise_2classifier),
                        np.mean(average_jaccard_pairwise_3classifier),
                        np.std(average_jaccard_pairwise_3classifier),
                        np.mean(average_jaccard_pairwise_4classifier),
                        np.std(average_jaccard_pairwise_4classifier),
                        np.mean(average_jaccard_ECC),
                        np.std(average_jaccard_ECC),
                    ],
                    [
                        np.mean(average_subset0_1_pairwise_2classifier),
                        np.std(average_subset0_1_pairwise_2classifier),
                        np.mean(average_subset0_1_pairwise_3classifier),
                        np.std(average_subset0_1_pairwise_3classifier),
                        np.mean(average_subset0_1_pairwise_4classifier),
                        np.std(average_subset0_1_pairwise_4classifier),
                        np.mean(average_subset0_1_ECC),
                        np.std(average_subset0_1_ECC),
                    ],
                    [
                        np.mean(average_recall_pairwise_2classifier),
                        np.std(average_recall_pairwise_2classifier),
                        np.mean(average_recall_pairwise_3classifier),
                        np.std(average_recall_pairwise_3classifier),
                        np.mean(average_recall_pairwise_4classifier),
                        np.std(average_recall_pairwise_4classifier),
                        np.mean(average_recall_ECC),
                        np.std(average_recall_ECC),
                    ],
                ]

                res_file = "noisy_4_w_try_compareAcc_%i_%i_%i_%s_%s" % (
                    int(noisy_rate * 10),
                    total_repeat,
                    folds,
                    dataFile,
                    base_learner,
                )
                file = open(res_file, "w")
                file.writelines("%s\n" % line for line in final_results)
                file.close()

                # final_results = [list(np.mean(average_subset_exact_match_pairwise_2classifier, axis=0)),
                #                  list(np.mean(average_subset_exact_match_pairwise_3classifier, axis=0)),
                #                  list(np.mean(average_subset_exact_match_pairwise_4classifier, axis=0))]
                # res_file = "noisy_w_s_try_compareAcc_%i_%i_%s_%s" %(total_repeat, folds, dataFile, base_learner)
                # file = open(res_file, "w")
                # file.writelines("%s\n" %line for line in final_results)
                # file.close()
