# TODO: complete this file.

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_predictions,
    evaluate,
)
import numpy as np
import matplotlib as plt

from item_response import irt, sigmoid
from matrix_factorization import als

# Also import the Matrix factorization. 
# ALS 


# BOOTSTRAP THE DATASET. 
# From LEC03 Slides
#Take a dataset D with n examples.
#Generate m new datasets (“resamples” or “bootstrap samples”)
#Each dataset has n examples sampled from D with replacement.

def bootstrap_dataset(data):
    """Bootstrap the data to create three datasets
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :return: A list containing three data dictionaries
    """
    datasets = []
    for i in range(3):
        indexes = np.random.choice(len(data["is_correct"])-1, size=len(data["is_correct"]), replace=True)     #values 0-len(data), len(data) number of indexes, replace enables replacement
        user_list = []
        question_list = []
        correct_list = []
        for i in indexes:
            user_list.append(data["user_id"][i])
            question_list.append(data["question_id"][i])
            correct_list.append(data["is_correct"][i])
        bootstrap = {}
        bootstrap["user_id"] = user_list
        bootstrap["question_id"] = question_list
        bootstrap["is_correct"] = correct_list
        datasets.append(bootstrap)
    return datasets

def modified_evaluate(data, theta, beta):   
    """Evaluate the model given data and return the prediction
    *Note: returns list of predictions instead of accuracy now.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: List
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        if p_a > 0.5:
            pred.append(1.0)
        else:
            pred.append(0.0)
    return pred


def main():
    initial = load_train_csv()
    val_data = load_valid_csv()
    test_data = load_public_test_csv()

    datasets = bootstrap_dataset(initial)

    trainingset1 = datasets[0]
    trainingset2 = datasets[1]
    trainingset3 = datasets[2]
    
    # TRAIN IRT'S WITH DIFFERENT LR ITERATIONS
    LR1 = 0.01
    iter1 = 200
    LR2 = 0.001
    iter2 = 500
    
    theta1, beta1, val_acc_lst1, train_lld1, val_lld1 = irt(trainingset1, val_data, LR1, iter1)
    theta2, beta2, val_acc_lst2, train_lld2, val_lld2 = irt(trainingset2, val_data, LR2, iter2)
 
    # TRAIN MF USING ALS. USE THE K LR ITERATIONS THAT WERE USED IN MF.PY
    #  gto these from mf.py commnets.
    LR3 = 0.015
    iter3 = 3000
    k = 26 # came up as the best k wghen i ran mf.py

    mat, trainging_loss, valid_loss = als(trainingset1, val_data, k, LR3, iter3)

    # GET THE PREDICTIONS. USE Validation data now 
    val_pred1 = modified_evaluate(val_data, theta1, beta1)
    val_pred2 = modified_evaluate(val_data, theta2, beta2)
    val_pred3 = sparse_matrix_predictions(val_data, mat, threshold=0.5)
    
    # SUM THE PREDICTIONS. IF >= 2. THEN 1, ELSE 0
    prelim_pred = []
    final_pred = []
    for i  in range(len(val_pred1)):
        prelim_pred.append(val_pred1[i] + val_pred2[i] + val_pred3[i])
    for i in prelim_pred:
        if i >= 2:
            final_pred.append(1.0)
        else:
            final_pred.append(0.0)

    # GET THE PREDICTIONS. USE Test data now 
    test_pred1 = modified_evaluate(test_data, theta1, beta1)
    test_pred2 = modified_evaluate(test_data, theta2, beta2)
    test_pred3 = sparse_matrix_predictions(test_data, mat, threshold=0.5)
    
    # SUM THE PREDICTIONS. IF >= 2. THEN 1, ELSE 0
    test_prelim_pred = []
    test_final_pred = []
    for i  in range(len(test_pred1)):
        test_prelim_pred.append(test_pred1[i] + test_pred2[i] + test_pred3[i])
    for i in test_prelim_pred:
        if i >= 2:
            test_final_pred.append(1.0)
        else:
            test_final_pred.append(0.0)

    # DETERMINE ACCURACY
    val_acc = evaluate(val_data, final_pred, threshold=0.5)
    print(f"validation accuracy: {val_acc}")
    test_acc = evaluate(test_data, test_final_pred, threshold=0.5)
    print(f"test accuracy: {test_acc}")

# RESULTS
# validation accuracy: 0.6989839119390348
# test accuracy: 0.6948913350268134

if __name__ == "__main__":
    main()




