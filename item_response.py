from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0

    for iteration in range(len(data["is_correct"])):
        i = data["user_id"][iteration]
        j = data["question_id"][iteration]
        c_ij = data["is_correct"][iteration] # c_ij
        # log p = (c_ij)log(sigmoid(theta_i - beta_j)) + (1 - c_ij)(log(1 - sigmoid(theta_i - beta_j)))
        log_lklihood += c_ij * np.log(sigmoid(theta[i] - beta[j])) + (1 - c_ij) * np.log(1 - sigmoid(theta[i] - beta[j]))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # We don't need to consider the number of iteration since that will be done in IRT
    # Simply need to iterate through the size of the data (len of the lists)
    # And perform new_theta += 

    for iteration in range(len(data["is_correct"])):
        i = data["user_id"][iteration]
        j = data["question_id"][iteration]
        c_ij = data["is_correct"][iteration] # c_ij
        update = lr * (c_ij - sigmoid(theta[i] - beta[j]))
        theta[i] += update

    # need to do seperately so beta only uses the updated theta value
    for iteration in range(len(data["is_correct"])):
        i = data["user_id"][iteration]
        j = data["question_id"][iteration]
        c_ij = data["is_correct"][iteration] # c_ij
        update = lr * (sigmoid(theta[i] - beta[j]) - c_ij)
        beta[j] += update

    return theta, beta
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542) # 542 students listed in student metadata 
    beta = np.zeros(1774) # 1774 questions listed in question meta

    val_acc_lst = []
    train_lld = []
    val_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)

        # lld's needed for (b)
        # Divided in order to normalize the log-likelihood
        # Stated in a piazza post. 

        normalized_neg_lld = neg_lld / 56688 
        normalized_val_neg_lld = neg_lld_val / 7086
        train_lld.append(normalized_neg_lld)
        val_lld.append(normalized_val_neg_lld)

        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld, val_lld


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])




def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01   # Use 0.01
    iterations = 200    # Use 200. as it plateaus there anyways

    print(f"Hyperparameter: Larning Rate = {lr}, Iterations = {iterations}")
    
    theta, beta, val_acc_lst, train_lld, val_lld = irt(train_data, val_data, lr, iterations)

    val_acc = evaluate(val_data, theta, beta)
    print(f"validation accuracy: {val_acc}")
    test_acc = evaluate(test_data, theta, beta)
    print(f"test accuracy: {test_acc}")

    
    # validation accuracy: 0.7070279424216765
    # test accuracy: 0.7084391758396839

    # Need to work on Plots for (b) and (d)
    # Need to normalize or average the Neg _lld
    plt.figure()
    #plt.plot(train_lld, label="Training Loss")
    #plt.plot(val_lld, label="Validation Loss")
    #plt.legend()
    #plt.xlabel("Iterations")
    #plt.ylabel("Log-likelihood")
    #plt.show()
    #plt.savefig("A2b-2.svg")

    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    ####################################################################
   
    # Select 3 questions from beta. 
    
    question1 = beta[0]
    question2 = beta[len(beta)//2]
    question3 = beta[len(beta)-1]

    #plt.plot(theta, sigmoid(theta - question1), label="j1")
    #plt.plot(theta, sigmoid(theta - question2), label="j2")
    #plt.plot(theta, sigmoid(theta - question3), label="j3")
    # Can see the shape of the curve, but its very messy and illegible since the curve goes back and forth to the last point.
    # Trying scatterplot to see if its cleaner.

    plt.scatter(theta, sigmoid(theta - question1), label="j1")
    plt.scatter(theta, sigmoid(theta - question2), label="j2")
    plt.scatter(theta, sigmoid(theta - question3), label="j3")
    plt.xlabel("theta")
    plt.ylabel("P(c_ij) = 1")
    #plt.show()
    plt.savefig("A2d.svg")
    # 1 of the lines is not quite visible. The blue one just barely peaks out behind the green one
    

    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
