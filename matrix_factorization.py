import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix 
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    #gradient = 2("x" - c)
    u[n] = u[n] - (lr * (u[n] - c))
    z[q] = z[q] - (lr * (z[q] - c))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_loss = []
    val_loss = []

    for i in range(num_iteration):
        #Run SGD
        u, z = update_u_z(train_data, lr, u, z)
        train_loss.append(squared_error_loss(train_data, u, z))
        val_loss.append(squared_error_loss(val_data, u, z))

    mat = np.dot(u, np.transpose(z))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_loss, val_loss


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    #ks = [1, 6, 11, 16, 21, 26, 30]
    ks = [26]
    best_k_s = 0
    best_a_s = 0
    # Find the most accurate k value
    for k in ks:
        svd = svd_reconstruct(train_matrix, k)
        out = sparse_matrix_evaluate(val_data, svd)
        if out > best_a_s:
            best_k_s, best_a_s = k, out
        print("Validation Accuracy: {}".format(out))
        test_out = sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, best_k_s))
        print(f"test performance: {test_out}")
    print(f"Best k: {best_k_s}")
    print(f"Best accuracy: {best_a_s}")
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    best_k_a = 0
    best_a_a = 0
    best_mat = 0
    best_training_loss = 0
    best_validation_loss = 0
    #      vvvv number of repetitions here
    reps = 3500
    
    #Loop through k values and store best
    for k in ks:
        mat, training_loss, validation_loss = als(train_data, val_data, k, 0.015, reps)
        print("Done ALS")
        als_val = sparse_matrix_evaluate(val_data, mat)
        if als_val > best_a_a:
            best_k_a, best_a_a, best_mat, best_training_loss, best_validation_loss = k, als_val, mat, training_loss, validation_loss
        print("Validation Accuracy: {}".format(als_val))
    
    #Plots and terminal output
    print(f"Best k: {best_k_a}")
    print(f"learning rate: TBA")
    print(f"number of iterations: TBA")
    print(f"Best accuracy: {best_a_a}")
    als_test = sparse_matrix_evaluate(test_data, best_mat)
    print(f"test performance: {als_test}")
    print(f"validation performance: {best_a_a}")
    
   
    #Plot
    plt.plot(np.arange(reps), best_training_loss, label="Training Loss")
    plt.plot(np.arange(reps), best_validation_loss, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()