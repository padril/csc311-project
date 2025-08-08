import numpy as np
from sklearn.impute import KNNImputer
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_question_metadata,
    load_train_sparse,
    evaluate,
)
from item_response import evaluate as evaluate_irt
from tqdm import tqdm


# 388 originally
K_VALUES = [40,80,120,160]
ITS = 2000

LR = 1e-3
LR_a = 5e-4

def onehot(Q):
    """
    Convert question metadata into onehot encoding

    Returns Q1 of size |q| x |t| where |q| is the number of questions and |t|
    is the number of subjects
    """
    absq = len(Q["question_id"])
    abst = max(sum(Q["subject_id"], [])) + 1
    Q1 = np.zeros((absq, abst))
    for q, ts in zip(Q["question_id"], Q["subject_id"]):
        oh = np.zeros(abst)
        oh[ts] = 1
        Q1[q] = oh
    return Q1


def train_svd(Q1, k):
    """
    Returns latent encoding of the question types, which is 
    |q| x k
    Where k is the latent dimension hyperparameter and |q| is the number
    of question types
    Uses unsupervised SVD
    """
    # Compute the average and subtract it.
    item_means = np.mean(Q1, axis=0)
    mu = np.tile(item_means, (Q1.shape[0], 1))
    Q1 = Q1 - mu

    # Perform SVD.
    Q, s, _ = np.linalg.svd(Q1, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    mu = mu[:, 0:k]
    s_root = np.array(sqrtm(s))

    # Dimensionality reduction 
    L = np.dot(Q, s_root)
    L = L + mu
    return np.array(L)


def compute_aptitude(H, L):
    """
    Computes S, which is essentially the "student aptitude" using L and H

    Returns S, which is of size |s| x k where |s| is the number of students
    and k is the dimension of the latent space of L
    """
    _, k = L.shape
    abss, _ = H.shape
    S = np.zeros((abss, k))
    for i in range(abss):
        bb1 = ~np.isnan(H[i])[:, None]
        p = ((2 * bb1 - 1) * L).sum(axis=0)
        q = bb1.sum()
        S[i] = p / q
    return S


def sigmoid(x):
    """Apply sigmoid function."""
    return np.where(
            x >= 0, # condition
            1 / (1 + np.exp(-x)), # For positive values
            np.exp(x) / (1 + np.exp(x)) # For negative values
    )

def train_irt(H, S, L, val):
    """
    Train Modified IRT model.
    """
    _, k = L.shape
    np.random.seed(42)
    Wth = np.random.randn(k, k) * 0.1
    Wb = np.random.randn(k, k) * 0.1
    a = 5
    b = 0

    # perform gradient descent on Wth and Wb

    # Adam state
    adam_m_Wth = 0
    adam_m_Wb = 0
    adam_m_a = 0
    adam_m_b = 0
    adam_v_Wth = 0
    adam_v_Wb = 0
    adam_v_a = 0
    adam_v_b = 0
    adam_t = 0
    b1, b2 = 0.9, 0.999

    mask = ~np.isnan(H)
    for i in tqdm(range(ITS)):
        theta = S @ Wth
        beta = L @ Wb
        abstheta = np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12
        absbeta = np.linalg.norm(beta, axis=1, keepdims=True) + 1e-12
        cs = (theta @ beta.T) / (abstheta @ absbeta.T)
        g = sigmoid(a * cs + b)
        g = np.clip(g, 1e-12, 1-1e-12)

        lld = np.zeros_like(H)
        lld[mask] = H[mask] * np.log(g[mask]) + (1 - H[mask]) * np.log(1 - g[mask])
        nlld = -lld.sum()

        if i % (ITS // 10) == 0:
            score = evaluate_irt(val, theta, beta)
            print(f"({i}) NLLD: {nlld}\tScore: {score}\na: {a}\tb: {b}")
            print(f"g: {g.min()}, {g.mean()}, {g.max()}")
        
        D = np.zeros_like(H)
        D[mask] = g[mask] - H[mask]
        grad_th = (((D @ (beta / absbeta)) / abstheta)
                   - ((D * cs).sum(axis=1, keepdims=True)
                      / (abstheta * abstheta)) * theta)
        grad_bt = (((D.T @ (theta / abstheta)) / absbeta)
                  - ((D.T * cs.T).sum(axis=1, keepdims=True)
                     / (absbeta * absbeta)) * beta)
        grad_Wth = S.T @ grad_th
        grad_Wb = L.T @ grad_bt
        grad_a = (D * cs)[mask].sum()     # equivalently: np.sum(D * cs)
        grad_b  = D[mask].sum()

        adam_t += 1

        lr = LR
        lr_a = LR_a
        lr_b = LR
        if i <= 0.05 * ITS:
            lr *= 5

        adam_m_Wth = b1 * adam_m_Wth + (1 - b1) * grad_Wth
        adam_v_Wth = b2 * adam_v_Wth + (1 - b2) * (grad_Wth ** 2)
        m_hat = adam_m_Wth / (1 - b1 ** adam_t)
        v_hat = adam_v_Wth / (1 - b2 ** adam_t)
        Wth -= lr * m_hat / (np.sqrt(v_hat) + 1e-12)

        adam_m_Wb = b1 * adam_m_Wb + (1 - b1) * grad_Wb
        adam_v_Wb = b2 * adam_v_Wb + (1 - b2) * (grad_Wb ** 2)
        m_hat = adam_m_Wb / (1 - b1 ** adam_t)
        v_hat = adam_v_Wb / (1 - b2 ** adam_t)
        Wb -= lr * m_hat / (np.sqrt(v_hat) + 1e-12)

        adam_m_a = b1 * adam_m_a + (1 - b1) * grad_a
        adam_v_a = b2 * adam_v_a + (1 - b2) * (grad_a ** 2)
        m_hat = adam_m_a / (1 - b1 ** adam_t)
        v_hat = adam_v_a / (1 - b2 ** adam_t)
        a -= lr_a * m_hat / (np.sqrt(v_hat) + 1e-12)

        adam_m_b = b1 * adam_m_b + (1 - b1) * grad_b
        adam_v_b = b2 * adam_v_b + (1 - b2) * (grad_b ** 2)
        m_hat = adam_m_b / (1 - b1 ** adam_t)
        v_hat = adam_v_b / (1 - b2 ** adam_t)
        b -= lr_b * m_hat / (np.sqrt(v_hat) + 1e-12)

    return Wth, Wb, a, b


def decode(S, L, Wth, Wb, a, b, ref):
    hyp = []
    for u, q, _ in zip(ref["user_id"], ref["question_id"], ref["is_correct"]):
        theta = S[u].T @ Wth
        beta = L[q].T @ Wb
        abstheta = np.linalg.norm(theta)
        absbeta = np.linalg.norm(beta)
        cs = (theta @ beta) / (abstheta * absbeta)
        g = sigmoid(a * cs + b)
        hyp.append(g)
    return hyp
        



def get_acc(H, Q1, k, ref):
    L = train_svd(Q1, k)
    S = compute_aptitude(H, L)
    Wth, Wb, a, b = train_irt(H, S, L, ref)
    hyp = decode(S, L, Wth, Wb, a, b, ref)
    return evaluate(ref, hyp)


def main():
    H = load_train_sparse("./data").toarray()
    Q = load_question_metadata("./data")
    Q1 = onehot(Q)
    val = load_valid_csv("./data")
    test = load_public_test_csv("./data")

    accs = []
    kstar, accstar = np.nan, 0
    for k in K_VALUES:
        acc = get_acc(H, Q1, k, val)
        accs.append(acc)
        if acc >= accstar:
            kstar, accstar = k, acc
        print("k: {} \t Acc: {}".format(k, acc))

    plt.plot(K_VALUES, accs)
    plt.xlabel("$k$-values")
    plt.ylabel("Accuracy")
    plt.savefig("partb_k_search.svg")

    # testacc = get_acc(H, Q1, kstar, test)
    # print(f"k*:\t{kstar}\nTest acc:\t{testacc:.4f}")


if __name__ == "__main__":
    main()
