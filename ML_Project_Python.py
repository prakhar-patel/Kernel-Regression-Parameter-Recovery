import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel
from numpy.linalg import svd, lstsq

# You may define your own variables here
_K_DEGREE = 3
_K_COEF0 = 1.0
_RIDGE_LAMBDA = 1e-6


################################
# Non Editable Region Starting #
################################
def my_kernel(X1, Z1, X2, Z2):
################################
#  Non Editable Region Ending  #
################################
    # Convert to numpy arrays
    X1 = np.asarray(X1).reshape(-1)
    X2 = np.asarray(X2).reshape(-1)
    Z1 = np.asarray(Z1).reshape(len(X1), -1)
    Z2 = np.asarray(Z2).reshape(len(X2), -1)

    # Polynomial kernel on Z
    Kz = polynomial_kernel(Z1, Z2, degree=_K_DEGREE, coef0=_K_COEF0)

    # Outer product X1 * X2^T
    outer = np.multiply.outer(X1, X2)

    # Combined kernel: x1*x2*Kz + 1
    G = outer * Kz
    G = G + 1.0  # bias term

    return G


################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################
    w = np.asarray(w).reshape(-1)
    if w.size != 1089:
        raise ValueError("Input w must be length 1089.")

    d_dim = int(np.sqrt(w.size))  # 33
    W = w.reshape(d_dim, d_dim)

    # SVD rank-1 decomposition
    U, S, Vt = svd(W, full_matrices=False)
    sigma0 = S[0]
    u_vec = U[:, 0] * np.sqrt(sigma0)
    v_vec = Vt[0, :] * np.sqrt(sigma0)

    # Fix sign ambiguity
    if np.abs(u_vec[0]) > 1e-12:
        sign = np.sign(u_vec[0])
    else:
        sign = np.sign(u_vec.sum()) if np.abs(u_vec.sum()) > 1e-12 else 1.0
    u_vec *= sign
    v_vec *= sign

    # Build A matrix (33 x 128)
    n_mux = d_dim - 1  # 32
    n_vars = 4 * n_mux  # 128
    A = np.zeros((d_dim, n_vars))

    def idx_p(i): return i
    def idx_q(i): return n_mux + i
    def idx_r(i): return 2 * n_mux + i
    def idx_s(i): return 3 * n_mux + i

    # Row 0
    A[0, idx_p(0)] = 0.5
    A[0, idx_q(0)] = -0.5
    A[0, idx_r(0)] = 0.5
    A[0, idx_s(0)] = -0.5

    # Rows 1..31
    for i in range(1, d_dim - 1):
        # α_i term
        A[i, idx_p(i)] += 0.5
        A[i, idx_q(i)] += -0.5
        A[i, idx_r(i)] += 0.5
        A[i, idx_s(i)] += -0.5

        # β_(i−1) term
        A[i, idx_p(i - 1)] += 0.5
        A[i, idx_q(i - 1)] += -0.5
        A[i, idx_r(i - 1)] += -0.5
        A[i, idx_s(i - 1)] += 0.5

    # Last row (row 32): w32 = beta31 = 0.5*(p31 - q31 - r31 + s31)
    last_row = 32        # row index for w32
    last_stage = 31      # stage index 31 -> p31,q31,r31,s31 (valid 0..31)

    A[last_row, idx_p(last_stage)] = 0.5
    A[last_row, idx_q(last_stage)] = -0.5
    A[last_row, idx_r(last_stage)] = -0.5
    A[last_row, idx_s(last_stage)] = 0.5
    

    # Solver
    def solve_model(vec):
        vec = np.asarray(vec).reshape(-1)
        ATA = A.T.dot(A) + _RIDGE_LAMBDA * np.eye(n_vars)
        ATb = A.T.dot(vec)

        try:
            x = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            x, *_ = lstsq(ATA, ATb, rcond=None)

        return np.maximum(x, 0)  # enforce non-negativity

    # Solve for u and v
    x1 = solve_model(u_vec)
    x2 = solve_model(v_vec)

    # Split into 4 delay vectors
    def split4(x):
        return (
            x[0:n_mux],
            x[n_mux:2*n_mux],
            x[2*n_mux:3*n_mux],
            x[3*n_mux:4*n_mux]
        )

    a, b, c, d = split4(x1)
    p, q, r, s = split4(x2)

    # Enforce non-negativity
    a = np.maximum(a, 0)
    b = np.maximum(b, 0)
    c = np.maximum(c, 0)
    d = np.maximum(d, 0)
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    r = np.maximum(r, 0)
    s = np.maximum(s, 0)

    
    return a, b, c, d, p, q, r, s
