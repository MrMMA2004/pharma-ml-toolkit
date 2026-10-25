#!/usr/bin/env python3
"""Numerical gate: manual Laplace GP classification vs sklearn.

The web mirror's fitGaussianProcessClassification (ml-math.ts) is a
hand-rolled port of GPML Algorithm 3.1 + 3.2 exactly as sklearn
implements them (Newton mode-finding with the B = I + W^(1/2) K W^(1/2)
Cholesky; Williams-Barber 5-erf sigmoid integral for the predictive
probability).  This script proves the formulas in numpy FIRST, against
sklearn's GaussianProcessClassifier with a pinned kernel
(optimizer=None), so the TypeScript port has a verified reference.

Run:  python scratch/verify_gpc_parity.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve
from scipy.special import erf, expit
from sklearn.datasets import make_moons
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# sklearn's exact sigmoid-integral constants (Williams & Barber 1998,
# logistic CDF as a linear combination of 5 error functions).
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])
COEFS = np.array(
    [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]
)


def manual_laplace_gpc(K, y, K_star, k_diag, max_iter=100):
    """GPML Alg 3.1 (Newton mode) + Alg 3.2 (predictive), sklearn-exact.

    Mirrors sklearn's _posterior_mode loop structure INCLUDING the
    off-by-one-iteration temporaries (the returned pi / W_sr / L come
    from the top of the final executed iteration, exactly as sklearn
    stores them) -- the price of bit-level parity with the reference.
    """
    n = len(y)
    f = np.zeros(n)
    lml_prev = -np.inf
    iters = 0
    pi = W_sr = L = None
    for _ in range(max_iter):
        iters += 1
        pi = expit(f)
        W = pi * (1 - pi)
        W_sr = np.sqrt(W)
        W_sr_K = W_sr[:, None] * K
        B = np.eye(n) + W_sr_K * W_sr
        L = cholesky(B, lower=True)
        b = W * f + (y - pi)
        a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
        f = K.dot(a)
        lml = (
            -0.5 * a.T.dot(f)
            - np.log1p(np.exp(-(y * 2 - 1) * f)).sum()
            - np.log(np.diag(L)).sum()
        )
        if lml - lml_prev < 1e-10:
            break
        lml_prev = lml

    # Alg 3.2 with the stored temporaries.  NOTE the two DIFFERENT
    # triangular solves: the Newton loop uses cho_solve((L, True), ...) --
    # solving B x = rhs with B = L L^T (the SPD Newton system) -- while
    # the predictive variance uses solve(L, ...): plain FORWARD
    # substitution, v = L^-1 rhs, so that sum(v^2) = rhs^T B^-1 rhs
    # = k*^T (K + W^-1)^-1 k* (the R&W eq 3.26 variance).  Using
    # cho_solve here instead would square the B^-1 -- the exact slip
    # this script's first version made (proba off by 3e-2).
    q = y - pi  # = K^{-1} f_hat at the mode
    f_star = K_star.T.dot(q)
    v = solve(L, W_sr[:, None] * K_star)
    var_f_star = k_diag - np.einsum("ij,ij->j", v, v)
    alpha = 1 / (2 * var_f_star)
    gamma = LAMBDAS[:, None] * f_star
    integrals = (
        np.sqrt(np.pi / alpha)
        * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS[:, None] ** 2)))
        / (2 * np.sqrt(var_f_star * 2 * np.pi))
    )
    pi_star = (COEFS[:, None] * integrals).sum(axis=0) + 0.5 * COEFS.sum()
    return pi_star, iters, lml_prev


def main() -> None:
    rng = np.random.default_rng(7)
    X, y = make_moons(n_samples=120, noise=0.2, random_state=7)
    X_test = rng.uniform(-1.5, 2.5, size=(25, 2))

    signal_std = 1.2
    length_scale = 0.8
    kernel = ConstantKernel(signal_std**2) * RBF(length_scale=length_scale)
    clf = GaussianProcessClassifier(kernel=kernel, optimizer=None, random_state=42)
    clf.fit(X, y)
    sk_proba = clf.predict_proba(X_test)[:, 1]
    sk_lml = clf.log_marginal_likelihood()

    K = kernel(X)
    K_star = kernel(X, X_test)
    k_diag = kernel.diag(X_test)
    my_proba, iters, my_lml = manual_laplace_gpc(K, y.astype(float), K_star, k_diag)

    d = np.abs(sk_proba - my_proba).max()
    print(f"iterations to mode:        {iters}")
    print(f"sklearn lml:               {sk_lml:.10f}")
    print(f"manual   lml:              {my_lml:.10f}")
    print(f"lml delta:                 {abs(sk_lml - my_lml):.3e}")
    print(f"max |proba delta|:         {d:.3e}")
    print(f"proba range:               [{my_proba.min():.4f}, {my_proba.max():.4f}]")

    # Latent-quantity parity (formula-exact, no integral approximation).
    # sklearn 1.5 wraps the binary Laplace engine in base_estimator_.
    base = getattr(clf, "base_estimator_", clf)
    f_star_sk = clf.kernel_(X, X_test).T.dot(base.y_train_ - base.pi_)
    f_star_my = K_star.T.dot(y - base.pi_)
    print(f"max |latent mean delta|:   {np.abs(f_star_sk - f_star_my).max():.3e}")

    ok = d < 1e-8 and abs(sk_lml - my_lml) < 1e-8
    print("GATE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
