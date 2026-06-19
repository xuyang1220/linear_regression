"""
Multidimensional linear regression via batch gradient descent (NumPy only).

Model:      y_hat = X @ w + b
Loss (MSE): J(w, b) = (1 / 2n) * sum_i (y_hat_i - y_i)^2
Gradients:  dJ/dw = (1/n) * X^T @ (y_hat - y)
            dJ/db = (1/n) * sum(y_hat - y)

Features are standardized internally so a single learning rate works well
regardless of the scale of each input dimension. The learned parameters are
folded back to the original feature space at the end, so `predict` takes raw X.
"""

import numpy as np


class LinearRegressionGD:
    def __init__(self, lr=0.1, n_iters=1000, tol=1e-9, standardize=True):
        self.lr = lr
        self.n_iters = n_iters
        self.tol = tol                # early-stop when loss improvement < tol
        self.standardize = standardize
        self.w = None                 # weights in ORIGINAL feature space, shape (d,)
        self.b = None                 # intercept in ORIGINAL feature space, scalar
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape

        # Standardize features: z = (X - mu) / sigma. Helps GD converge when
        # dimensions have very different scales (otherwise the loss surface is
        # badly conditioned and you'd need a tiny lr).
        if self.standardize:
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma[sigma == 0] = 1.0   # guard against constant columns
            Xs = (X - mu) / sigma
        else:
            mu = np.zeros(d)
            sigma = np.ones(d)
            Xs = X

        # Parameters in the STANDARDIZED space.
        w = np.zeros(d)
        b = 0.0
        prev_loss = np.inf
        self.loss_history = []

        for _ in range(self.n_iters):
            y_hat = Xs @ w + b
            err = y_hat - y                      # shape (n,)

            loss = 0.5 * np.mean(err ** 2)
            self.loss_history.append(loss)

            grad_w = (Xs.T @ err) / n            # shape (d,)
            grad_b = err.mean()                  # scalar

            w -= self.lr * grad_w
            b -= self.lr * grad_b

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        # Fold standardization back into the parameters so predict() works on
        # raw X:  y = w_s . ((X - mu)/sigma) + b_s
        #           = (w_s / sigma) . X + (b_s - sum(w_s * mu / sigma))
        self.w = w / sigma
        self.b = b - np.sum(w * mu / sigma)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b


def normal_equation(X, y):
    """Closed-form solution as a correctness reference. Returns (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])   # append bias column
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta[:-1], theta[-1]


if __name__ == "__main__":
    # print("noob")
    rng = np.random.default_rng(0)

    # Synthetic data: 5 features on wildly different scales + noise.
    n, d = 500, 5
    X = rng.normal(size=(n, d)) * np.array([1.0, 100.0, 0.01, 50.0, 5.0])
    true_w = np.array([3.0, -1.5, 40.0, 0.2, -7.0])
    true_b = 12.0
    y = X @ true_w + true_b + rng.normal(scale=2.0, size=n)

    model = LinearRegressionGD(lr=0.3, n_iters=5000).fit(X, y)
    w_ne, b_ne = normal_equation(X, y)

    def mse(a, b_):
        return np.mean((a - b_) ** 2)

    print(f"Iterations run        : {len(model.loss_history)}")
    print(f"Final training loss   : {model.loss_history[-1]:.6f}")
    print(f"Train MSE (GD)        : {mse(model.predict(X), y):.4f}")
    print(f"Train MSE (normal eq) : {mse(X @ w_ne + b_ne, y):.4f}\n")

    print("                  true        gradient-descent   normal-equation")
    for i in range(d):
        print(f"  w[{i}]        {true_w[i]:10.4f}   {model.w[i]:14.4f}   {w_ne[i]:14.4f}")
    print(f"  bias       {true_b:10.4f}   {model.b:14.4f}   {b_ne:14.4f}")