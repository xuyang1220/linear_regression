# Model:      y_hat = X @ w + b
# Loss (MSE): J(w, b) = (1 / 2n) * sum_i (y_hat_i - y_i)^2
# Gradients:  dJ/dw = (1/n) * X^T @ (y_hat - y)
#             dJ/db = (1/n) * sum(y_hat - y)

# Normalize the input
#  mu = X.mean(axis=0)
#  sigma = X.std(axis=0)
#  X = (X - mu) / sigma
#  w, b = w / sigma, b - np.sum(w * mu / sigma)

import numpy as np

class LinearRegression:
    def __init__(self, lr=0.1, n_iters=1000, stop_loss_delta=1e-9):
        self.lr = lr
        self.n_iters = n_iters
        self.stop_loss_delta = stop_loss_delta
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n, d = X.shape
        rng = np.random.default_rng(0)
        w = np.zeros(d)
        b = 0.0
        prev_loss = np.inf

        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0   # guard against constant columns
        X = (X - mu) / sigma

        for _ in range(self.n_iters):
            y_hat = X @ w  + b
            err = y_hat - y
            loss = 0.5 * np.mean(err ** 2)
            gradW = (X.transpose() @ err) / n
            gradb = np.mean(err)
            w -= self.lr * gradW
            b -= self.lr * gradb
            self.loss_history.append(loss)
            if abs(prev_loss-loss) < self.stop_loss_delta:
                break
            prev_loss = loss
        self.w, self.b = w / sigma, b - np.sum(w * mu / sigma)
        return self.w, self.b

    def predict(self, X):
        X = np.asarray(X, dtype = float)
        y = X @ self.w  + self.b
        return y


def normal_equation(X, y):
    """Closed-form solution as a correctness reference. Returns (w, b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])   # append bias column
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta[:-1], theta[-1]


def main():
    rng = np.random.default_rng(0)

    # Synthetic data: 5 features on wildly different scales + noise.
    n, d = 5000, 10
    X = rng.normal(size=(n, d)) * np.array([1.0, 100.0, 0.01, 50.0, 5.0, 1.0, 100.0, 0.01, 50.0, 5.0])
    true_w = np.array([3.0, -1.5, 40.0, 0.2, -7.0, 6.0, -3.5, 40.0, 180, -17.0])
    true_b = 125.0
    y = X @ true_w + true_b + rng.normal(scale=100.0, size=n)
    # print(X)
    # print(y)
    fitter = LinearRegression()
    w, b = fitter.fit(X, y)
    y_predict = fitter.predict(X)
    print(fitter.loss_history)

    w_ne, b_ne = normal_equation(X, y)

    def mse(a, b):
        return np.mean((a - b) ** 2)

    print(f"Iterations run        : {len(fitter.loss_history)}")
    print(f"Final training loss   : {fitter.loss_history[-1]:.6f}")
    print(f"Train MSE (GD)        : {mse(y_predict, y):.4f}")
    print(f"Train MSE (normal eq) : {mse(X @ w_ne + b_ne, y):.4f}\n")

    print("                  true        gradient-descent   normal-equation")
    for i in range(d):
        print(f"  w[{i}]        {true_w[i]:10.4f}   {fitter.w[i]:14.4f}   {w_ne[i]:14.4f}")
    print(f"  bias         {true_b:10.4f}   {fitter.b:14.4f}   {b_ne:14.4f}")


if __name__ == "__main__":
    main()