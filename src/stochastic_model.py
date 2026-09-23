import numpy as np
from sklearn.linear_model import LinearRegression


def simulate_ou(x0, mu, kappa, sigma, n_steps, dt=1.0):
    x = np.zeros(n_steps)
    x[0] = x0

    for t in range(1, n_steps):
        z = np.random.normal(0, 1)

        mean = mu + np.exp(-kappa * dt) * (x[t-1] - mu)

        std = sigma * np.sqrt(
            (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
        )

        x[t] = mean + std * z

    return x

def estimate_ou_parameters(x, dt = 1):
        x_t = x[:-1]
        x_t = x_t.reshape(-1, 1)
        x_next = x[1:]
        model = LinearRegression()
        model.fit(x_t, x_next)

        intercept = model.intercept_
        slope = model.coef_[0]

        kappa = -np.log(slope) / dt
        mu = intercept / (1 - slope)

        predicted = model.predict(x_t)
        residuals = x_next - predicted
        residual_std = residuals.std()

        sigma = residual_std * np.sqrt(
            (2 * kappa) / (1 - np.exp(-2 * kappa * dt))
        )

        return kappa, mu, sigma


def simulate_ou_paths(x0,mu, kappa, sigma, n_steps, n_paths, dt=1.0):
    paths = np.zeros((n_steps, n_paths))
    paths[0, :] = x0

    for t in range(1, n_steps):
        z = np.random.normal(0, 1, size=n_paths)
    
        mean = mu + np.exp(-kappa * dt) * (paths[t-1, :] - mu)
    
        std = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    
        paths[t, :] = mean + std * z
    
    return paths

