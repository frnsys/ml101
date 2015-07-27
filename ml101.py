import numpy as np
import pandas as pd

true_params = np.array([3, 120, -0.4])


def f(X):
    """
    The true function, with some noise
    """
    return np.dot(true_params, X.T) + np.random.randn(X.shape[0])


def sample_data(n_samples=10000):
    """
    Simulate data

    Features:
    - current heart rate
    - has criminal relative
    - number of Facebook friends
    """
    X = np.hstack([
            np.random.normal(70, 10, (n_samples, 1)),
            np.random.binomial(1, 0.2, (n_samples, 1)),
            np.random.binomial(600, 0.7, (n_samples, 1))
        ])

    # 20% have very few friends
    n_friendless = int(n_samples * 0.2)
    X[np.random.choice(X.shape[0], n_friendless),2] = np.random.binomial(600, 0.5, (n_friendless,))

    # Compute true values
    y = f(X)

    # Format wrangling
    data = np.hstack([X, y[:,np.newaxis]])
    return pd.DataFrame(data, columns=['heart_rate',
                                       'has_criminal_relative',
                                       'num_facebook_friends',
                                       'crime_coef'])