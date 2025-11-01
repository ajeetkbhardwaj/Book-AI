import numpy as np

def mean_absolute_error(y_true, y_pred):
    """ MAE: Average absolute difference """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """ MSE: Average squared difference """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """ RMSE: Square root of MSE """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """ R²: 1 - (SS_res / SS_tot) """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def adjusted_r2(y_true, y_pred, n_features):
    """ Adjusted R²: Adjusted for number of predictors """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def mean_absolute_percentage_error(y_true, y_pred):
    """ MAPE: Mean absolute percentage error """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


def symmetric_mape(y_true, y_pred):
    """ SMAPE: Symmetric mean absolute percentage error """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100


def median_absolute_error(y_true, y_pred):
    """ Median absolute error (robust to outliers) """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs(y_true - y_pred))

if __name__=='__main__':
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    # Synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

   # Train a simple linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_true = y
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", root_mean_squared_error(y_true, y_pred))
    print("R²:", r2_score(y_true, y_pred))
    print("Adjusted R²:", adjusted_r2(y_true, y_pred, n_features=1))
    print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
    print("SMAPE:", symmetric_mape(y_true, y_pred))
    print("Median AE:", median_absolute_error(y_true, y_pred))