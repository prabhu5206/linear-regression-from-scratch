import numpy as np
def predict(x, w, b):
    return w * x + b


def mean_squared_error(actual, predicted):
    errors = predicted - actual
    squared_errors = errors ** 2
    return squared_errors.mean()


def r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    if ss_tot == 0:
        return float("nan")

    return 1 - (ss_res / ss_tot)




def train(x, y, w, b, learning_rate, epochs):
    n = len(x)
    loss_history = []

    for epoch in range(epochs):
        predictions = predict(x, w, b)
        errors = predictions - y

        dw = (2 / n) * np.sum(x * errors)
        db = (2 / n) * np.sum(errors)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        mse = mean_squared_error(y, predict(x, w, b))
        loss_history.append(mse)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, MSE: {mse:.4f}")

    return w, b, loss_history

