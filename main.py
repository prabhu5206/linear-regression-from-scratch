
import numpy as np


def predict(x, w, b):
    return w * x + b


def mean_squared_error(actual, predicted):
    errors = predicted - actual
    squared_errors = errors ** 2
    return squared_errors.mean()



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


hours = np.array([1, 2, 3, 4, 5], dtype=float)
actual_marks = np.array([35, 40, 50, 55, 65], dtype=float)

w, b, loss_history = train(
    hours,
    actual_marks,
    w=0.0,
    b=0.0,
    learning_rate=0.01,
    epochs=1000
)

print("\nTraining complete!")
print("Learned weight:", w)
print("Learned bias:", b)
print("Final MSE:", mean_squared_error(actual_marks, predict(hours, w, b)))

import matplotlib.pyplot as plt

predicted_marks = predict(hours, w, b)

plt.scatter(hours, actual_marks, label="Actual marks")
plt.plot(hours, predicted_marks, label="Learned regression line")

plt.xlabel("Hours studied")
plt.ylabel("Marks")
plt.title("Linear Regression From Scratch")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, len(loss_history) + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()