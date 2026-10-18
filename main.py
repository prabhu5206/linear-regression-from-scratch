
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


hours = np.array(
    [1, 2, 3, 4, 5, 2.5, 3.5, 4.5, 5.5, 6],
    dtype=float
)

actual_marks = np.array(
    [35, 40, 50, 55, 65, 45, 52, 60, 68, 75],
    dtype=float
)

# First 7 students for training
x_train = hours[:7]
y_train = actual_marks[:7]

# Last 3 students for testing
x_test = hours[7:]
y_test = actual_marks[7:]

w, b, loss_history = train(
    x_train,
    y_train,
    w=0.0,
    b=0.0,
    learning_rate=0.01,
    epochs=1000
)

print("\nTraining complete!")
print("Learned weight:", w)
print("Learned bias:", b)
print("Final MSE:", mean_squared_error(actual_marks, predict(hours, w, b)))
predicted_marks = predict(hours, w, b)
print("R-squared:", r_squared(actual_marks, predicted_marks))


import matplotlib.pyplot as plt

predicted_marks = predict(hours, w, b)
# Evaluate on the unseen test student
test_prediction = predict(x_test, w, b)
test_mae = np.mean(np.abs(y_test - test_prediction))
test_mean_error = np.mean(test_prediction - y_test)
print("Test Mean Error:", test_mean_error)

print("Test MAE:", test_mae)

for i in range(len(y_test)):
    print(f"\nTest Student {i + 1}")
    print("Actual marks:", y_test[i])
    print("Predicted marks:", test_prediction[i])
    print("Absolute error:", abs(y_test[i] - test_prediction[i]))

# Visualize actual vs predicted test marks
# Visualize actual vs predicted test marks
plt.figure(figsize=(8, 5))

students = np.arange(len(y_test))
width = 0.3

plt.bar(
    students - width * 0.7,
    y_test,
    width,
    label="Actual Marks"
)

plt.bar(
    students + width *0.7,
    test_prediction,
    width,
    label="Predicted Marks"
)

plt.title("Actual vs Predicted Marks (Test Data)")
plt.xlabel("Test Student")
plt.ylabel("Marks")
plt.xticks(students, [f"Student {i + 1}" for i in students])
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
plt.bar_label(bars, fmt="%.2f", padding=3)

plt.title("Actual vs Predicted Marks (Test Data)")
plt.ylabel("Marks")
plt.ylim(0, 100)

plt.legend(loc="upper right")
plt.subplots_adjust(right=0.75, bottom=0.2)
plt.show()
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


# Predict marks for a new number of study hours
new_hours = float(input("Enter study hours: "))

predicted_new_marks = predict(new_hours, w, b)

print(f"Predicted marks: {predicted_new_marks:.2f}")
