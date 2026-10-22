from model import predict, mean_squared_error, r_squared, train
import numpy as np

data = np.genfromtxt(
    "student_data.csv",
    delimiter=",",
    names=True
)

hours = data["hours"]
actual_marks = data["marks"]

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
train_prediction = predict(x_train, w, b)
train_mse = mean_squared_error(y_train, train_prediction)
print("Training MSE:", train_mse)
all_data_mse = mean_squared_error(
    actual_marks,
    predict(hours, w, b)
)
print("All-data MSE (training + test):", all_data_mse)
predicted_marks = predict(hours, w, b)
train_r2 = r_squared(y_train, train_prediction)
print("Training R-squared:", train_r2)


import matplotlib.pyplot as plt

predicted_marks = predict(hours, w, b)
# Evaluate on the unseen test student
test_prediction = predict(x_test, w, b)
test_mae = np.mean(np.abs(y_test - test_prediction))
test_mean_error = np.mean(test_prediction - y_test)
test_r2 = r_squared(y_test, test_prediction)
print("Test R-squared:", test_r2)
print("Test Mean Error:", test_mean_error)

print("Test MAE:", test_mae)
train_prediction = predict(x_train, w, b)


train_mae = np.mean(np.abs(y_train - train_prediction))
print("Training MAE:", train_mae)

for i in range(len(y_test)):
    print(f"\nTest Student {i + 1}")
    print("Actual marks:", y_test[i])
    print("Predicted marks:", test_prediction[i])
    print("Absolute error:", abs(y_test[i] - test_prediction[i]))
    # Compare training and test MAE
plt.figure(figsize=(7, 5))

labels = ["Training MAE", "Test MAE"]
values = [train_mae, test_mae]

print("Actual test marks:", y_test)
print("Predicted test marks:", test_prediction)
print("Number of test students:", len(y_test))
bars = plt.bar(labels, values)
plt.bar_label(bars, fmt="%.3f", padding=3)

plt.title("Training vs Test Error")
plt.ylabel("Mean Absolute Error (marks)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))

students = np.arange(len(y_test))
width = 0.35

plt.bar(students - width / 2, y_test, width, label="Actual Marks")
plt.bar(students + width / 2, test_prediction, width, label="Predicted Marks")

plt.title("Actual vs Predicted Marks (Test Data)")
plt.xlabel("Test Student")
plt.ylabel("Marks")
plt.xticks(students, [f"Student {i + 1}" for i in students])
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show()
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
try:
    new_hours = float(input("Enter study hours: "))

    if new_hours < 0:
        print("Study hours cannot be negative.")
    else:
        predicted_new_marks = predict(new_hours, w, b)
        print(f"Predicted marks: {predicted_new_marks:.2f}")

except ValueError:
    print("Invalid input! Please enter a number.")