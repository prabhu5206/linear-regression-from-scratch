# Linear Regression from Scratch

A simple machine learning project that implements **linear regression using NumPy**, without relying on machine learning libraries such as scikit-learn.

## Overview

This project predicts student marks based on study hours. It demonstrates how a linear regression model learns its parameters using gradient descent and evaluates predictions on unseen test data.

## Features

- Linear regression implemented from scratch
- Prediction using a learned weight and bias
- Mean Squared Error (MSE) loss function
- Gradient descent optimization
- Training loss visualization
- Train/test data split
- Model evaluation using MAE, MSE, and R²
- Actual vs. predicted marks visualization
- Training vs. test error comparison
- Interactive predictions based on study hours

## Technologies Used

- Python
- NumPy
- Matplotlib

## How It Works

1. Load sample data containing study hours and student marks.
2. Split the data into training and test sets.
3. Train the model using gradient descent.
4. Evaluate predictions using regression metrics.
5. Visualize the results using Matplotlib.
6. Enter study hours to generate an interactive prediction.

## Dataset

The project uses a small illustrative dataset of 10 student examples. The data is intended for learning and experimentation, not for making real-world predictions about academic performance.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/prabhu5206/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

Create and activate a virtual environment:

```bash
python3
