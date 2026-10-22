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

## Project Structure

- `main.py` — Loads data, runs training and evaluation, generates plots, and accepts user input.
- `model.py` — Contains the linear regression prediction, gradient descent, and evaluation functions.
- `student_data.csv` — Stores the study-hours and marks dataset.
- `requirements.txt` — Lists the Python dependencies.
- `.gitignore` — Excludes virtual environment and unnecessary files from Git.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/prabhu5206/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

Keep `student_data.csv` in the same directory as `main.py`.

## Limitations

The dataset contains only 10 illustrative examples, with 7 used for training and 3 for testing. Evaluation results are therefore not reliable evidence of real-world predictive performance.

## Future Improvements

- Support larger datasets.
- Improve input validation and error handling.
- Add automated tests for model functions.
- Build an interactive interface for predictions.
