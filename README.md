# Machine Learning Model for Predicting Breakdown Labels

This repository contains a Python-based machine learning model developed for predicting breakdown labels in a dataset related to vehicle maintenance or similar scenarios.

## Dataset

The model is trained and tested using two CSV files:

- `train_labeled.csv`: This file contains labeled data used for training the model.
- `prediction_unlabeled.csv`: This file contains unlabeled data on which the trained model makes predictions.

## Features

The model considers various features, including:

- Vehicle attributes (Maker, Model, Color, Body Type, Engine Size, Gearbox, Fuel Type, etc.).
- Maintenance-related data (Price, Seat Number, Repair Cost, Repair Hours, etc.).
- Breakdown and repair dates (used to calculate 'repair_time_days').

## Data Preprocessing

The code includes:

- **Handling missing values:** Removes rows with null values.
- **Handling duplicate rows:** Removes duplicate rows from the dataset.
- **Feature engineering:** Creates a new feature `repair_time_days` to reflect the time between a breakdown and its repair.
- **Feature encoding:** Encodes categorical variables (using Label Encoding).
- **Outlier detection and removal:** Detects and removes outliers from specified numerical features using the IQR method.

## Model Training and Evaluation

Several machine learning algorithms are used and evaluated:

- **Logistic Regression:** Used as a baseline model.
- **Random Forest:** Evaluated using accuracy, precision, recall, F1-score.
- **Decision Tree:** Also evaluated using similar metrics.
- **Support Vector Machine (SVM):** Used with a linear kernel for classification.

The code also performs hyperparameter tuning for the Random Forest model using GridSearchCV.

## Model Prediction

After training, the model is used to predict labels for the unlabeled data in `prediction_unlabeled.csv`. 

## Output

The predicted labels are stored in a new CSV file named `prediction_labeled.csv`.

## How to Run

1. Ensure that you have the necessary Python libraries installed (numpy, pandas, scikit-learn, seaborn, matplotlib).
2. Replace `/content/train_labeled.csv` and `/content/prediction_unlabeled.csv` with the correct paths to your data files.
3. Execute the code in a Jupyter notebook or Python environment.

## Contributing

Contributions to this project are welcome.

