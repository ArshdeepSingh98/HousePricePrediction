# HousePricePrediction

## Real Estate Price Prediction

## 1. Summary

This project aims to develop a machine learning model that accurately predicts real estate prices based on various features such as location, size, and amenities. By analyzing historical data, the model provides insights into property valuation, assisting stakeholders in making informed decisions.

## 2. Installation Steps

To set up the project locally, follow these steps:

1. Unzip the git submission files:
2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

4. Place the Dataset:

## 3. File Folder Explanation

- data: Dataset is present in this folder, which has 3 subfolders namely

  - Raw: It contains raw intial train and test csv files
  - Processed: It contains cleaned final csv to be used for training and building models
  - Interim: It contains intermediate csv files between pipeline steps

- notebooks: All Jupyter notebooks that are used in this project are present in this

  - 01_data_ingestion.ipynb: 1st Step, containing the ingestion code
  - 02_EDA.ipynb: 2nd Step, containing the Exploratory Data Analysis code
  - 03_feature_engineering.ipynb: 3rd Step, containing the feature engineering and data preprocessing code
  - 04_model_training_and_inference.ipynb: Last step, containing the model training, hyper-parameter tuning, and inference code
  - Complete_all_steps_one_file.ipynb: All in one steps notebook

- pdf: This folder contains the pdf of all steps in one file
- reports: This folder contains the model's evaluation scores, as well as a summary file
- model: This folder contains the saved final model for this project

<!--
4.  Data Preprocessing

    Handle missing values (e.g., imputation).
    Convert categorical features to numerical (one-hot encoding, label encoding).
    Normalize or standardize numerical features.
    Remove outliers.

5.  Feature Engineering

    Create new relevant features (e.g., price per square foot).
    Perform dimensionality reduction if needed.

6.  Data Splitting

    Split data into training, validation, and test sets (e.g., 70%-15%-15%).

7.  Model Selection

    Choose models like Linear Regression, Decision Trees, Random Forest, Gradient Boosting (XGBoost, LightGBM), or Neural Networks.

8.  Model Training

    Train models using the training data.
    Use hyperparameter tuning (Grid Search, Random Search, Bayesian Optimization).

9.  Model Evaluation

    Evaluate models using metrics like RMSE, MAE, and R².

10. Model Deployment

    Save the trained model (pickle, joblib, or TensorFlow format).
    Deploy using Flask, FastAPI, or a cloud service (AWS, GCP, Azure). -->
