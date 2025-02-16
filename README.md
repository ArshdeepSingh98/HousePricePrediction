# HousePricePrediction

## Real Estate Price Prediction

## Summary

This project aims to develop a machine learning model that accurately predicts real estate prices based on various features such as location, size, and amenities. By analyzing historical data, the model provides insights into property valuation, assisting stakeholders in making informed decisions.

## Installation Steps

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

## File Folder Explanation

- `data/`: Dataset is present in this folder, which has 3 subfolders namely

  - `data/raw`: It contains raw intial train and test csv files
  - `data/processed`: It contains cleaned final train and test (both scaled and unscaled) csv to be used for training and building models
  - `data/interim`: It contains intermediate csv files between pipeline steps

- `notebooks/`: All Jupyter notebooks that are used in this project are present in this

  - 01_data_ingestion.ipynb: 1st Step, containing the ingestion code
  - 02_EDA.ipynb: 2nd Step, containing the Exploratory Data Analysis code
  - 03_feature_engineering.ipynb: 3rd Step, containing the feature engineering and data preprocessing code
  - 04_model_training_and_inference.ipynb: Last step, containing the model training, hyper-parameter tuning, and inference code
  - Complete_all_steps_one_file.ipynb: All in one steps notebook

- `pdf/`: This folder contains the pdf of all steps in one file
- `reports/`: This folder contains the model's evaluation scores, as well as a summary file
- `model/`: This folder contains the saved final model for this project
- `requirements.txt`: List of dependencies

## How to Run Pipeline

### Step1: Run feature engineering

```bash
python scripts/run_preprocessing.py
```

### Step2:Train the model

```bash
python scripts/run_training.py
```

### Step3:Evaluate the model

```bash
python scripts/run_evaluation.py
```

## Pipeline Parts

The project pipeline consists of the following parts:

1. Data Ingestion (01_data_ingestion.ipynb):

   - Load the raw dataset.

2. Exploratory Data Analysis (EDA) (02_EDA.ipynb):

   - Handle missing values and data types.
   - Analyze data distributions and relationships.
   - Visualize key features and their correlations.
   - Identify patterns and outliers.
   - Save the intermediate data both scaled and unscaled to data/interim/.

3. Feature Engineering (03_feature_engineering.ipynb):

   - Create new features from existing data.
   - Encode categorical variables.
   - Scale numerical features.
   - Save the cleaned and transformed data to data/processed/ for modeling.

4. Model Training (04_model_training.ipynb):

   - Train machine learning models (e.g., Linear Regression, Random Forest, XGBoost).
   - Evaluate model performance using metrics like MAE, MSE, and R².
   - Tune the hyperparameters using Cross Validation for best results
   - Save the best-performing model to models/.

## Results

The Random Forest model demonstrated superior performance in predicting the prices:

- Mean Absolute Error (MAE): $115.20
- Mean Squared Error (MSE): 57,591.04
- Root Mean Squared Error (RMSE): $239.98
- R² Score: 0.8252

These metrics indicate that Random Forest model predicted property prices more accurately as compared to other models.

## Submission details

- Arshdeep Singh
- asroideva@gmail.com
