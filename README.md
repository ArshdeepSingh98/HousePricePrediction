# HousePricePrediction

Attempt 1
"n_estimators": [50, 100, 200],
"max_features" : [1, 2, 4, 6]

Region derived feature shouldn't contain price, as it can cause data leakage

2. Data Preprocessing

   Handle missing values (e.g., imputation).
   Convert categorical features to numerical (one-hot encoding, label encoding).
   Normalize or standardize numerical features.
   Remove outliers.

3. Feature Engineering

   Create new relevant features (e.g., price per square foot).
   Perform dimensionality reduction if needed.

4. Data Splitting

   Split data into training, validation, and test sets (e.g., 70%-15%-15%).

5. Model Selection

   Choose models like Linear Regression, Decision Trees, Random Forest, Gradient Boosting (XGBoost, LightGBM), or Neural Networks.

6. Model Training

   Train models using the training data.
   Use hyperparameter tuning (Grid Search, Random Search, Bayesian Optimization).

7. Model Evaluation

   Evaluate models using metrics like RMSE, MAE, and R².

8. Model Deployment

   Save the trained model (pickle, joblib, or TensorFlow format).
   Deploy using Flask, FastAPI, or a cloud service (AWS, GCP, Azure).
