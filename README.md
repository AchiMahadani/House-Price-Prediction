# House Price Prediction

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [How to Run the Project](#how-to-run-the-project)
- [References](#references)

## Introduction
House prices are influenced by various factors such as location, size, number of rooms, and more. Accurately predicting house prices is crucial for real estate agents, buyers, and sellers. This project aims to build a machine learning model that can predict house prices based on features such as the number of bedrooms, size of the house, and other property details.

## Problem Statement
The challenge is to develop a machine learning model that accurately predicts house prices given the property's features. The model will be trained on a dataset containing various house attributes and their corresponding prices.

## Objective
The objective of this project is to:
- Build a predictive model that can estimate the price of a house based on its features.
- Evaluate the model's performance using relevant metrics such as mean squared error (MSE) and R² score.

## Dataset
The dataset contains the following features:
- **Lot Area**: The size of the property in square feet.
- **Year Built**: The year the house was built.
- **Total Rooms**: The total number of rooms.
- **Number of Bedrooms**: The number of bedrooms in the house.
- **Number of Bathrooms**: The number of bathrooms.
- **Garage Area**: The size of the garage.
- **Neighborhood**: The neighborhood where the house is located.
- **Sale Price**: The target variable, i.e., the price the house was sold for.

## Methodology

### Data Preprocessing
- Handling missing values.
- Encoding categorical variables (e.g., neighborhood).
- Scaling numerical features (e.g., lot area, garage area).
- Splitting the data into training and test sets.

### Exploratory Data Analysis
- Distribution of house prices.
- Correlation analysis between features and house prices.
- Insights derived from relationships between features such as house size, location, and price.

### Feature Engineering
- Creating new features such as **Age of the house** (current year - year built).
- Handling skewed data by applying transformations if necessary.
- Feature selection to improve model performance.

### Model Selection
The models considered for this project include:
- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

### Model Training and Evaluation
- Train models on the training dataset.
- Evaluate model performance on the test set using:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
- Perform hyperparameter tuning using techniques like GridSearchCV or RandomSearchCV.

## Results
- **Best Performing Model**: Random Forest Regressor achieved an RMSE of **X** and an R² score of **Y**.
- **Model Performance**:
  - Linear Regression: RMSE = **X**, R² = **Y**
  - Random Forest: RMSE = **X**, R² = **Y**
  - Gradient Boosting: RMSE = **X**, R² = **Y**

## Conclusion
This project successfully predicted house prices using machine learning models. The Random Forest Regressor model outperformed others in terms of RMSE and R² score. However, there is still room for improvement by experimenting with additional features and advanced models.

## Future Work
- Explore deep learning models such as neural networks to further improve accuracy.
- Incorporate more features like real-time market trends, proximity to amenities, etc.
- Build a web interface to allow users to input house features and get price predictions.

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/house-price-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd house-price-prediction
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the model training script:
    ```bash
    python train_model.py
    ```

5. Predict house prices using the trained model:
    ```bash
    python predict.py --input house_features.csv
    ```

## References
- [Kaggle House Price Prediction Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

