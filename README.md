# Diabetes Feature Engineering Project

This project focuses on exploratory data analysis (EDA) and feature engineering on the diabetes dataset.

## Project Content

- General overview of the dataset
- Identification of categorical and numerical variables
- Target variable analysis
- Outlier analysis
- Missing value analysis
- Correlation analysis
- Treating zero values as missing values in selected variables
- Filling missing values with median
- Outlier capping
- Creating new features
- One-Hot Encoding
- Scaling with StandardScaler
- A simple Streamlit application for feature engineering

## Created Features

The following new features were created in this project:

- `NEW_AGE_CAT`
- `NEW_BMI_CAT`
- `NEW_GLUCOSE_CAT`
- `NEW_AGE_BMI`
- `NEW_AGE_GLUCOSE`
- `NEW_BMI_GLUCOSE`

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Streamlit
- Missingno

## Files

- `diabetes_feature_engineering.py` → Main Python file containing EDA and feature engineering steps
- `app.py` → Streamlit app that displays created features based on user input

## How to Run the App

Run the following commands in the terminal:

```bash
pip install -r requirements.txt
streamlit run app.py

## Project Purpose 

The aim of this project is to apply feature engineering techniques on the diabetes dataset and to
develop a simple application that dynamically displays the created features based on user input.
