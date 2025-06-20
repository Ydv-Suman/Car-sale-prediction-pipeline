# 🚗 Car Sales Prediction Pipeline

A machine learning pipeline for predicting used car prices using features like brand, color, odometer, and doors, built with Scikit-learn and Random Forest.

## 📊 Project Overview

This project demonstrates an end-to-end machine learning workflow for predicting car sale prices. It includes data preprocessing, feature encoding, imputation, model training, and evaluation—all wrapped inside a Scikit-learn pipeline.

## 🔧 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook (for development)

## 🗂️ Features

- Handles missing data using `SimpleImputer`
- Encodes categorical features with `OneHotEncoder`
- Builds a clean and reusable pipeline using `Pipeline` and `ColumnTransformer`
- Uses `RandomForestRegressor` for training and prediction
- Evaluates model using `.score()` (R² score)

## 📁 Dataset Structure

The dataset includes the following columns:

| Column            | Description                           |
|-------------------|---------------------------------------|
| `Make`            | Brand of the car                      |
| `Color`           | Exterior color of the car             |
| `Odometer(Miles)` | Total miles driven                    |
| `Doors`           | Number of doors (e.g., 2, 4, 5)       |
| `Price`           | Target variable (car price in USD)    |
