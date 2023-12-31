# Credit Score Classification Model

Please use this link to view the full interactive notebook code and output: [Credit-Score-Classification](https://nbviewer.org/github/kamalavi/Credit-Score-Classification/blob/main/credit_score_classification.ipynb) 

This repository contains code and resources for a Credit Score Classification Model. The model is designed to predict the creditworthiness of individuals based on various features and attributes.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Data Cleaning](#data-cleaning)
- [Data Visualization](#data-visualization)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
  
## Introduction

A credit score classification model is an essential tool for financial institutions to evaluate the creditworthiness of individuals seeking loans or credit products. By utilizing machine learning techniques, this model provides a predictive solution to categorize applicants into different credit risk categories, such as low-risk, medium-risk, and high-risk.

The model is built using state-of-the-art machine learning algorithms, including Decision Trees, K-Nearest Neighbors (KNN), and Random Forest. The combination of these models allows for robust and accurate predictions. The repository provides all the necessary code and resources to clean the data, visualize its distribution, and train the credit score classification model using various algorithms.

## Data

The model is trained on a labeled dataset containing historical credit application data. The dataset comprises both input features (e.g., income, age, occupation, etc.) and the corresponding credit risk labels. However, raw data may often contain missing values, null entries, and outliers that need to be addressed before modeling.

## Data Cleaning

Data cleaning is an essential step in preparing the dataset for modeling. The repository contains a data cleaning script (`data_cleaning.py`) that performs the following tasks:

- Handles missing data: Any missing values in the dataset are appropriately imputed using suitable techniques to avoid bias in the model.
- Handles outliers: Outliers, if present, are identified and either removed or transformed to prevent them from affecting model performance.
- Drops variables out of scope: If certain variables are irrelevant or redundant for the credit score classification task, they are dropped from the dataset.

## Data Visualization

Data visualization is a crucial step to gain insights into the dataset and understand the distribution of variables. The repository provides visualization scripts (`data_visualization.py`) that create various plots, such as histograms, scatter plots, and box plots, to visualize the distribution of each variable. These visualizations can help identify skewed data, potential correlations, and patterns that may influence the credit score classification model's performance.

## Modeling

The credit score classification model utilizes three different machine learning algorithms to provide diverse perspectives on the data:

1. Decision Trees: A decision tree algorithm is used to construct a tree-like model of decisions and their possible consequences. It helps in understanding the hierarchical relationships among features and their importance in predicting credit risk.

2. K-Nearest Neighbors (KNN): The KNN algorithm is a simple yet effective method for classification. It classifies new data points based on the majority class of their K nearest neighbors. KNN can reveal patterns and local variations in the data.

3. Random Forest: The Random Forest algorithm builds multiple decision trees and combines their outputs to improve prediction accuracy and reduce overfitting. It can handle a large number of features and maintain robustness against noise.

## Evaluation

The model's performance can be assessed using various evaluation metrics, such as accuracy, precision, recall, and F1 score. Additionally, you can perform cross-validation to validate the model's generalization on unseen data. The repository provides evaluation scripts that can be used to measure the model's performance for each algorithm (Decision Trees, KNN, and Random Forest).
