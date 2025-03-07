# Credit Card Fraud Detection with Machine Learning

This repository contains a machine learning project focused on detecting credit card fraud using a `RandomForestClassifier`. The model is evaluated using Stratified K-fold cross-validation and is tuned for precision, recall, and F1-score.

## Project Overview

In this project, the following steps are performed:
1. **Dataset Loading**: The dataset is downloaded from Kaggle using the `kagglehub` library. The dataset has been collected and analysed during a research collaboration of   Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
2. **Data Preprocessing**: The dataset is preprocessed for training, including feature scaling and splitting into training and validation sets.
3. **Model Training**: A `RandomForestClassifier` is trained on the preprocessed data.
4. **Stratified K-fold Cross-Validation**: The model is evaluated using 5-fold cross-validation, ensuring that class distributions are preserved across folds.
5. **Model Evaluation**: The model is evaluated using precision, recall, and F1-score across each fold.

## Libraries and Dependencies

The following Python libraries are required to run this project:

- `kagglehub`
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
