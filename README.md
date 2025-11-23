# Loan_approval_prediction

## Loan Approval Prediction using Machine Learning
This project builds an end‑to‑end machine learning pipeline to predict whether a loan application should be approved based on an applicant’s demographic, financial, and credit information. The goal is to support financial institutions in making faster and more consistent lending decisions by learning approval patterns from historical data.

## Problem statement
Financial institutions receive thousands of loan applications and must decide which applicants are likely to repay their loans. Traditional manual review is time‑consuming, subjective, and difficult to scale. This project uses historical loan data to learn patterns that distinguish approved from rejected applications and then applies those patterns to new applicants. The target variable is a binary flag (loan_status) indicating whether a loan was approved (1) or not approved (0).

## Dataset description
#### The project uses a structured dataset with 50,000 records and 10 features describing each applicant and their loan request. Key variables include:

age: Age of the applicant in years.

income: Annual income of the applicant.

years_experience: Total work experience in years.

credit_score: Numerical credit score summarizing the applicant’s credit history.

loan_amount: Requested loan amount.

employment_type: Employment category (e.g., Salaried, Self‑Employed).

city: City of residence.

education: Highest education level (e.g., High School, Graduate, PhD).

marital_status: Marital status of the applicant.

loan_status: Target label indicating loan approval (0/1).

This mix of numerical and categorical features makes the problem a good fit for tree‑based and ensemble models.

Exploratory data analysis and preprocessing
The notebook starts with data extraction and auditing:

Inspects dataset shape, data types, and sample rows.

Computes descriptive statistics for numerical features (mean, standard deviation, quartiles, min, max) to understand distributions and detect possible outliers.

Uses correlation matrices to study relationships between numerical variables like age, income, credit_score, and loan_amount.

For preprocessing and feature engineering, the workflow includes:

Selecting relevant predictors, with a focus on important numeric variables (age, income, credit_score, loan_amount) and converting categorical features such as employment_type into numerical form via encoding.

Ensuring the data is in a model‑ready format (no incompatible dtypes, consistent feature set for train and test).

Splitting the dataset into training and testing sets to evaluate model generalization.

Modeling and algorithms
Several supervised classification algorithms are implemented and compared:

### Decision Tree Classifier

Serves as a baseline interpretable model.

Hyperparameters like max_depth, min_samples_split, and min_samples_leaf are tuned to control overfitting and improve performance.

### Random Forest Classifier

Uses an ensemble of decision trees to reduce variance and improve robustness.

Important hyperparameters explored include n_estimators (number of trees), max_depth, min_samples_split, and min_samples_leaf.

### Gradient Boosting Classifier

Builds trees sequentially, where each new tree focuses on correcting the errors of the previous ones.

Tuning covers n_estimators, learning_rate, min_samples_split, and min_samples_leaf to balance bias and variance.

### XGBoost Classifier

A powerful gradient boosting implementation optimized for speed and performance.

Parameters such as max_depth, n_estimators, learning_rate, and colsample_bytree are adjusted to improve accuracy and prevent overfitting.

For each algorithm, the project starts with a default model and then refines it using more appropriate hyperparameters based on observed performance.

## Model evaluation
Model performance is evaluated using:

Train–test split: The data is divided into training and testing subsets so that model accuracy on unseen examples can be measured.

Accuracy score: Primary metric reported for all models to compare how many predictions are correct.

Classification report and confusion matrix: Provide deeper insight into class‑wise performance (true positives, false positives, etc.) and any imbalance between predicting approvals and rejections.

## A consolidated results table compares all models on train and test accuracy. In this project:

All tuned tree‑based and ensemble models reach strong accuracy on the test set, around 91%, indicating that the models capture meaningful patterns in applicant data without extreme overfitting.

Random Forest, Gradient Boosting, and tuned XGBoost achieve the best balance between training and test performance, making them strong candidates for deployment.

Project structure and technologies
The core implementation is contained in a Jupyter notebook, which walks through:

Importing libraries (Pandas, NumPy, Seaborn, Matplotlib, scikit‑learn, XGBoost).

### Loading and auditing the dataset.

Exploratory data analysis and feature selection.

Encoding categorical variables and preparing features/labels.

Training baseline models (Decision Tree, Random Forest, Gradient Boosting, XGBoost).

Hyperparameter tuning for improved generalization.

Evaluating and comparing models using accuracy, classification reports, and confusion matrices.

Summarizing results in a model comparison table.

