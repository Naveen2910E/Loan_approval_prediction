# Loan_approval_prediction

# This repository contains a Loan Approval Prediction project that explores how applicant profile and credit information can be used to automatically predict loan approval using machine learning. The notebook covers the full workflow from data loading and exploratory analysis through feature preprocessing, training multiple tree-based and ensemble models (Decision Tree, Random Forest, Gradient Boosting, XGBoost), hyperparameter tuning, and performance comparison, achieving about 91% accuracy on unseen data
# Project overview
The notebook starts by loading a structured loan dataset with 50,000 records and 10 features such as age, income, years of experience, credit score, loan amount, employment type, city, education, marital status, and a binary loan_status target. It then performs initial data auditing using shape, head, describe, and correlation checks to understand distributions and relationships among variables.​

# Data preprocessing
The workflow includes selecting relevant numeric features (age, income, credit_score, loan_amount) and encoding categorical variables like employment_type into numerical form for modeling. Basic exploratory statistics and correlation matrices are used to assess feature behavior and multicollinearity before model training.​

# Modeling approach
Multiple classification algorithms are implemented: Decision Tree, Random Forest, Gradient Boosting, and XGBoost. For each model, an initial baseline is trained, followed by hyperparameter tuning (e.g., max_depth, min_samples_split, n_estimators, learning_rate, colsample_bytree) to improve generalization.​

# Evaluation and results
Models are evaluated using train–test split and standard classification metrics such as accuracy, along with classification reports and confusion matrices from scikit-learn. A comparison table summarizes train and test accuracy for all models; tuned ensembles like Random Forest and Gradient Boosting achieve around 91% accuracy on the test set, showing strong performance without severe overfitting
