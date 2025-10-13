# Machine-Learnig-for-Cybersecurity---Credit-Card-Fraud-Detection-
Project for the class "Hands-on machine learning for cybersecurity". This project has for objective to create a machine learning model to detect credit card fraud and analyze the results.

Proposition 1 : Financial Fraud Detection with ensemble learning

=> problem statement : fraudulent financial transactions (ex: credit card fraud) 
are much more difficult to detect than legitimate transactions. So,
we have a dataset that will be imbalanced !

=> objective : explore how ensemble learning methods can be applied
(Random Forest, Gradient Boosting and XGBoost)

=> Methodology :

1) Dataset selection & preprocessing
- publicly available dataset => Kaggle Credit Card Fraud Detection
- parameters : transaction frequency, time-based features, merchant risk profiling ??

2) Handling class imbalance
- SMOT strategy
- cost-sensitive learning where misclassifying fraud carries a higher penalty

3) Modeling with ensemble learning
- Random forest (good to solve overfitting, random selection for features), Gradient boosting and XGBoost models
- compare performances between models

4) Evaluation & interpretation
- Evaluate using F1-score, Precision-recall AUC and ROC-AUC
- analyze false positives and false negatives
- SHAP values to understand model decisions
