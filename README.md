# Machine-Learnig-for-Cybersecurity---Credit-Card-Fraud-Detection-

**Authors**

- Dylan Sansas 
- Maxime Thim-Siong
- Fabien Leboucher
- Mélissa Deffarges



----------





## Credit Card Detection - Project Overview

Credit card fraud detection is one of the most important applications of machine learning in finance today. Every year, billions are lost to fraudulent transactions. Therefore, the financial institutions need increasingly better solutions to fight this threat. However, building an effective fraud detection system is a hard task due to the imbalance in datasets. Fraudulent transactions are extremely rare compared to legitimates ones. In our dataset, only 0.6% of transactions are fraudulent, meaning that a model that predicts everything as legitimate would achieve 99.4% accuracy while being completely useless in reality. 

This project explores and compares four algorithms (Random Forest, LightGBM, XGBoost, and Gradient Boosting) for detecting fraudulent credit card transactions, evaluating their performance both with and without SMOTE.



----------




## Dataset

- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 1.3 millions transactions (training) + 555 thousands transactions (testing)
- **Features**: 73 (after the preprocessing)
- **Class Distribution**: 99.4% is legitimate and 0.6% is fraudulent 
- **Time Period**: January 2019 - June 2020

The dataset contains real credit card transactions with both legitimate and fraudulent cases. Each transaction includes information about the cardholder, merchant, transaction details, and location.

This data reflects the pratical challenges of fraud detection because, not only is fraud very rare, but it's also difficult to distinguish from legitimate outliers like travel or large purchase



----------




## Project Structure

```

TO DO 

```


----------



## Methodology 

### 1. Data Preprocessing

Before applying any machine learning model, we needed to prepare the raw data. To do this, we followed those steps: 

**Feature Engineering** : We created neew features from the raw data to capture patterns that might indicate fraud.

    • Temporal Features: We extracted the hour of the day, day of the week, and wether the transaction occured on a weekend. These are important because fraud often happens at unusual times. 

    • Geographic Features: We calculated the distance between the customer's registered address and the merchant's location. The idea is that stolen cards might be used far from where it was stolen.

    • Demographic Features: We calculted the age of the customer since it can be linked to fraud risk (young adults and seniors are more susceptible to fall for a scam then a middle-age person)

**Categorical Encoding**: The dataset contained many categorical variables with thousands of unique values. So we chose to use three different encoding strategies depending on the feature's cardinality:

    • Label Encoding for binary variables (like gender: M→1, F→0)

    • One-Hot Encoding for moderate cardinality features (51 states)

    • Target Encoding for high-cardinality features (thousands of merchants), where we replaced each category with the average fraud rate for that category


**Normalization**: We normalized numerical features using **StandardScaled**, which transforms each features to have mean 0 and standard deviation 1. This ensures no feature will dominate due to its scale.


### 2. Handling Class Imbalance

The 172:1 imbalance ratio is the central problem in this project. Without addressing it, a model could achieve 99.4% accuracy by simply predicting "legitimate" for everything. To counter this, we decided to use 2 approaches and compare them to determine which one is the best. 

**Approach A : Class Weights Without Synthetic Data**

This approach tells the model to "care more" about fraud cases. To do that, it penalizes the errors on minority class more heavily by modifying their weight. 

By doing this, we get faster models and we work with the original data. However, the model only see about 0.6% of fraud examples during training, which might not be enough to capture all the fraud patterns.

**Approach B : SMOTE (Synthetic Minority Over-sampling Technique)**

Instead of reweighting, SMOTE generates synthetic fraud examples. Here's how it works : 

    1. For each real transaction, SMOTE finds its k-nearest fraud neighbors
    
    2. It creates new synthetic frauds by interpolating between these neighbors

    3. We set "sampling_strategy = 0.3" to create a roughly 3:1 imbalance (not 1:1 because this would not be realistic)

This generates about 300 000 synthetic fraud examples, giving the model more fraud to learn from. However, this makes the model slower.


### 3. The Four Models 

- **Random Forest**: 


- **Gradient Boosting**:  An ensemble model based on weak (shallow) decision trees combined sequentially, where each new tree is trained to correct the errors of the previous ones by following the gradient descent of a loss function.


- **XGBoost**:


- **LightGBM**: Microsoft's alternative to XGBoost. It uses histograms instead of exact values and grows trees leaf-wise instead of level-wise. It is significantly faster than XGBoost but is more prone to overfitting.

<table style="border-collapse: collapse; width: 100%; max-width: 900px;">
  <caption style="caption-side: top; text-align: left; font-weight: 600; padding: 4px 0;">
    Modèles et temps d'entraînement
  </caption>
  <thead>
    <tr>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left;">Model</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left;">Algorithm</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left;">Key Strength</th>
      <th style="border: 1px solid #d0d7de; padding: 8px; text-align: left;">Training Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Random Forest</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Parallel</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Stability, interpretability</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">2–3 min</td>
    </tr>
    <tr>
      <td style="border: 1px solid #e6edf3; padding: 8px;">LightGBM</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Sequential</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Speed, memory efficiency</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">30–45 s</td>
    </tr>
    <tr>
      <td style="border: 1px solid #e6edf3; padding: 8px;">XGBoost</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Sequential</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Accuracy, robustness</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">3–5 min</td>
    </tr>
    <tr>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Gradient Boosting</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Sequential</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">Stability, simplicity</td>
      <td style="border: 1px solid #e6edf3; padding: 8px;">2–3 min</td>
    </tr>
  </tbody>
</table>



### 4. Evaluation Metrics

- **Precision**: Shows how many transactions that were flagged as fraudulent actually were.

- **Recall**: Shows the percentage of fraud the model caught.

- **F1-Score**: The mean of precision and recall. Important because it allows us to see if our model is ignore false positives or false negatives.

- **PR-AUC (Precision-Recall - Area Under Curve)**: Shows the trade-off between precision and recall.

- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Shows the trade-off between true positive rate and false positive rate.


----------


## Results 



### Model Performance Comparison


TABLEAUX + GRAPH 





----------




## Usage

To reproduce our results, run the **main.ipynb** notebook as it contains the complete pipeline: data loading, preprocessing, feature engienering, model training and evaluation.



----------




## Requirements 

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
jupyter>=1.0.0
```




----------





## References 

- Lucas, Y., & Jurgovsky, J. (2020). Credit card fraud detection using machine learning: A survey. arXiv preprint arXiv:2010.06479.

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-Generation Hyperparameter Optimization Framework. arXiv preprint arXiv:1907.10902

- Niu, X., Wang, L., & Yang, X. (2019). A Comparison Study of Credit Card Fraud Detection: Supervised versus Unsupervised. arXiv preprint arXiv:1904.10604

- Ghanem, M., Elkaffas, S. M., & Madbouly, M. Machine Learning Technique for Credit Card Fraud Detection

- Kochnev, R., Goodarzi, A. T., Bentyn, Z. A., Ignatov, D., & Timofte, R. (2025). Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning? arXiv preprint arXiv:2504.06006


















