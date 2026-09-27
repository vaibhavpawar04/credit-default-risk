# Credit Card Default Risk: 7 Models Compared

Predicting which credit card clients will default next month, comparing five machine learning models against two deep neural networks, and connecting the results to how banks provision for loan losses.

## Results

Evaluated on a stratified test set of 6,000 clients.

| Model | Accuracy | AUC | Recall | Precision | F1 | KS |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.688 | 0.714 | 0.616 | 0.375 | 0.466 | 0.358 |
| Random Forest | 0.792 | 0.747 | 0.460 | 0.535 | 0.494 | 0.388 |
| **Gradient Boosting** | 0.792 | **0.768** | 0.494 | 0.532 | 0.512 | **0.403** |
| SVM | 0.782 | 0.753 | 0.546 | 0.506 | **0.525** | 0.397 |
| XGBoost | **0.806** | 0.751 | 0.399 | **0.590** | 0.476 | 0.378 |
| **DNN1** (MLP + Dropout) | 0.736 | **0.768** | **0.628** | 0.433 | 0.513 | 0.401 |
| DNN2 (Deep MLP + BatchNorm) | 0.707 | 0.717 | 0.565 | 0.389 | 0.461 | 0.333 |

## Dataset

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI Machine Learning Repository, also mirrored on Kaggle)

- 30,000 credit card clients in Taiwan, April to September 2005
- Credit limit, demographics, 6 months of repayment status, bill amounts and payments
- Target: default on next month's payment (22.1% default, 77.9% no default)

Save the file as `default_of_credit_card_clients.csv` in the same folder as the notebook.

## Approach

**1. Data cleaning**
- Standardised undocumented `EDUCATION` and `MARRIAGE` category codes
- Renamed `PAY_0` to `PAY_1` so the six repayment columns line up with the bill and payment columns
- Checked for missing values and duplicates (none), and dropped the ID column

**2. Exploratory analysis**
- Default rates by gender, education, marital status, age, credit limit and repayment status
- Correlation heatmap across all features

**3. Feature engineering**
- `UTILIZATION_RATIO` = latest bill ÷ credit limit (how much of the limit is used)
- `PAYMENT_RATIO` = latest payment ÷ latest bill (how much of the bill is actually paid)

**4. Modelling**
- Stratified 80 / 20 train/test split, with features scaled by `StandardScaler`
- **SMOTE applied to the training set only** (5,309 defaults balanced up to 18,691), so the test set keeps the real-world class balance
- Five ML models: Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost
- Two neural networks in TensorFlow/Keras, both with early stopping:
  - **DNN1:** 64 → 32 → 16 units with 30% dropout
  - **DNN2:** 128 → 64 → 32 → 16 units with batch normalisation

**5. Evaluation**
- Credit-scoring metrics beyond accuracy: AUC, KS statistic, recall, precision, F1, MCC, G-mean, balanced accuracy
- ROC curves, confusion matrices, DNN training curves and XGBoost feature importance

## Key findings

- **Recent repayment behaviour (`PAY_1` to `PAY_6`) is the strongest predictor of default** across the EDA, correlations and feature importance.
- **Gradient Boosting and DNN1 tie for the best ranking ability** (AUC 0.768). DNN1 also catches the most defaulters (recall 0.628).
- **XGBoost has the highest accuracy and precision but the lowest recall.** It is conservative: it rarely wrongly flags a good customer, but it misses 60% of defaulters.
- **More depth made the neural network worse.** DNN2 fit the training data better but generalised worse than the smaller DNN1, a typical result on small tabular datasets.
- **KS statistics of 0.33 to 0.40** fall within the range generally considered acceptable to good for credit scoring models.

## From model to balance sheet

A model's predicted probability of default is the **PD** in expected-loss provisioning under IFRS 9 (Expected Loss = PD × LGD × EAD). Choosing between these models is a business decision, not only a statistical one:

- **Favour recall** (DNN1, Logistic Regression) to catch more defaulters and reduce credit losses
- **Favour precision** (XGBoost) to avoid rejecting creditworthy customers, who generate interest income

The notebook also covers how card issuers earn money (net interest margin, interchange, fees) and why revolving customers are both the most profitable and the riskiest.

## Limitations and next steps

- An AUC of about 0.77 is close to the typical ceiling reported for this dataset, so bigger gains are more likely to come from features than from more complex models. For example, trends across all six months of bills and payments, where the current ratios use only the latest month.
- A cost-sensitive decision threshold, like the one in my card fraud detection project, would turn these probabilities into an approve/decline policy.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow / Keras · imbalanced-learn · SciPy · matplotlib · seaborn

## Run it

```bash
pip install pandas numpy scikit-learn xgboost tensorflow imbalanced-learn scipy matplotlib seaborn
jupyter notebook credit_default_risk.ipynb
```

## Context

Built for the Payments and Transaction Services module, MSc in FinTech, National College of Ireland.

**Author:** Vaibhav Pawar · [GitHub](https://github.com/vaibhavpawar04)
