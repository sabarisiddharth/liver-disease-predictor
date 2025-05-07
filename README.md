# liver-disease-predictor

# Liver Disease Prediction using Machine Learning

A machine learning project designed to predict the presence of liver disease in patients based on clinical data. This model aims to assist in early detection and potential intervention by healthcare professionals.

##  Problem Statement

Liver disease, if undiagnosed, can lead to severe health complications. Using patient records (age, enzyme levels, etc.), this project predicts whether a person is likely to have liver disease using classification models.

##  Dataset

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))
- Records: 583 patient records
- Features: 10 medical attributes + 1 target label

##  Technologies Used

- **Language**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, SMOTE
- **Environment**: Google Colab / Jupyter Notebook / VS code

##  Workflow

1. **Data Cleaning**: Handled missing values and outliers
2. **Exploratory Data Analysis (EDA)**: Visualized distributions, correlations
3. **Feature Engineering**: Created new ratio features, log-transformations for skewed values
4. **Balancing**: Used SMOTE for handling class imbalance
5. **Modeling**:
    - Random Forest Classifier (primary model)
    - Tuned using GridSearchCV
6. **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC

##  Results

- **Model Accuracy**: ~84.4% after tuning and feature enhancements
- **Best Performing Model**: Stacked Ensemble (RandomForest + GradientBoosting)

##  Sample Output

```python
Input: { 'Age': 52, 'Total_Bilirubin': 1.3, 'ALT': 45, ... }
Output: Predicted Class - "Liver Disease"
Confidence: 92%
