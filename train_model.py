# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
import joblib
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Liverd.csv")

# Basic preprocessing
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Dataset'] = df['Dataset'].map({1: 0, 2: 1})
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)

# Derived features
df['Bilirubin_Ratio'] = df['Direct_Bilirubin'] / (df['Total_Bilirubin'] + 1e-5)
df['Enzyme_Sum'] = df['Alamine_Aminotransferase'] + df['Aspartate_Aminotransferase']
df['Protein_Gap'] = df['Total_Protiens'] - df['Albumin']

# Log transform
df['Total_Bilirubin'] = np.log1p(df['Total_Bilirubin'])
df['Alkaline_Phosphotase'] = np.log1p(df['Alkaline_Phosphotase'])
df['Alamine_Aminotransferase'] = np.log1p(df['Alamine_Aminotransferase'])
df['Aspartate_Aminotransferase'] = np.log1p(df['Aspartate_Aminotransferase'])

# Split features and labels
X = df.drop(columns=["Dataset"])
y = df["Dataset"]

# Apply SMOTE
print("Before SMOTE:", Counter(y))
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:", Counter(y_res))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Define models
tuned_rf = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42)
tuned_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
meta = make_pipeline(StandardScaler(), LogisticRegression())

# Stack
stacked = StackingClassifier(
    estimators=[('rf', tuned_rf), ('gb', tuned_gb)],
    final_estimator=meta,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

# Fit and evaluate
stacked.fit(X_train, y_train)
y_pred = stacked.predict(X_test)

print(f"\nFinal Ensemble Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(stacked, X_res, y_res, cv=StratifiedKFold(5), scoring='accuracy')
print(f"\nCV Accuracy Mean: {cv_scores.mean():.4f}")
print(f"CV Std Deviation: {cv_scores.std():.4f}")

# Save model
joblib.dump(stacked, "model.pkl")
print("\n🎉 Model saved successfully as 'model.pkl'")
