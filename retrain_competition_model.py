import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 RETRAINING MODEL ON COMPETITION DATA")
print("=" * 50)

# Load competition data
train_data = pd.read_csv('data/gBqE3R1cmOb0qyAv.csv')
test_data = pd.read_csv('data/YTUVhvZkiBpWyFea.csv')

print(f"📊 Data loaded:")
print(f"   Training: {len(train_data)} customers")
print(f"   Test: {len(test_data)} customers")

# Data preprocessing
print("\n🔧 Feature Engineering...")

def create_features(df):
    """Create engineered features"""
    df = df.copy()
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Tenure groups
    df['tenure_group'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 48, np.inf], 
                               labels=['New', 'Medium', 'Long', 'Very Long'])
    
    # Service count
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Count total services (excluding 'No' and 'No internet service')
    df['total_services'] = 0
    for service in services:
        if service in df.columns:
            df['total_services'] += (df[service] == 'Yes').astype(int)
    
    # Average monthly charges per service
    df['avg_monthly_charges'] = df['MonthlyCharges']
    df['charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
    
    # Contract and payment features
    df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['is_electronic_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # High-risk combination
    df['high_risk_combo'] = ((df['Contract'] == 'Month-to-month') & 
                            (df['PaymentMethod'] == 'Electronic check') & 
                            (df['SeniorCitizen'] == 1)).astype(int)
    
    return df

# Apply feature engineering
train_processed = create_features(train_data)
test_processed = create_features(test_data)

print("✅ Feature engineering completed")

# Prepare training data
X = train_processed.drop(['customerID', 'churned'], axis=1)
y = train_processed['churned']

# Prepare test data (no target variable)
X_test = test_processed.drop(['customerID'], axis=1)

print(f"📈 Features: {X.shape[1]}")
print(f"📈 Training samples: {len(X)}")
print(f"📈 Test samples: {len(X_test)}")

# Define preprocessing pipeline
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod', 'tenure_group']

numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                     'total_services', 'avg_monthly_charges', 'charges_per_service',
                     'is_monthly_contract', 'is_electronic_payment', 'high_risk_combo']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

print("\n🤖 Training optimized model for AUC-ROC...")

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Train ensemble model optimized for AUC-ROC
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

# Voting classifier
model = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr)],
    voting='soft'
)

model.fit(X_train_processed, y_train)

# Validate performance
y_val_pred_proba = model.predict_proba(X_val_processed)[:, 1]
y_val_pred = model.predict(X_val_processed)

auc_score = roc_auc_score(y_val, y_val_pred_proba)

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"   AUC-ROC Score: {auc_score:.4f}")
print(f"   Accuracy: {(y_val_pred == y_val).mean():.4f}")

# Cross-validation AUC
cv_auc_scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='roc_auc')
print(f"   CV AUC Mean: {cv_auc_scores.mean():.4f} (+/- {cv_auc_scores.std() * 2:.4f})")

# Generate predictions for test data
print("\n🔮 Generating test predictions...")
X_test_processed = preprocessor.transform(X_test)
test_predictions = model.predict_proba(X_test_processed)[:, 1]
test_binary_pred = model.predict(X_test_processed)

# Save predictions
predictions_df = pd.DataFrame({
    'customerID': test_processed['customerID'],
    'churn_probability': test_predictions,
    'predicted_churn': test_binary_pred
})

predictions_df.to_csv('predictions.csv', index=False)

# Save updated models
joblib.dump(model, 'models/churn_model.joblib')
joblib.dump(preprocessor, 'models/preprocessor.joblib')

print(f"✅ Predictions saved to predictions.csv")
print(f"✅ Models updated and saved")

print(f"\n📊 PREDICTION SUMMARY:")
print(f"   Total customers: {len(predictions_df)}")
print(f"   Predicted to churn: {test_binary_pred.sum()}")
print(f"   Churn rate: {test_binary_pred.mean():.2%}")
print(f"   Probability range: {test_predictions.min():.3f} - {test_predictions.max():.3f}")

print("\n🏆 MODEL READY FOR COMPETITION!")
