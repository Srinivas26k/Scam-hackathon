
"""
Complete Customer Churn Prediction Model Training Pipeline
Team Ideavaults: Sri, Na, and Ka
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.feature_names = None

    def load_data(self, filepath):
        """Load and initial preprocessing of data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data shape: {df.shape}")
        return df

    def preprocess_data(self, df):
        """Advanced data preprocessing and feature engineering"""
        print("Preprocessing data...")

        # Handle TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

        # Feature Engineering
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, float('inf')], 
                                   labels=['New', 'Medium', 'Long', 'Very Long'])

        # Service count
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['total_services'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

        # Financial features
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)

        # Contract features
        df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['is_electronic_payment'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

        # High-risk combinations
        df['high_risk_combo'] = ((df['Contract'] == 'Month-to-month') & 
                                (df['PaymentMethod'] == 'Electronic check')).astype(int)

        # Separate features and target
        if 'churned' in df.columns:
            X = df.drop(['churned', 'customerID'], axis=1)
            y = df['churned']
            return X, y
        else:
            X = df.drop(['customerID'], axis=1)
            return X

    def create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        print("Creating preprocessor...")

        # Identify column types
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Remove tenure_group from categorical if it exists (it's already object type)
        categorical_features = [col for col in categorical_features if col != 'tenure_group']
        if 'tenure_group' in X.columns:
            categorical_features.append('tenure_group')

        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")

        from sklearn.preprocessing import OneHotEncoder

        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ]
        )

        return preprocessor

    def train_model(self, X, y):
        """Train ensemble model with advanced techniques"""
        print("Training model...")

        # Create preprocessor
        self.preprocessor = self.create_preprocessor(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Apply SMOTE for class balance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

        print(f"Original training set: {X_train_processed.shape}")
        print(f"Balanced training set: {X_train_balanced.shape}")

        # Create ensemble model
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            random_state=42,
            verbose=-1
        )

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
        )

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        # Ensemble model
        self.model = VotingClassifier([
            ('lgbm', lgbm),
            ('xgb', xgb),
            ('rf', rf)
        ], voting='soft')

        # Train model
        self.model.fit(X_train_balanced, y_train_balanced)

        # Evaluate model
        train_pred = self.model.predict(X_train_processed)
        test_pred = self.model.predict(X_test_processed)

        train_f1 = f1_score(y_train, train_pred)
        test_f1 = f1_score(y_test, test_pred)

        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Testing F1 Score: {test_f1:.4f}")

        # Detailed evaluation
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

        return X_test, y_test, test_pred

    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained yet!")

        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]

        return predictions, probabilities

    def save_model(self, model_path='churn_model.joblib', preprocessor_path='preprocessor.joblib'):
        """Save trained model and preprocessor"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")

    def load_model(self, model_path='churn_model.joblib', preprocessor_path='preprocessor.joblib'):
        """Load trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("Model and preprocessor loaded successfully!")

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor()

    # Load and preprocess data
    train_data = predictor.load_data('gBqE3R1cmOb0qyAv.csv')
    X, y = predictor.preprocess_data(train_data)

    # Train model
    X_test, y_test, predictions = predictor.train_model(X, y)

    # Save model
    predictor.save_model()

    print("\n🎉 Model training completed successfully!")
    print("Ready for deployment!")
