# 🔮 Customer Churn Prediction - Team Ideavaults

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 👥 Team Information
**Team Name:** Ideavaults  
**Members:** Srinivas, Hasvitha , & Srija
**Competition:** Advanced Customer Churn Prediction Challenge

## 🎯 Project Overview

This project delivers a comprehensive, production-ready customer churn prediction system that combines advanced machine learning techniques with an intuitive user interface. Our solution processes customer data to predict churn probability with 79.0% accuracy, helping businesses proactively retain valuable customers.

## ✨ Key Features

### 🤖 Advanced ML Pipeline
- **Ensemble Learning**: LightGBM, XGBoost, and Random Forest combination
- **Feature Engineering**: 52 sophisticated features from 20 original attributes
- **Class Balance Handling**: SMOTE and class weighting techniques
- **Hyperparameter Optimization**: Optuna-powered automatic tuning
- **Cross-Validation**: 5-fold stratified validation for robust performance

### 🎨 Interactive Dashboard
- **Single Customer Prediction**: Real-time churn risk assessment
- **Batch Processing**: Upload CSV files for bulk predictions
- **Analytics Dashboard**: Comprehensive business insights
- **Model Explainability**: SHAP values and feature importance
- **Risk Categorization**: Low, Medium, and High risk classifications

### 📊 Performance Metrics
- **F1 Score**: 0.599
- **AUC Score**: 0.720
- **Accuracy**: 79.0%
- **Prediction Speed**: < 100ms per customer

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/churn-prediction-ideavaults.git
   cd churn-prediction-ideavaults
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run churn_prediction_app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Start predicting customer churn!

## 📁 Project Structure

```
churn-prediction-ideavaults/
├── churn_prediction_app.py      # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── train_data.csv         # Training dataset
│   └── test_data.csv          # Test dataset
├── models/
│   ├── churn_model.joblib     # Trained model
│   └── preprocessor.joblib    # Data preprocessor
├── notebooks/
│   ├── data_exploration.ipynb # EDA and analysis
│   ├── model_training.ipynb   # Model development
│   └── feature_engineering.ipynb # Feature creation
└── src/
    ├── data_preprocessing.py   # Data processing utilities
    ├── feature_engineering.py # Feature creation functions
    ├── model_training.py      # Model training pipeline
    └── utils.py               # Helper functions
```

## 🛠️ Technical Implementation

### Data Preprocessing
- **Missing Value Handling**: Advanced imputation techniques
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Data Validation**: Comprehensive input validation and error handling

### Feature Engineering
Our advanced feature engineering creates 52 features from the original 20:

1. **Demographic Features**: Age groups, family status combinations
2. **Service Utilization**: Total services count, service combinations
3. **Financial Behavior**: Charges per service, tenure-based metrics
4. **Customer Lifecycle**: Tenure groupings, contract risk factors
5. **Interaction Features**: Cross-feature combinations

### Model Architecture
```python
# Primary ensemble approach
ensemble = VotingClassifier([
    ('lgbm', LGBMClassifier(n_estimators=200, max_depth=7)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=6)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8))
])
```

### Deployment Pipeline
1. **Data Validation**: Input validation and preprocessing
2. **Model Loading**: Efficient model and preprocessor loading
3. **Prediction Generation**: Fast inference with probability scores
4. **Result Formatting**: User-friendly output with risk categorization

## 📈 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| F1 Score | 0.599 | Balanced precision-recall measure |
| AUC Score | 0.720 | Area under ROC curve |
| Accuracy | 79.0% | Overall prediction accuracy |
| Precision | 0.61 | Positive prediction accuracy |
| Recall | 0.59 | True positive detection rate |

### Feature Importance (Top 10)
1. **tenure** (25.0%) - Customer retention period
2. **MonthlyCharges** (18.0%) - Monthly billing amount
3. **TotalCharges** (15.0%) - Lifetime customer value
4. **Contract_Month-to-month** (12.0%) - Contract flexibility risk
5. **PaymentMethod_Electronic check** (8.0%) - Payment method risk
6. **InternetService_Fiber optic** (7.0%) - Service type impact
7. **avg_monthly_charges** (5.0%) - Engineered financial metric
8. **total_services** (4.0%) - Service utilization count
9. **charges_per_service** (3.0%) - Value per service ratio
10. **is_monthly_contract** (3.0%) - Contract type binary flag

## 🎨 User Interface Features

### 🏠 Home Dashboard
- **Performance Metrics**: Real-time model statistics
- **Quick Stats**: Customer base overview
- **Feature Highlights**: System capabilities showcase

### 👤 Single Customer Prediction
- **Interactive Form**: Easy-to-use customer data input
- **Real-time Prediction**: Instant churn probability calculation
- **Risk Visualization**: Color-coded risk assessment
- **Probability Gauge**: Visual probability display

### 📈 Batch Prediction
- **CSV Upload**: Drag-and-drop file upload
- **Progress Tracking**: Real-time processing status
- **Results Download**: Exportable prediction results
- **Batch Analytics**: Aggregate insights and visualizations

### 📊 Analytics Dashboard
- **Customer Segmentation**: Demographic and behavioral analysis
- **Churn Patterns**: Trend identification and visualization
- **Financial Insights**: Revenue impact analysis
- **Interactive Charts**: Plotly-powered visualizations

### 🔍 Model Insights
- **Feature Importance**: Detailed feature contribution analysis
- **Model Architecture**: Technical implementation details
- **Performance Metrics**: Comprehensive evaluation results
- **Confusion Matrix**: Prediction accuracy visualization

## 🔧 Advanced Features

### Model Explainability
- **SHAP Integration**: Individual prediction explanations
- **Feature Importance**: Global model interpretability
- **Decision Trees**: Visual decision pathways
- **Local Explanations**: Customer-specific insights

### Data Validation
- **Input Validation**: Comprehensive data type and range checks
- **Error Handling**: Graceful error management and user feedback
- **Data Consistency**: Cross-field validation and logical checks
- **Missing Data**: Intelligent imputation strategies

### Performance Optimization
- **Model Caching**: Efficient model loading and caching
- **Batch Processing**: Optimized bulk prediction handling
- **Memory Management**: Efficient data processing
- **Response Time**: < 100ms prediction latency

## 📱 Deployment Options

### Local Deployment
```bash
streamlit run churn_prediction_app.py
```

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload all project files
3. Set runtime to Streamlit
4. Your app will be live at: `https://huggingface.co/spaces/username/space-name`

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "churn_prediction_app.py"]
```

## 📊 Business Impact

### Cost Savings
- **Customer Retention**: Proactive churn prevention
- **Revenue Protection**: Identify high-value at-risk customers
- **Resource Optimization**: Targeted retention campaigns

### Operational Efficiency
- **Automated Scoring**: Real-time churn risk assessment
- **Batch Processing**: Efficient large-scale analysis
- **Dashboard Insights**: Data-driven decision making

### Strategic Benefits
- **Predictive Analytics**: Forward-looking customer insights
- **Risk Management**: Early warning system for customer attrition
- **Competitive Advantage**: Advanced ML-powered customer intelligence

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **Competition Organizers**: For providing the challenging dataset
- **Open Source Community**: For the amazing ML libraries
- **Team Ideavaults**: Srinivas, Hasvitha , & Srija for their dedication and expertise

## 📞 Contact

**Team Ideavaults**
- **Email**: team.ideavaults@example.com
- **GitHub**: [Team Ideavaults](https://github.com/team-ideavaults)
- **LinkedIn**: [Connect with us](https://linkedin.com/company/ideavaults)

---

<div align="center">
  <p><strong>🚀 Built with ❤️ by Team Ideavaults</strong></p>
  <p>Empowering businesses with AI-driven customer insights</p>
</div>
