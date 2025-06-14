# 🏆 Team Ideavaults - Advanced Customer Retention Intelligence
## DS-2 Hackathon Submission - Business Intelligence Enhanced

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)

## 👥 Team Information
**Team Name:** Ideavaults  
**Members:** Srinivas, Hasvitha , & Srija
**Competition:** Advanced Customer Churn Prediction Challenge

## 🎯 Project Overview

This project delivers a comprehensive, production-ready **Customer Retention Business Intelligence Platform** that goes beyond simple churn prediction. Our solution combines advanced machine learning with actionable business insights, ROI calculations, and continuous learning capabilities to help businesses proactively retain valuable customers with measurable financial impact.

## 🌟 NEW: Business Intelligence Features

### 💰 Advanced Financial Analytics
- **ROI Calculations**: Real return on investment for retention campaigns
- **Revenue Impact**: Potential loss and retention opportunity analysis
- **Customer Lifetime Value**: Comprehensive CLV calculations
- **Cost-Benefit Analysis**: Intervention costs vs acquisition costs
- **Net Retention Value**: Overall business impact assessment

### 🔄 Continuous Learning System
- **User Feedback Collection**: 5-point rating scales for prediction accuracy
- **Quality Assessment**: Automatic prediction confidence tracking
- **Model Improvement**: Self-learning from user interactions
- **Performance Monitoring**: Real-time system effectiveness metrics

### 🎯 Intelligent Retention Strategies
- **Personalized Recommendations**: Contract, service, and payment optimizations
- **Intervention Timing**: Risk-based action urgency (24h to quarterly)
- **Loyalty Programs**: Tenure-based VIP and rewards enrollment
- **Cost Optimization**: Price sensitivity and package reviews

## ✨ Core Technical Features

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

4. **Access the Enhanced Business Intelligence Platform**
   - Open your browser and navigate to `http://localhost:8501`
   - Experience advanced churn prediction with business intelligence!

## 🚀 Quick Demo of Enhanced Features

### Test the Business Intelligence Features:
```bash
# Run the enhanced demo
./scripts/business_demo.sh

# Test with sample data
# 1. Go to "📈 Batch Prediction" page
# 2. Upload: assets/test_customers.csv
# 3. See business impact analysis & ROI calculations
# 4. Provide feedback for continuous learning
```

### Key Pages to Explore:
- **🏠 Home**: Overview of business impact and quick stats
- **👤 Single Prediction**: Individual customer analysis with recommendations
- **📈 Batch Prediction**: Bulk processing with business metrics
- **📊 Analytics**: Business intelligence dashboard with financial insights
- **🔍 Model Insights**: Technical performance and feature importance

## 📁 Project Structure

```
ideavaults-churn-prediction/
├── 📊 dashboard/              # Interactive web applications
│   ├── churn_prediction_app.py   # Main Streamlit dashboard
│   ├── index.html                # Alternative web interface
│   ├── style.css                 # UI styling
│   └── app.js                    # Interactive features
├── 🤖 models/                 # Trained ML models
│   ├── churn_model.joblib        # Ensemble classifier
│   └── preprocessor.joblib       # Data preprocessor
├── 📈 data/                   # Competition datasets
│   ├── gBqE3R1cmOb0qyAv.csv     # Training data (5634 customers)
│   └── YTUVhvZkiBpWyFea.csv     # Test data (1409 customers)
├── 🧮 src/                    # Source code modules
│   ├── data_preprocessing.py     # Data cleaning utilities
│   ├── feature_engineering.py   # Feature creation functions
│   └── utils.py                  # Helper functions
├── 📚 docs/                   # Documentation
│   ├── README.md                 # This file
│   ├── COMPETITION_VERIFICATION.md  # Requirements checklist
│   ├── FINAL_SUMMARY.md          # Submission summary
│   └── QUICK_START.md            # Quick setup guide
├── 🔧 scripts/               # Automation scripts
│   ├── jury_demo.sh              # Competition demo script
│   ├── model_training.py         # ML training pipeline
│   └── retrain_competition_model.py  # Competition retraining
├── ⚙️ config/                # Configuration files
│   ├── requirements.txt          # Python dependencies
│   └── pyproject.toml           # Project configuration
├── 📦 assets/                 # Sample data and outputs
│   ├── predictions.csv           # Competition predictions
│   ├── sample_data.csv           # Demo dataset
│   └── sample_customers.csv      # Test samples
├── 🧪 tests/                  # Test files (for future)
├── 📓 notebooks/              # Jupyter analysis (for future)
└── main.py                    # Main entry point
```

---

## 🚀 Quick Start

### **For Jury Evaluation:**
```bash
# One-click demo
./scripts/jury_demo.sh

# Manual start
python main.py
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
