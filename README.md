# 🏆 Team Ideavaults - Customer Churn Prediction
## DS-2 Hackathon Submission

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)

### 👥 Team Information
- **Team Name:** Ideavaults  
- **Members:** Srinivas, Hasvitha, & Srija
- **Competition:** DS-2 (Stop the Churn)
- **Date:** June 14, 2025

### 🎯 Project Overview
Professional customer churn prediction system with **AUC-ROC: 0.8499** performance, featuring ensemble machine learning and interactive dashboard with all competition requirements.

---

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

### **Competition Demo Flow:**
1. Open: http://localhost:8501
2. Navigate to "📈 Batch Prediction"
3. Upload: `data/YTUVhvZkiBpWyFea.csv` 
4. View all required elements:
   - 📊 Churn probability distribution
   - 🥧 Churn vs retain pie chart
   - 🚨 Top-10 risk customers table
   - 📥 Download predictions

---

## 🏅 Competition Requirements

### ✅ **Core Must-Haves (100% Complete)**
- **Upload & Parse**: CSV handling with validation
- **Processing Engine**: AUC-ROC optimized (0.8499)
- **Output/UX**: All 4 required dashboard elements

### 🎯 **Judging Criteria Coverage**
- **Prediction Accuracy (40%)**: 🏆 AUC-ROC 0.8499
- **Innovation (30%)**: 🚀 Ensemble ML + Advanced Features  
- **Dashboard Usability (25%)**: 🎨 Professional Streamlit UI
- **Code Quality (5%)**: 📚 Clean, documented structure

---

## 🔧 Technical Stack

- **ML Framework**: scikit-learn, LightGBM
- **Frontend**: Streamlit, HTML/CSS/JS
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib
- **Model Persistence**: joblib

---

## 📊 Performance Metrics

| Metric | Score | Status |
|--------|-------|--------|
| **AUC-ROC** | **0.8499** | 🏅 Excellent |
| **Accuracy** | 77.8% | ✅ Strong |
| **F1-Score** | 0.599 | ✅ Balanced |
| **Speed** | <50ms | ⚡ Fast |

---

## 🏆 Competitive Advantages

1. **🎯 Superior Performance**: AUC-ROC 0.8499 vs basic models
2. **🤖 Real ML Pipeline**: Ensemble learning, not mock
3. **🎨 Professional UI**: Complete dashboard with all requirements
4. **⚡ Production Ready**: Error handling, validation, scaling
5. **📚 Excellent Documentation**: Clear setup and usage

---

## 📞 Support

**Team Ideavaults**
- **Demo Issues**: Run `./scripts/jury_demo.sh`
- **Manual Start**: `python main.py` 
- **Documentation**: See `docs/` folder

---

**🚀 Ready to win the hackathon! 🏆**
