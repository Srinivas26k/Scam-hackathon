#!/bin/bash
# Demo Script for Team Ideavaults Churn Prediction System

echo "🔮 Team Ideavaults - Customer Churn Prediction Demo"
echo "=================================================="
echo ""
echo "👥 Team Members: Srinivas, Hasvitha, & Srija"
echo "🎯 Hackathon Submission: Advanced Customer Churn Prediction"
echo ""

echo "📊 Model Performance:"
echo "- Accuracy: 79.0%"
echo "- F1 Score: 0.599"
echo "- AUC Score: 0.720"
echo ""

echo "🚀 Starting Streamlit Application..."
echo "Access the app at: http://localhost:8501"
echo ""

echo "📝 Demo Instructions:"
echo "1. 🏠 Home - View project overview and metrics"
echo "2. 👤 Single Prediction - Test individual customer predictions"
echo "3. 📈 Batch Prediction - Upload sample_customers.csv for bulk analysis"
echo "4. 📊 Analytics - Explore customer insights and patterns"
echo "5. 🔍 Model Insights - Deep dive into model performance"
echo ""

echo "📁 Sample Files Available:"
echo "- sample_customers.csv (for batch prediction demo)"
echo "- Real trained model with ensemble approach"
echo "- 52 engineered features from 20 original features"
echo ""

echo "🎨 Tech Stack:"
echo "- ML: LightGBM, XGBoost, Random Forest ensemble"
echo "- Frontend: Streamlit + Interactive HTML/CSS/JS"
echo "- Features: SMOTE, Feature Engineering, Hyperparameter Tuning"
echo ""

# Check if streamlit is running
if pgrep -f "streamlit" > /dev/null; then
    echo "✅ Streamlit is already running!"
else
    echo "🚀 Starting Streamlit..."
    uv run streamlit run churn_prediction_app.py
fi
