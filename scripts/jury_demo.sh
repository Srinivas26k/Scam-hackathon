#!/bin/zsh

# 🏆 TEAM IDEAVAULTS - COMPETITION DEMO SCRIPT
# Customer Churn Prediction - Hackathon Submission
# Date: June 14, 2025

echo "🏆 TEAM IDEAVAULTS - HACKATHON DEMO"
echo "===================================="
echo ""
echo "👥 Team: Srinivas, Hasvitha, & Srija"
echo "🎯 Challenge: DS-2 (Stop the Churn)"
echo "📅 Date: June 14, 2025"
echo ""

echo "🔥 COMPETITION HIGHLIGHTS:"
echo "   🏅 AUC-ROC Score: 0.8499 (Excellent!)"
echo "   ⚡ Real-time predictions: <50ms"
echo "   📊 Advanced ML: Ensemble model"
echo "   🎨 Professional UI: Complete dashboard"
echo ""

echo "📊 JUDGING CRITERIA COVERAGE:"
echo "   ✅ Prediction Accuracy (40%): AUC-ROC 0.8499"
echo "   ✅ Innovation (30%): Ensemble + Feature Engineering"
echo "   ✅ Dashboard Usability (25%): All 4 required elements"
echo "   ✅ Code Quality (5%): Professional documentation"
echo ""

echo "🎯 REQUIRED DASHBOARD ELEMENTS:"
echo "   📊 Churn probability distribution plot"
echo "   🥧 Churn vs retain pie chart"
echo "   🚨 Top-10 highest risk customers table"
echo "   📥 Download predictions option"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    echo "✅ Virtual environment ready"
    source .venv/bin/activate
else
    echo "⚠️  Setting up virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r config/requirements.txt
fi

echo ""
echo "🔍 VERIFYING COMPETITION READINESS..."

# Check all required files
echo "📁 Checking project structure:"
echo -n "   Competition data: "
if [[ -f "data/gBqE3R1cmOb0qyAv.csv" ]] && [[ -f "data/YTUVhvZkiBpWyFea.csv" ]]; then
    echo "✅"
else
    echo "❌"
fi

echo -n "   Trained models: "
if [[ -f "models/churn_model.joblib" ]] && [[ -f "models/preprocessor.joblib" ]]; then
    echo "✅"
else
    echo "❌"
fi

echo -n "   Predictions ready: "
if [[ -f "assets/predictions.csv" ]]; then
    echo "✅"
else
    echo "❌"
fi

echo -n "   Dashboard app: "
if [[ -f "dashboard/churn_prediction_app.py" ]]; then
    echo "✅"
else
    echo "❌"
fi

echo ""
echo "🚀 STARTING COMPETITION DASHBOARD..."
echo ""
echo "🌐 Dashboard will open at: http://localhost:8501"
echo ""
echo "📋 DEMO FLOW FOR JURY:"
echo "   1. Home - View project overview & metrics"
echo "   2. Batch Prediction - Upload CSV & see all 4 required elements:"
echo "      📊 Probability distribution"
echo "      🥧 Churn vs retain pie chart"
echo "      🚨 Top-10 risk table"
echo "      📥 Download predictions"
echo "   3. Single Prediction - Test individual customer"
echo "   4. Analytics - Business insights"
echo "   5. Model Insights - Technical details"
echo ""

echo "💡 JURY TESTING INSTRUCTIONS:"
echo "   • Upload 'data/YTUVhvZkiBpWyFea.csv' for batch prediction"
echo "   • Try individual customer prediction"
echo "   • Download results to verify CSV format"
echo "   • Check AUC-ROC score: 0.8499"
echo ""

echo "🏆 COMPETITIVE ADVANTAGES:"
echo "   🎯 Superior AUC-ROC performance (0.8499)"
echo "   🤖 Real ensemble ML model (not mock)"
echo "   🎨 Professional dashboard with all requirements"
echo "   📈 Advanced feature engineering (26 features)"
echo "   ⚡ Production-ready performance"
echo ""

echo "⏱️  Starting application in 3 seconds..."
sleep 3

echo ""
echo "🚀 LAUNCHING DASHBOARD..."

# Kill any existing streamlit processes
pkill -f streamlit 2>/dev/null

# Start the application
streamlit run dashboard/churn_prediction_app.py --server.port 8501 --server.address localhost
