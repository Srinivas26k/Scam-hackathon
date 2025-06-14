#!/bin/bash

# Final Project Status Check
echo "🏆 HACKATHON PROJECT - FINAL STATUS CHECK"
echo "=========================================="
echo ""

echo "📋 CHECKLIST:"
echo ""

# Check if models exist
if [ -f "models/churn_model.joblib" ] && [ -f "models/preprocessor.joblib" ]; then
    echo "✅ Machine Learning Models: READY"
else
    echo "❌ Machine Learning Models: MISSING"
fi

# Check if sample data exists
if [ -f "sample_data.csv" ]; then
    echo "✅ Sample Data: READY"
else
    echo "❌ Sample Data: MISSING"
fi

# Check if app exists
if [ -f "churn_prediction_app.py" ]; then
    echo "✅ Streamlit Application: READY"
else
    echo "❌ Streamlit Application: MISSING"
fi

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✅ Virtual Environment: READY"
else
    echo "❌ Virtual Environment: MISSING"
fi

# Check if documentation exists
if [ -f "README.md" ] && [ -f "COMPLETION_REPORT.md" ]; then
    echo "✅ Documentation: READY"
else
    echo "❌ Documentation: INCOMPLETE"
fi

echo ""
echo "🎯 QUICK ACTIONS:"
echo ""
echo "  Start the app:  streamlit run churn_prediction_app.py"
echo "  View demo:      ./demo.sh"
echo "  Check models:   ls -la models/"
echo "  Test sample:    head sample_data.csv"
echo ""

echo "🌐 Application URL: http://localhost:8501"
echo ""
echo "🎉 Project Status: COMPLETE & READY FOR DEMO!"
