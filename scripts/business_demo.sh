#!/bin/bash

echo "🏆 Enhanced Churn Prediction System - Business Intelligence Demo"
echo "============================================================="
echo ""

# Check if the app is running
if pgrep -f "streamlit run churn_prediction_app.py" > /dev/null; then
    echo "✅ Streamlit app is already running"
else
    echo "🚀 Starting Enhanced Churn Prediction System..."
    cd /home/nampallisrinivas26/Desktop/Hackathin/submission/dashboard
    nohup streamlit run churn_prediction_app.py --server.port 8501 > /dev/null 2>&1 &
    sleep 3
    echo "✅ App started successfully!"
fi

echo ""
echo "🌟 NEW BUSINESS INTELLIGENCE FEATURES:"
echo "-------------------------------------"
echo "💰 Advanced Business Impact Analysis"
echo "   • Potential Revenue Loss Calculation"
echo "   • ROI Predictions for Retention Campaigns"
echo "   • Customer Lifetime Value (CLV)"
echo "   • Net Retention Value Analysis"
echo ""
echo "📊 Consistent Analytics Across All Pages"
echo "   • Session state data persistence"
echo "   • Real-time business metrics updates"
echo "   • Unified data source tracking"
echo ""
echo "🔄 Continuous Learning & Feedback System"
echo "   • User feedback collection (5-point scales)"
echo "   • Prediction quality assessment"
echo "   • Automatic model improvement logging"
echo ""
echo "🎯 Enhanced Retention Recommendations"
echo "   • Contract-based personalized offers"
echo "   • Service addition recommendations"
echo "   • Payment method optimization"
echo "   • Loyalty program enrollment"
echo ""
echo "⏰ Intelligent Intervention Timing"
echo "   • Risk-based action urgency (24h to quarterly)"
echo "   • Tenure-adjusted intervention strategies"
echo "   • Specific action plan recommendations"
echo ""

echo "🔗 Access the enhanced system at: http://localhost:8501"
echo ""
echo "📝 DEMO FLOW:"
echo "1. Try Single Prediction → Enter customer details → See business impact"
echo "2. Try Batch Prediction → Upload assets/test_customers.csv → Review analytics"
echo "3. Check Analytics page → See business metrics and retention strategies"
echo "4. Provide feedback → Help the system improve continuously"
echo ""
echo "🎯 Key Competitive Advantages:"
echo "• Real business ROI calculations (not just predictions)"
echo "• Actionable, time-bound recommendations"
echo "• Continuous learning from user feedback"
echo "• Complete business intelligence solution"
echo ""
echo "✨ This system goes beyond churn prediction to provide"
echo "   comprehensive customer retention business strategy!"
