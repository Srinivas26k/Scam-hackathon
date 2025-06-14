#!/usr/bin/env python3
"""
🏆 Team Ideavaults - Customer Churn Prediction
Main Entry Point for Hackathon Submission

This is the primary entry point for the competition submission.
Launches the Streamlit dashboard for jury evaluation.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point for the application."""
    
    print("🏆 TEAM IDEAVAULTS - CUSTOMER CHURN PREDICTION")
    print("=" * 50)
    print("👥 Team: Srinivas, Hasvitha, & Srija")
    print("🎯 Competition: DS-2 (Stop the Churn)")
    print("📅 Date: June 14, 2025")
    print()
    
    # Get the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🔍 Verifying project structure...")
    
    # Check required directories
    required_dirs = ['dashboard', 'models', 'data', 'config']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
        else:
            print(f"   ✅ {dir_name}/")
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    # Check required files
    required_files = [
        'dashboard/churn_prediction_app.py',
        'models/churn_model.joblib',
        'models/preprocessor.joblib',
        'data/YTUVhvZkiBpWyFea.csv',
        'config/requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"   ❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print()
    print("🚀 Launching Streamlit Dashboard...")
    print("📱 URL: http://localhost:8501")
    print()
    print("🎯 JURY DEMO INSTRUCTIONS:")
    print("   1. Navigate to '📈 Batch Prediction' tab")
    print("   2. Upload 'data/YTUVhvZkiBpWyFea.csv'")
    print("   3. Click 'Generate Predictions'")
    print("   4. View all 4 required dashboard elements")
    print("   5. Verify AUC-ROC score: 0.8499")
    print()
    
    # Launch Streamlit
    try:
        # Try different ports to avoid conflicts
        for port in [8501, 8503, 8504, 8502]:
            try:
                print(f"🚀 Trying to launch on port {port}...")
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run", 
                    "dashboard/churn_prediction_app.py",
                    "--server.port", str(port),
                    "--server.address", "localhost"
                ], check=True, timeout=5)
                break
            except subprocess.TimeoutExpired:
                print(f"✅ Streamlit launched successfully on port {port}")
                break
            except subprocess.CalledProcessError:
                print(f"   Port {port} unavailable, trying next...")
                continue
        else:
            print("❌ All ports busy. Please stop other Streamlit instances.")
            return False
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
        
    except FileNotFoundError:
        print("❌ Python or Streamlit not found")
        print("💡 Make sure you're in the virtual environment:")
        print("   source .venv/bin/activate")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
