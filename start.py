#!/usr/bin/env python3
"""
Startup script for AI Medical Disease Detection System
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Package names and their import names (some packages have different import names)
    package_imports = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'tensorflow': 'tensorflow',
        'opencv-python': 'cv2',     # opencv-python imports as cv2
        'pillow': 'PIL',            # pillow imports as PIL
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'python-multipart': 'multipart',
        'pydantic': 'pydantic',
        'joblib': 'joblib',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'imbalanced-learn': 'imblearn',  # imbalanced-learn imports as imblearn
        'nibabel': 'nibabel',
        'pydicom': 'pydicom',
        'scipy': 'scipy',
        'scikit-image': 'skimage'   # scikit-image imports as skimage
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_data_files():
    """Check if sample data files exist"""
    print("\n📁 Checking data files...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    required_files = [
        "diabetes_data.csv",
        "cardiovascular_data.csv", 
        "cancer_data.csv",
        "kidney_disease_data.csv",
        "liver_disease_data.csv",
        "combined_medical_data.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file}")
    
    if missing_files:
        print(f"\n⚠️ Missing data files: {', '.join(missing_files)}")
        print("Generate them using: python data/generate_sample_data.py")
        return False
    
    print("✅ All data files are present!")
    return True

def check_models():
    """Check if trained models exist"""
    print("\n🤖 Checking trained models...")
    
    models_dir = Path("models/saved_models")
    if not models_dir.exists():
        print("❌ Trained models not found")
        print("Train models using: python train_models.py")
        return False
    
    # Check for each disease model
    diseases = ['diabetes', 'cardiovascular', 'cancer', 'kidney_disease', 'liver_disease']
    
    for disease in diseases:
        disease_dir = models_dir / disease
        if disease_dir.exists():
            print(f"✅ {disease} models")
        else:
            print(f"❌ {disease} models")
    
    return True

def generate_sample_data():
    """Generate sample data if not present"""
    print("\n📊 Generating sample data...")
    
    try:
        result = subprocess.run([
            sys.executable, "data/generate_sample_data.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample data generated successfully!")
            return True
        else:
            print(f"❌ Error generating data: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running data generation: {e}")
        return False

def train_models():
    """Train models if not present"""
    print("\n🚀 Training models...")
    
    try:
        result = subprocess.run([
            sys.executable, "train_models.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Models trained successfully!")
            return True
        else:
            print(f"❌ Error training models: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running model training: {e}")
        return False

def start_streamlit_app():
    """Start the Streamlit web application"""
    print("\n🌐 Starting Streamlit web application...")
    print("📱 Open your browser and go to: http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def start_api_server():
    """Start the FastAPI server"""
    print("\n🔌 Starting FastAPI server...")
    print("📡 API available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def run_demo():
    """Run the interactive demo"""
    print("\n🎭 Starting interactive demo...")
    
    try:
        subprocess.run([
            sys.executable, "demo.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="AI Medical Disease Detection System")
    parser.add_argument("--mode", choices=["web", "api", "demo", "check"], 
                       default="check", help="Startup mode")
    parser.add_argument("--auto-setup", action="store_true", 
                       help="Automatically generate data and train models if missing")
    
    args = parser.parse_args()
    
    print("🏥 AI Medical Disease Detection System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check data files
    if not check_data_files():
        if args.auto_setup:
            print("\n🔄 Auto-setup: Generating sample data...")
            if not generate_sample_data():
                print("❌ Failed to generate sample data")
                return
        else:
            print("\n❌ Please generate sample data first")
            return
    
    # Check models
    if not check_models():
        if args.auto_setup:
            print("\n🔄 Auto-setup: Training models...")
            if not train_models():
                print("❌ Failed to train models")
                return
        else:
            print("\n❌ Please train models first")
            return
    
    print("\n✅ System is ready!")
    
    # Start based on mode
    if args.mode == "web":
        start_streamlit_app()
    elif args.mode == "api":
        start_api_server()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "check":
        print("\n💡 System is ready to use!")
        print("\nTo start the system:")
        print("  • Web Interface: python start.py --mode web")
        print("  • API Server: python start.py --mode api")
        print("  • Interactive Demo: python start.py --mode demo")
        print("  • Auto-setup: python start.py --auto-setup")

if __name__ == "__main__":
    main()
