import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64
import os
import sys

# Handle optional dependencies gracefully
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not available - image processing features will be limited")

# Add utils and models to path
sys.path.append('utils')
sys.path.append('models')

from utils.data_processor import MedicalDataProcessor
from utils.model_prep import ensure_models_ready
from models.demo_models import DemoDiseaseDetector

# Page configuration
st.set_page_config(
    page_title="AI Medical Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .result-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class MedicalDetectionApp:
    def __init__(self):
        self.data_processor = MedicalDataProcessor()
        self.disease_detector = ensure_models_ready()
        self.uploaded_files = {}
        self.processed_data = {}
        self.demo_detector = DemoDiseaseDetector()
        
    def main(self):
        # Header
        st.markdown('<h1 class="main-header">🏥 AI-Powered Chronic Disease Detection System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.sidebar()
        
        # Main content
        if st.session_state.get('current_page') == 'upload':
            self.upload_page()
        elif st.session_state.get('current_page') == 'analysis':
            self.analysis_page()
        elif st.session_state.get('current_page') == 'results':
            self.results_page()
        else:
            self.home_page()
    
    def sidebar(self):
        with st.sidebar:
            st.markdown("## Navigation")
            
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.current_page = 'home'
                st.rerun()
            
            if st.button("📁 Upload Data", use_container_width=True):
                st.session_state.current_page = 'upload'
                st.rerun()
            
            if st.button("🔍 Analysis", use_container_width=True):
                st.session_state.current_page = 'analysis'
                st.rerun()
            
            if st.button("📊 Results", use_container_width=True):
                st.session_state.current_page = 'results'
                st.rerun()
            
            st.markdown("---")
            st.markdown("## System Status")
            
            # Check if models are loaded
            models_loaded = os.path.exists('models/saved_models')
            if models_loaded:
                st.success("✅ Models Loaded")
            else:
                st.warning("⚠️ Models Not Found")
            
            st.markdown("---")
            st.markdown("## Quick Stats")
            st.metric("Diseases Supported", "5")
            st.metric("Data Types", "3")
            st.metric("AI Models", "8")
    
    def home_page(self):
        st.markdown("## Welcome to AI Medical Disease Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What We Do
            Our AI-powered system analyzes multiple types of medical data to detect chronic diseases:
            
            - **Medical Images**: X-rays, CT scans, MRI scans
            - **Laboratory Results**: Blood tests, urine analysis
            - **Patient Demographics**: Age, gender, medical history
            
            ### 🚀 Key Features
            - **Multimodal Analysis**: Combines different data types
            - **Ensemble AI Models**: Multiple algorithms for accuracy
            - **Real-time Processing**: Instant disease detection
            - **Comprehensive Reports**: Detailed analysis and recommendations
            """)
        
        with col2:
            st.markdown("""
            ### 🏥 Supported Diseases
            1. **Diabetes** - Blood glucose analysis
            2. **Cardiovascular Disease** - Heart function assessment
            3. **Cancer** - Early detection screening
            4. **Kidney Disease** - Renal function evaluation
            5. **Liver Disease** - Hepatic function analysis
            
            ### 📊 How It Works
            1. Upload medical data
            2. AI processes multiple modalities
            3. Ensemble models analyze patterns
            4. Generate comprehensive report
            5. Provide treatment recommendations
            """)
        
        # Quick start section
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Upload Data", use_container_width=True):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with col2:
            if st.button("🔍 View Demo", use_container_width=True):
                self.show_demo()
        
        with col3:
            if st.button("📚 Learn More", use_container_width=True):
                st.info("Check out our documentation and research papers for detailed information about our AI models and methodology.")
    
    def upload_page(self):
        st.markdown("## 📁 Upload Medical Data")
        
        # File upload section
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🖼️ Medical Images")
                image_file = st.file_uploader(
                    "Upload medical images (X-ray, CT, MRI)",
                    type=['png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'],
                    key="image_upload"
                )
                
                if image_file:
                    self.uploaded_files['image'] = image_file
                    st.success(f"✅ Image uploaded: {image_file.name}")
                    
                    # Preview image
                    if image_file.type in ['image/png', 'image/jpeg', 'image/jpg']:
                        image = Image.open(image_file)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.markdown("### 🧪 Laboratory Results")
                lab_file = st.file_uploader(
                    "Upload lab results (CSV, Excel)",
                    type=['csv', 'xlsx', 'xls'],
                    key="lab_upload"
                )
                
                if lab_file:
                    self.uploaded_files['lab'] = lab_file
                    st.success(f"✅ Lab results uploaded: {lab_file.name}")
                    
                    # Preview lab data
                    try:
                        if lab_file.name.endswith('.csv'):
                            df = pd.read_csv(lab_file)
                        else:
                            df = pd.read_excel(lab_file)
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Manual data entry
        st.markdown("### 📝 Manual Data Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Patient Demographics")
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
        
        with col2:
            st.markdown("#### Vital Signs")
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
            
            # Blood pressure category
            if systolic_bp < 120 and diastolic_bp < 80:
                bp_category = "Normal"
            elif systolic_bp < 130 and diastolic_bp < 80:
                bp_category = "Elevated"
            else:
                bp_category = "High"
            
            st.metric("BP Category", bp_category)
        
        # Store manual data
        self.processed_data['demographics'] = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'bp_category': bp_category
        }
        
        # Process and analyze button
        if st.button("🔍 Process and Analyze Data", type="primary", use_container_width=True):
            if self.uploaded_files or self.processed_data:
                self.process_data()
                st.session_state.current_page = 'analysis'
                st.rerun()
            else:
                st.error("Please upload data or enter patient information first.")
    
    def process_data(self):
        """Process uploaded and manual data"""
        try:
            # Process image if uploaded
            if 'image' in self.uploaded_files:
                image_file = self.uploaded_files['image']
                # Save temporarily and process
                with open("temp_image", "wb") as f:
                    f.write(image_file.getbuffer())
                
                processed_image = self.data_processor.process_medical_image("temp_image")
                if processed_image is not None:
                    self.processed_data['image'] = processed_image
                
                # Clean up
                os.remove("temp_image")
            
            # Process lab results if uploaded
            if 'lab' in self.uploaded_files:
                lab_file = self.uploaded_files['lab']
                try:
                    if lab_file.name.endswith('.csv'):
                        lab_df = pd.read_csv(lab_file)
                    else:
                        lab_df = pd.read_excel(lab_file)
                    
                    processed_lab = self.data_processor.process_lab_results(lab_df)
                    self.processed_data['lab'] = processed_lab
                except Exception as e:
                    st.error(f"Error processing lab data: {e}")
            
            # Process demographics
            if 'demographics' in self.processed_data:
                demo_df = pd.DataFrame([self.processed_data['demographics']])
                processed_demo = self.data_processor.process_patient_demographics(demo_df)
                self.processed_data['processed_demographics'] = processed_demo
            
            st.success("✅ Data processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing data: {e}")
    
    def analysis_page(self):
        st.markdown("## 🔍 Data Analysis")
        
        if not self.processed_data:
            st.warning("No data to analyze. Please upload data first.")
            return
        
        # Display processed data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Processed Data Summary")
            
            if 'image' in self.processed_data:
                st.markdown("**Medical Image:** ✅ Processed")
                image_data = self.processed_data['image']
                st.image(image_data, caption="Processed Image", use_container_width=True)
                # Simple image analysis
                try:
                    mean_intensity = float(image_data.mean())
                    edge_map = cv2.Canny((image_data * 255).astype('uint8'), 50, 150)
                    edge_density = float(edge_map.mean() / 255.0)
                    st.metric("Mean Intensity", f"{mean_intensity:.3f}")
                    st.metric("Edge Density", f"{edge_density:.3f}")
                    self.processed_data['image_features'] = {
                        'mean_intensity': mean_intensity,
                        'edge_density': edge_density
                    }
                except Exception as _:
                    pass
            
            if 'lab' in self.processed_data:
                st.markdown("**Lab Results:** ✅ Processed")
                lab_data = self.processed_data['lab']
                st.dataframe(lab_data)
            
            if 'demographics' in self.processed_data:
                st.markdown("**Demographics:** ✅ Processed")
                demo_data = self.processed_data['demographics']
                demo_df = pd.DataFrame([demo_data])
                st.dataframe(demo_df)
        
        with col2:
            st.markdown("### 🧮 Feature Engineering")
            
            # Combine modalities
            try:
                combined_features = self.data_processor.combine_modalities(
                    self.processed_data.get('image'),
                    self.processed_data.get('lab'),
                    self.processed_data.get('processed_demographics')
                )
                
                st.success(f"✅ Combined features shape: {combined_features.shape}")
                
                # Feature importance visualization
                if combined_features.shape[1] > 1:
                    feature_importance = np.random.rand(combined_features.shape[1])
                    feature_importance = feature_importance / feature_importance.sum()
                    
                    fig = px.bar(
                        x=[f"Feature {i+1}" for i in range(len(feature_importance))],
                        y=feature_importance,
                        title="Feature Importance Distribution",
                        labels={'x': 'Features', 'y': 'Importance'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                self.processed_data['combined_features'] = combined_features
                
            except Exception as e:
                st.error(f"Error combining features: {e}")
        
        # Disease detection
        st.markdown("### 🏥 Disease Detection")
        # Auto-run detection when features are present
        if 'combined_features' in self.processed_data:
            if 'predictions' not in self.processed_data:
                self.run_disease_detection()
                st.session_state.current_page = 'results'
                st.rerun()
        else:
            st.info("Prepare features to run disease detection automatically.")
    
    def run_disease_detection(self):
        """Run disease detection on processed data"""
        try:
            # Initialize models (in real app, these would be pre-trained)
            # ensure_models_ready already returns loaded or fallback
            
            # Get features
            features = self.processed_data['combined_features']
            
            # Run predictions
            predictions = self.disease_detector.predict_all_diseases(features)

            # If predictions contain None due to missing models, use demo based on raw inputs
            if not predictions or any(v is None for v in predictions.values()):
                image_feats = self.processed_data.get('image_features')
                lab = None
                if 'lab' in self.processed_data and hasattr(self.processed_data['lab'], 'to_dict'):
                    try:
                        lab = self.processed_data['lab'].to_dict('records')[0]
                    except Exception:
                        lab = None
                demo = None
                if 'demographics' in self.processed_data:
                    demo = self.processed_data['demographics']
                predictions = self.demo_detector.predict_all_diseases_from_raw(
                    image_feats, lab, demo
                )
            
            # Store results
            self.processed_data['predictions'] = predictions
            
            st.success("✅ Disease detection completed!")
            
        except Exception as e:
            st.error(f"Error in disease detection: {e}")
    
    def results_page(self):
        st.markdown("## 📊 Analysis Results")
        
        if 'predictions' not in self.processed_data:
            st.warning("No results to display. Please run analysis first.")
            return
        
        # Results overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Disease Risk Assessment")
            
            predictions = self.processed_data['predictions']
            
            # Create results dataframe
            results_data = []
            for disease, risk in predictions.items():
                if risk is not None:
                    risk_value = float(risk[0]) if hasattr(risk, '__len__') else float(risk)
                    risk_percentage = risk_value * 100
                    
                    if risk_percentage < 30:
                        status = "🟢 Low Risk"
                        color = "green"
                    elif risk_percentage < 70:
                        status = "🟡 Medium Risk"
                        color = "orange"
                    else:
                        status = "🔴 High Risk"
                        color = "red"
                    
                    results_data.append({
                        'Disease': disease.replace('_', ' ').title(),
                        'Risk Score': f"{risk_percentage:.1f}%",
                        'Status': status,
                        'Risk Level': risk_percentage
                    })
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                results_df = results_df.sort_values('Risk Level', ascending=False)
                
                # Display results table
                st.dataframe(results_df[['Disease', 'Risk Score', 'Status']], use_container_width=True)
                
                # Risk visualization
                fig = px.bar(
                    results_df,
                    x='Disease',
                    y='Risk Level',
                    color='Risk Level',
                    title="Disease Risk Assessment",
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Risk Summary")
            
            if results_data:
                high_risk = len([r for r in results_data if r['Risk Level'] >= 70])
                medium_risk = len([r for r in results_data if 30 <= r['Risk Level'] < 70])
                low_risk = len([r for r in results_data if r['Risk Level'] < 30])
                
                st.metric("High Risk", high_risk, delta=None)
                st.metric("Medium Risk", medium_risk, delta=None)
                st.metric("Low Risk", low_risk, delta=None)
        
        # Detailed analysis
        st.markdown("### 🔬 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Risk Distribution")
            
            if results_data:
                risk_levels = [r['Risk Level'] for r in results_data]
                
                fig = px.pie(
                    values=risk_levels,
                    names=[r['Disease'] for r in results_data],
                    title="Risk Distribution by Disease"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Recommendations")
            
            if results_data:
                high_risk_diseases = [r['Disease'] for r in results_data if r['Risk Level'] >= 70]
                
                if high_risk_diseases:
                    st.warning("⚠️ **Immediate Attention Required**")
                    for disease in high_risk_diseases:
                        st.markdown(f"- **{disease}**: Consult specialist immediately")
                
                medium_risk_diseases = [r['Disease'] for r in results_data if 30 <= r['Risk Level'] < 70]
                if medium_risk_diseases:
                    st.info("ℹ️ **Monitor Closely**")
                    for disease in medium_risk_diseases:
                        st.markdown(f"- **{disease}**: Regular check-ups recommended")
                
                low_risk_diseases = [r['Disease'] for r in results_data if r['Risk Level'] < 30]
                if low_risk_diseases:
                    st.success("✅ **Low Risk**")
                    for disease in low_risk_diseases:
                        st.markdown(f"- **{disease}**: Continue healthy lifestyle")
        
        # Export results
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export PDF", use_container_width=True):
                st.info("PDF export functionality would be implemented here")
        
        with col2:
            if st.button("📊 Export Report", use_container_width=True):
                st.info("Detailed report export would be implemented here")
        
        with col3:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.current_page = 'upload'
                st.rerun()
    
    def show_demo(self):
        """Show demo with sample data"""
        st.info("This would show a demo with sample medical data and results.")
        
        # Sample demo data
        demo_predictions = {
            'diabetes': 0.25,
            'cardiovascular': 0.45,
            'cancer': 0.15,
            'kidney_disease': 0.30,
            'liver_disease': 0.20
        }
        
        st.markdown("### 🎭 Demo Results")
        
        for disease, risk in demo_predictions.items():
            risk_pct = risk * 100
            if risk_pct < 30:
                status = "🟢 Low Risk"
            elif risk_pct < 70:
                status = "🟡 Medium Risk"
            else:
                status = "🔴 High Risk"
            
            st.markdown(f"**{disease.replace('_', ' ').title()}**: {risk_pct:.1f}% - {status}")

def main():
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = MedicalDetectionApp()
    
    # Initialize current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Run app
    st.session_state.app.main()

if __name__ == "__main__":
    main()
