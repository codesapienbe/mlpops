import subprocess
import streamlit as st
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import config_manager, setup_logger
from core import RobustIngestion, LanguageAwareCleaner

logger = setup_logger('web')

class WebApp:
    """Main Streamlit web application"""
    
    def __init__(self):
        self.config = config_manager.config
        self.web_cfg = self.config.get('web', {})
        self.api_cfg = self.config.get('api', {})
    
    def run(self):
        """Main application entry point"""
        # Page config
        st.set_page_config(
            page_title=self.web_cfg.get('title', 'MLOps Job Analysis'),
            layout="wide",
            page_icon=self.web_cfg.get('icon', '🔍')
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header { color: #1f77b4; }
        .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.title(self.web_cfg.get('sidebar_title', 'MLOps Suite'))
            nav_options = {
                "Analyze Jobs": "📊 Analyze",
                "Pipeline Status": "⚙️ Pipeline", 
                "Documentation": "📚 Docs"
            }
            selection = st.radio("Navigation", list(nav_options.values()))
        
        # Route to appropriate page
        if nav_options["Analyze Jobs"] in selection:
            self._render_analysis_page()
        elif nav_options["Pipeline Status"] in selection:
            self._render_pipeline_page()
        else:
            self._render_documentation_page()
    
    def _render_analysis_page(self):
        """Render job analysis interface"""
        st.header("📊 Job Analysis Dashboard")
        
        # Select analysis metric
        metric_option = st.selectbox("Select metric to analyze", ["Requirements Count", "Experience", "Remote"], index=0)
        
        # Single job analysis
        with st.expander("🔍 Single Job Analysis", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                job_title = st.text_input("Job Title", placeholder="Software Engineer")
                company = st.text_input("Company", placeholder="Tech Corp")
                location = st.text_input("Location", placeholder="San Francisco, CA")
            
            with col2:
                description = st.text_area("Description", placeholder="Job description...")
                requirements = st.text_area("Requirements", placeholder="Job requirements...")
                posted_date = st.date_input("Posted Date")
            
            if st.button("Analyze Single Job"):
                if all([job_title, company, location, description, requirements]):
                    # Compute selected metric locally
                    if metric_option == "Requirements Count":
                        count = len(requirements.split(","))
                        st.metric("Requirements Count", count)
                    elif metric_option == "Experience":
                        df_tmp = pd.DataFrame([{ 
                            'job_title': job_title,
                            'company': company,
                            'location': location,
                            'description': description,
                            'requirements': requirements,
                            'posted_date': str(posted_date)
                        }])
                        cleaner = LanguageAwareCleaner(self.config)
                        cleaned = cleaner.clean_data(df_tmp)
                        exp_val = cleaned['experience'].iloc[0]
                        st.metric("Experience (years)", exp_val)
                    elif metric_option == "Remote":
                        df_tmp = pd.DataFrame([{ 
                            'job_title': job_title,
                            'company': company,
                            'location': location,
                            'description': description,
                            'requirements': requirements,
                            'posted_date': str(posted_date)
                        }])
                        cleaner = LanguageAwareCleaner(self.config)
                        cleaned = cleaner.clean_data(df_tmp)
                        remote_val = cleaned['remote'].iloc[0]
                        st.metric("Remote", str(remote_val))
                    st.success("✅ Analysis completed successfully")
                else:
                    st.error("Please fill in all fields")
        
        # Batch analysis
        with st.expander("📊 Batch Analysis"):
            uploaded_file = st.file_uploader(
                "Upload job data (CSV)",
                type=["csv"],
                help="Upload a CSV file with job listings"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    validator = RobustIngestion(self.config)
                    validator._validate_schema(df)
                    
                    st.subheader("Sample Data")
                    st.dataframe(df.head())
                    
                    if st.button("Analyze Jobs"):
                        self._analyze_batch_jobs(df)
                        
                except Exception as e:
                    st.error(f"Data validation error: {str(e)}")
    
    def _analyze_single_job(self, job_data: dict):
        """Analyze single job via API"""
        try:
            with st.spinner("Analyzing job..."):
                url = f"http://{self.api_cfg.get('host', 'localhost')}:{self.api_cfg.get('port', 8000)}/analyze"
                response = requests.post(url, json=job_data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", f"{result['prediction']:.2f}")
                with col2:
                    if 'confidence' in result:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                with col3:
                    if 'processing_time' in result:
                        st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                
                st.success("✅ Analysis completed successfully")
                
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    def _analyze_batch_jobs(self, df: pd.DataFrame):
        """Analyze batch of jobs via API"""
        try:
            with st.spinner("Analyzing batch jobs..."):
                jobs_data = df.to_dict(orient="records")
                url = f"http://{self.api_cfg.get('host', 'localhost')}:{self.api_cfg.get('port', 8000)}/batch-analyze"
                
                response = requests.post(url, json={"jobs": jobs_data}, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                predictions = result.get('predictions', [])
                
                # Display results
                df_results = df.copy()
                df_results['Prediction'] = predictions
                
                st.subheader("Analysis Results")
                st.dataframe(df_results)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Jobs", len(predictions))
                with col2:
                    st.metric("Avg Prediction", f"{sum(predictions)/len(predictions):.2f}")
                with col3:
                    st.metric("Max Prediction", f"{max(predictions):.2f}")
                
                st.success("✅ Batch analysis completed")
                
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Batch analysis failed: {str(e)}")
    
    def _render_pipeline_page(self):
        """Render pipeline status page"""
        st.header("�� Pipeline Status")
        try:
            with st.spinner("Running pipeline evaluation..."):
                url = f"http://{self.api_cfg.get('host','localhost')}:{self.api_cfg.get('port',8000)}/evaluate"
                response = requests.post(url, timeout=120)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            st.error(f"Failed to run pipeline: {e}")
            return
        # Display overall pipeline result
        st.subheader("Pipeline Execution")
        st.write(f"Success: {data.get('success')}" )
        # Display each stage
        results = data.get('results', {})
        for stage_name, res in results.items():
            with st.expander(f"Stage: {stage_name.capitalize()}"):
                st.write(f"Success: {res.get('success')}")
                st.write(f"Duration: {res.get('duration'):.2f}s")
                if res.get('error'):
                    st.error(f"Error: {res.get('error')}")
                if res.get('metrics'):
                    st.write("Metrics:")
                    st.json(res.get('metrics'))
    
    def _render_documentation_page(self):
        """Render documentation page"""
        st.header("📚 Technical Documentation")
        
        # API Documentation
        st.subheader("🔌 API Endpoints")
        
        endpoints = [
            ("POST /analyze", "Analyze single job", "Single job analysis"),
            ("POST /batch-analyze", "Analyze multiple jobs", "Batch job analysis"),
            ("GET /health", "Health check", "System health status"),
            ("GET /metrics", "Prometheus metrics", "System metrics")
        ]
        
        for endpoint, desc, details in endpoints:
            with st.expander(f"`{endpoint}` - {desc}"):
                st.markdown(f"**Description:** {details}")
                st.code(f"curl -X POST http://localhost:8000{endpoint.split()[1]}")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        st.json(self.config)

def main():
    """Streamlit webapp entry point"""
    try:
        app = WebApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Web app error: {str(e)}", exc_info=True)

def cli_main():
    """CLI entry point to run Streamlit"""
    import sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__, "--server.port", "8501"
    ])

if __name__ == "__main__":
    main()
