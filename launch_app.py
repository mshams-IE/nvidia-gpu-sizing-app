#!/usr/bin/env python3
"""
NVIDIA NIM GPU Sizing Tool Launcher for Cloudera AI
Simple script to launch the Streamlit application with proper configuration
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not already installed"""
    try:
        import streamlit
        import pandas
        import plotly
        print("✅ All required packages are already installed")
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "streamlit", "pandas", "numpy", "plotly", "openpyxl"
        ])
        print("✅ Package installation complete")

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching NVIDIA NIM GPU Sizing Tool...")
    print("📊 Application will be available at: http://localhost:8080")
    print("🔗 Cloudera AI will provide the public URL for sharing")
    print("⏹️  Press Ctrl+C to stop the application")
    
    # Set environment variables for Cloudera AI
    os.environ['STREAMLIT_SERVER_PORT'] = '8080'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "nim_gpu_sizing_app.py",
        "--server.port", "8080",
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ])

if __name__ == "__main__":
    print("🔧 NVIDIA NIM GPU Sizing Tool - Cloudera AI Launcher")
    print("=" * 60)
    
    # Check if main app file exists
    if not os.path.exists("nim_gpu_sizing_app.py"):
        print("❌ Error: nim_gpu_sizing_app.py not found in current directory")
        print("📁 Make sure you're in the correct project directory")
        sys.exit(1)
    
    # Install requirements
    install_requirements()
    
    # Launch application
    launch_app()