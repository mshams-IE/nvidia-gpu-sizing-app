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

    # Get the port assigned by Cloudera AI
    port = os.environ.get("CDSW_APP_PORT", "8100")
    
    print(f"Starting NVIDIA GPU Sizing App on port {port}")
    
    # Launch Streamlit with Cloudera-compatible settings
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "nim_gpu_sizing_app.py",
        "--server.port", port,
        "--server.address", "127.0.0.1"  # Cloudera uses 127.0.0.1, not 0.0.0.0
    ])

    # port = os.environ.get('CDSW_APP_PORT', '8100')
    
    # print(f"Starting Streamlit on port {port}")
    
    # # Set environment variables
    # os.environ['STREAMLIT_SERVER_PORT'] = port
    # os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    # os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    # os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
    # # Launch Streamlit with the correct port
    # cmd = [
    #     sys.executable, '-m', 'streamlit', 'run',
    #     'nim_gpu_sizing_app.py',
    #     '--server.port', port,
    #     '--server.address', '0.0.0.0',
    #     '--server.headless', 'true',
    #     '--server.enableCORS', 'false'
    # ]
    
    # print(f"Running command: {' '.join(cmd)}")
    # subprocess.run(cmd)

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