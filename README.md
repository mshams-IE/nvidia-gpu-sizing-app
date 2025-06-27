# NVIDIA NIM GPU Sizing Tool - Cloudera AI Deployment Guide

## 📋 **Files Required for Deployment**

Your Cloudera AI project needs these files:
- `nim_gpu_sizing_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - This deployment guide

## 🚀 **Deployment Steps for Cloudera AI**

### **Step 1: Create New Project in Cloudera AI**

1. **Log into your Cloudera AI workspace**
2. **Create a new project:**
   - Click "New Project"
   - Project Name: `nvidia-nim-gpu-sizing-tool`
   - Description: `Interactive tool for sizing NVIDIA NIM GPU deployments`
   - Template: `Python` or `Blank Project`

### **Step 2: Upload Project Files**

1. **Upload the application files:**
   ```bash
   # In your Cloudera AI project terminal or file browser
   # Upload these files to the project root:
   nim_gpu_sizing_app.py
   requirements.txt
   ```

2. **Verify file structure:**
   ```
   /your-project/
   ├── nim_gpu_sizing_app.py
   ├── requirements.txt
   └── README.md
   ```

### **Step 3: Set up Python Environment**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Or install manually:**
   ```bash
   pip install streamlit pandas numpy plotly openpyxl
   ```

### **Step 4: Run the Application**

**Option A: Terminal Launch**
```bash
streamlit run nim_gpu_sizing_app.py --server.port 8080 --server.address 0.0.0.0
```

**Option B: Create a Launcher Script**
Create `launch_app.py`:
```python
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "nim_gpu_sizing_app.py", 
        "--server.port", "8080",
        "--server.address", "0.0.0.0"
    ])
```

Then run: `python launch_app.py`

### **Step 5: Access the Application**

1. **The app will be available at:**
   - Local URL: `http://localhost:8080`
   - Network URL: `http://[your-cloudera-instance]:8080`

2. **Cloudera AI will provide the public URL for sharing**

## 🔧 **Cloudera AI Specific Configuration**

### **Environment Variables (if needed)**
```bash
# Set in your Cloudera AI session
export STREAMLIT_SERVER_PORT=8080
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_ENABLE_CORS=false
```

### **Memory Requirements**
- **Minimum**: 2GB RAM
- **Recommended**: 4GB RAM for better performance
- **CPU**: 2 vCPUs minimum

### **Session Configuration**
```python
# Add to the top of nim_gpu_sizing_app.py if needed
import os
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '200'
```

## 📊 **Application Features**

### **Core Functionality**
✅ **Cascading Model Selection**: Choose from 12+ NVIDIA NIM models
✅ **Dynamic Filtering**: Options filter based on previous selections  
✅ **GPU Requirements**: Automatic calculation of GPU needs by type
✅ **Cost Estimation**: AWS instance recommendations with pricing
✅ **Interactive Charts**: Plotly visualizations for GPU distribution
✅ **Export Options**: Download configurations as CSV or JSON

### **Data Sources**
- Pre-loaded NVIDIA NIM optimization data
- AWS instance pricing (updated June 2025)
- GPU specifications and memory requirements

## 🛠️ **Customization Options**

### **Update Model Data**
Edit the `load_nim_data()` function in `nim_gpu_sizing_app.py` to add new models or update specifications.

### **Update AWS Pricing**
Edit the `load_aws_instances()` function to update instance types and pricing.

### **Add New Cloud Providers**
Extend the application by:
1. Adding new data loading functions for GCP, Azure, etc.
2. Creating additional recommendation functions
3. Adding new tabs or sections to the UI

## 🔍 **Troubleshooting**

### **Common Issues**

**Port Already in Use:**
```bash
# Find and kill existing Streamlit processes
ps aux | grep streamlit
kill [process_id]

# Or use a different port
streamlit run nim_gpu_sizing_app.py --server.port 8081
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install --upgrade streamlit pandas numpy plotly openpyxl
```

**Memory Issues:**
```bash
# Reduce data size or increase Cloudera AI session memory
# Edit the app to use @st.cache_data more aggressively
```

### **Performance Optimization**

1. **Enable caching:**
   ```python
   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def load_nim_data():
       # ... your data loading code
   ```

2. **Optimize session state:**
   ```python
   # Clear unused session state periodically
   if len(st.session_state.selections) > 8:
       st.session_state.selections = st.session_state.selections[:8]
   ```

## 📈 **Usage Analytics**

### **Track Usage (Optional)**
Add analytics to understand tool usage:
```python
import datetime

# Add to main() function
st.sidebar.markdown(f"**Session Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Log selections
if gpu_summary:
    st.sidebar.markdown(f"**Models Selected:** {len([s for s in st.session_state.selections if s.get('configuration')])}")
```

## 🔐 **Security Considerations**

1. **Data Privacy**: All calculations run locally in your Cloudera AI environment
2. **No External APIs**: App doesn't make external network calls
3. **File Access**: App only reads local data files

## 📝 **Next Steps**

1. **Deploy and test** the basic application
2. **Customize data** with your specific models and pricing
3. **Add features** like:
   - Multi-cloud support (Azure, GCP)
   - Custom model configurations
   - Integration with Cloudera AI model registry
   - Export to Excel with formatting

## 🆘 **Support**

For issues or enhancements:
1. Check Cloudera AI documentation for platform-specific issues
2. Streamlit documentation for UI/functionality questions
3. Customize the code based on your specific requirements

---

**Ready to deploy!** 🚀 Your NVIDIA NIM GPU sizing tool will help teams make informed infrastructure decisions for AI deployments.