# NVIDIA NIM GPU Sizing Tool - Complete Codebase

## 📦 **Complete File Structure for Cloudera AI Deployment**

```
nvidia-nim-gpu-sizing-tool/
├── nim_gpu_sizing_app.py          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── launch_app.py                   # Easy launcher script
├── README.md                       # Deployment instructions
└── deployment_summary.md           # This file
```

## 🚀 **Quick Start on Cloudera AI**

### **1. Create Project**
1. Create new Cloudera AI project: `nvidia-nim-gpu-sizing-tool`
2. Upload all 4 files above to the project root

### **2. Launch Application**
```bash
# Option 1: Use the launcher (easiest)
python launch_app.py

# Option 2: Direct Streamlit command
streamlit run nim_gpu_sizing_app.py --server.port 8080 --server.address 0.0.0.0
```

### **3. Access Tool**
- Local: `http://localhost:8080`
- Public: Cloudera AI will provide the shareable URL

## 🎯 **Application Capabilities**

### **Exact Cascading Filter Behavior You Requested**
✅ **Step 1**: Select Model → Shows only available strategies for that model
✅ **Step 2**: Select Strategy → Shows only available GPU types for that model+strategy  
✅ **Step 3**: Select GPU Type → Shows only available GPU counts for that specific combination
✅ **Step 4**: Select GPU Count → Auto-populates final configuration details

### **8 Model Selection Slots**
✅ Each selection row filters independently
✅ Clear button to reset individual selections
✅ Real-time GPU requirements aggregation

### **AWS Instance Recommendations**
✅ Automatically calculates optimal instance types based on total GPU requirements
✅ Cost estimates (hourly, daily, monthly)
✅ Multiple instance options per GPU type

### **Interactive Features**
✅ **Charts**: GPU distribution visualization with Plotly
✅ **Export**: Download configurations as CSV or JSON
✅ **Cost Calculator**: Real-time cost estimation
✅ **Responsive UI**: Works on mobile and desktop

## 📊 **Data Sources**

### **Pre-loaded Model Data (39+ configurations)**
- Llama 3.3 70B Instruct (4 optimization profiles)
- Deepseek R1 Distill Llama (2 profiles)
- StarCoder2-7B (2 profiles)
- Llama 3.2 Instruct (2 profiles)
- Llama 3.1 Nemotron Nano 8b V1 (2 profiles)
- Llama 3.2 Vision Instruct (2 profiles)
- Llama 3.1 Instruct (2 profiles)
- Mistral Instruct (2 profiles)
- Mixtral Instruct (2 profiles)
- NeMo Retriever-Parse (1 profile)
- Llama 3.2 NV EmbedQA 1b V2 (1 profile)
- Llama 3.2 NV RerankQA 1b V2 (1 profile)

### **AWS Instance Pricing (June 2025 rates)**
- P5.48xlarge (H100): $54.00/hour
- P4d.24xlarge (A100): $18.00/hour
- P4de.24xlarge (A100): $22.50/hour
- G5.12xlarge (A10G): $5.67/hour
- G5.48xlarge (A10G): $15.90/hour
- G6e.48xlarge (L40S): $65.40/hour
- G6.48xlarge (L4): $17.57/hour

## 🔧 **Technical Architecture**

### **Framework**: Streamlit (Perfect for Cloudera AI)
- **Why Streamlit**: 
  - Native support in Cloudera AI
  - Rapid development and deployment
  - Interactive widgets out-of-the-box
  - Easy sharing and collaboration

### **Key Libraries**:
- **Pandas**: Data manipulation and filtering
- **Plotly**: Interactive charts and visualizations
- **NumPy**: Numerical calculations
- **Streamlit**: Web application framework

### **State Management**:
- Session state for maintaining 8 model selections
- Real-time updates with `st.rerun()`
- Cached data loading for performance

## 💡 **Advanced Features**

### **Smart Filtering Logic**
```python
def get_filtered_options(df, model=None, strategy=None, gpu_type=None):
    # Progressively filters available options based on previous selections
    # Ensures only valid combinations are shown
```

### **Dynamic Cost Calculation**
```python
def generate_instance_recommendations(gpu_summary, aws_df):
    # Calculates optimal instance configurations
    # Provides multiple cost scenarios
    # Handles GPU resource optimization
```

### **Export Functionality**
- CSV export for spreadsheet analysis
- JSON export for programmatic use
- Configuration summary tables

## 🔄 **Customization Guide**

### **Add New Models**
Edit `load_nim_data()` function:
```python
nim_data.append({
    "model": "Your New Model",
    "strategy": "Latency", 
    "gpu_type": "H100",
    "gpu_count": 4,
    # ... other fields
})
```

### **Update Pricing**
Edit `load_aws_instances()` function:
```python
aws_data.append({
    "instance_type": "New.Instance",
    "gpu_type": "H100",
    "gpus_per_instance": 8,
    "cost_per_hour": 60.00
})
```

### **Add Cloud Providers**
1. Create new data loading function (e.g., `load_gcp_instances()`)
2. Add new recommendation logic
3. Create additional UI tabs or sections

## 📈 **Performance Optimizations**

### **Caching Strategy**
```python
@st.cache_data
def load_nim_data():
    # Data loaded once per session
    # Improves app responsiveness
```

### **Efficient State Management**
- Minimal session state usage
- Smart update triggers
- Clean state transitions

## 🚀 **Deployment Advantages**

### **Why This Solution is Perfect for Cloudera AI**:
1. **Native Integration**: Streamlit works seamlessly in Cloudera AI
2. **No External Dependencies**: All data embedded, no API calls needed
3. **Collaborative**: Easy to share URLs with team members
4. **Scalable**: Can handle multiple concurrent users
5. **Maintainable**: Pure Python code, easy to modify and extend

### **vs Excel Approach**:
- ✅ **Better UX**: True cascading dropdowns without formula complexity
- ✅ **Real-time**: Instant calculations and visualizations
- ✅ **Shareable**: Web-based, accessible from anywhere
- ✅ **Extensible**: Easy to add features, integrate with other systems
- ✅ **No Formula Errors**: Robust Python logic vs fragile Excel formulas

## 🎯 **Next Steps**

1. **Deploy** using the files provided
2. **Test** the cascading filter behavior
3. **Customize** with your specific models/pricing
4. **Extend** with additional cloud providers or features
5. **Integrate** with Cloudera AI model registry if needed

**Ready to deploy!** This gives you exactly the cascading filter behavior you wanted, deployed on Cloudera AI with minimal setup required. 🚀