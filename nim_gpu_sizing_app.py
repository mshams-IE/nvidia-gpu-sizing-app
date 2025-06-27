import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="NVIDIA NIM GPU Sizing Tool",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > label {
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selections' not in st.session_state:
    st.session_state.selections = [{}] * 8

@st.cache_data
def load_nim_data():
    """Load NVIDIA NIM optimization data"""
    nim_data = [
        {"model": "Llama 3.3 70B Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 4, "memory_per_gpu": "80 GB", "total_memory": "320 GB", "use_case": "Real-time inference", "notes": "Requires 400GB CPU memory"},
        {"model": "Llama 3.3 70B Instruct", "strategy": "Throughput", "gpu_type": "H100", "gpu_count": 8, "memory_per_gpu": "80 GB", "total_memory": "640 GB", "use_case": "Batch processing", "notes": "Best for high-volume requests"},
        {"model": "Llama 3.3 70B Instruct", "strategy": "Latency", "gpu_type": "A100", "gpu_count": 8, "memory_per_gpu": "80 GB", "total_memory": "640 GB", "use_case": "Real-time inference", "notes": "80GB A100 recommended"},
        {"model": "Llama 3.3 70B Instruct", "strategy": "Throughput", "gpu_type": "L40S", "gpu_count": 8, "memory_per_gpu": "48 GB", "total_memory": "384 GB", "use_case": "Cost-efficient batch", "notes": "Cloudera requires 8 L40S not 4"},
        {"model": "Deepseek R1 Distill Llama", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Fast reasoning", "notes": "Distilled for efficiency"},
        {"model": "Deepseek R1 Distill Llama", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Multi-user serving", "notes": "Good price/performance"},
        {"model": "Deepseek R1 Distill Llama", "strategy": "Latency", "gpu_type": "L40S", "gpu_count": 1, "memory_per_gpu": "48 GB", "total_memory": "48 GB", "use_case": "Edge deployment", "notes": "Single GPU deployment"},
        {"model": "StarCoder2-7B", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Code completion", "notes": "Set NIM_MAX_MODEL_LEN=4096"},
        {"model": "StarCoder2-7B", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Batch code generation", "notes": "Multi-user development"},
        {"model": "StarCoder2-7B", "strategy": "Latency", "gpu_type": "L40S", "gpu_count": 1, "memory_per_gpu": "48 GB", "total_memory": "48 GB", "use_case": "Developer workstation", "notes": "Cost-effective coding assistant"},
        {"model": "Llama 3.2 Instruct", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Lightweight inference", "notes": "Small footprint"},
        {"model": "Llama 3.2 Instruct", "strategy": "Throughput", "gpu_type": "L4", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Edge computing", "notes": "Very efficient"},
        {"model": "Llama 3.2 Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Production inference", "notes": "Balanced performance"},
        {"model": "Llama 3.2 Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Multi-user apps", "notes": "Good scalability"},
        {"model": "Llama 3.1 Nemotron Nano 8b V1", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Efficient inference", "notes": "NVIDIA optimized"},
        {"model": "Llama 3.1 Nemotron Nano 8b V1", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Production serving", "notes": "Balanced cost/performance"},
        {"model": "Llama 3.1 Nemotron Nano 8b V1", "strategy": "Latency", "gpu_type": "L40S", "gpu_count": 1, "memory_per_gpu": "48 GB", "total_memory": "48 GB", "use_case": "Cost-efficient inference", "notes": "Single GPU deployment"},
        {"model": "Llama 3.2 Vision Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 2, "memory_per_gpu": "80 GB", "total_memory": "160 GB", "use_case": "Multimodal real-time", "notes": "Image + text processing"},
        {"model": "Llama 3.2 Vision Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 2, "memory_per_gpu": "80 GB", "total_memory": "160 GB", "use_case": "Fast multimodal", "notes": "Optimized for speed"},
        {"model": "Llama 3.2 Vision Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 4, "memory_per_gpu": "24 GB", "total_memory": "96 GB", "use_case": "Batch vision tasks", "notes": "Document processing"},
        {"model": "Llama 3.2 Vision Instruct", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 8, "memory_per_gpu": "24 GB", "total_memory": "192 GB", "use_case": "High-throughput vision", "notes": "Multi-user vision apps"},
        {"model": "Llama 3.1 Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Fast text generation", "notes": "General purpose"},
        {"model": "Llama 3.1 Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Production serving", "notes": "Scalable deployment"},
        {"model": "Llama 3.1 Instruct", "strategy": "Latency", "gpu_type": "L40S", "gpu_count": 1, "memory_per_gpu": "48 GB", "total_memory": "48 GB", "use_case": "Cost-effective serving", "notes": "Budget-friendly"},
        {"model": "Llama 3.1 Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 4, "memory_per_gpu": "80 GB", "total_memory": "320 GB", "use_case": "High-performance inference", "notes": "Premium performance"},
        {"model": "Llama 3.1 Instruct", "strategy": "Throughput", "gpu_type": "H100", "gpu_count": 8, "memory_per_gpu": "80 GB", "total_memory": "640 GB", "use_case": "Large-scale serving", "notes": "Maximum throughput"},
        {"model": "Llama 3.1 Instruct", "strategy": "Throughput", "gpu_type": "L40S", "gpu_count": 8, "memory_per_gpu": "48 GB", "total_memory": "384 GB", "use_case": "Cost-efficient large model", "notes": "Cloudera requires 8 GPUs"},
        {"model": "Mistral Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Fast multilingual", "notes": "European LLM"},
        {"model": "Mistral Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Production multilingual", "notes": "Good for European languages"},
        {"model": "Mistral Instruct", "strategy": "Latency", "gpu_type": "L40S", "gpu_count": 1, "memory_per_gpu": "48 GB", "total_memory": "48 GB", "use_case": "Edge multilingual", "notes": "Single GPU deployment"},
        {"model": "Mixtral Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 2, "memory_per_gpu": "80 GB", "total_memory": "160 GB", "use_case": "MoE fast inference", "notes": "Mixture of Experts"},
        {"model": "Mixtral Instruct", "strategy": "Throughput", "gpu_type": "H100", "gpu_count": 4, "memory_per_gpu": "80 GB", "total_memory": "320 GB", "use_case": "MoE batch processing", "notes": "High-volume MoE"},
        {"model": "Mixtral Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 8, "memory_per_gpu": "24 GB", "total_memory": "192 GB", "use_case": "Cost-efficient MoE", "notes": "Multi-user MoE serving"},
        {"model": "Mixtral Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 4, "memory_per_gpu": "80 GB", "total_memory": "320 GB", "use_case": "Large MoE inference", "notes": "Premium MoE performance"},
        {"model": "Mixtral Instruct", "strategy": "Throughput", "gpu_type": "H100", "gpu_count": 8, "memory_per_gpu": "80 GB", "total_memory": "640 GB", "use_case": "Large MoE batch", "notes": "Maximum MoE throughput"},
        {"model": "NeMo Retriever-Parse", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Document parsing", "notes": "Text extraction"},
        {"model": "NeMo Retriever-Parse", "strategy": "Throughput", "gpu_type": "L40S", "gpu_count": 2, "memory_per_gpu": "48 GB", "total_memory": "96 GB", "use_case": "Batch document processing", "notes": "High-volume parsing"},
        {"model": "Llama 3.2 NV EmbedQA 1b V2", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Question answering", "notes": "Lightweight QA"},
        {"model": "Llama 3.2 NV EmbedQA 1b V2", "strategy": "Throughput", "gpu_type": "L4", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Edge QA", "notes": "Very efficient QA"},
        {"model": "Llama 3.2 NV RerankQA 1b V2", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Search reranking", "notes": "Lightweight reranking"},
        {"model": "Llama 3.2 NV RerankQA 1b V2", "strategy": "Throughput", "gpu_type": "L4", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Edge reranking", "notes": "Efficient search optimization"}]

    return pd.DataFrame(nim_data)

@st.cache_data
def load_aws_instances():
    """Load complete AWS instance pricing data"""
    aws_data = [
        # T4 Instances
        {"instance_type": "G4dn.xlarge", "gpu_type": "T4", "gpus_per_instance": 1, "cost_per_hour": 0.526, "total_gpu_memory": "16 GB"},
        {"instance_type": "G4dn.2xlarge", "gpu_type": "T4", "gpus_per_instance": 1, "cost_per_hour": 0.752, "total_gpu_memory": "16 GB"},
        {"instance_type": "G4dn.4xlarge", "gpu_type": "T4", "gpus_per_instance": 1, "cost_per_hour": 1.204, "total_gpu_memory": "16 GB"},
        {"instance_type": "G4dn.8xlarge", "gpu_type": "T4", "gpus_per_instance": 1, "cost_per_hour": 2.176, "total_gpu_memory": "16 GB"},
        {"instance_type": "G4dn.12xlarge", "gpu_type": "T4", "gpus_per_instance": 4, "cost_per_hour": 3.912, "total_gpu_memory": "64 GB"},
        {"instance_type": "G4dn.16xlarge", "gpu_type": "T4", "gpus_per_instance": 1, "cost_per_hour": 4.352, "total_gpu_memory": "16 GB"},
        {"instance_type": "G4dn.metal", "gpu_type": "T4", "gpus_per_instance": 8, "cost_per_hour": 7.824, "total_gpu_memory": "128 GB"},
        
        # A10G Instances
        {"instance_type": "G5.xlarge", "gpu_type": "A10G", "gpus_per_instance": 1, "cost_per_hour": 1.006, "total_gpu_memory": "24 GB"},
        {"instance_type": "G5.2xlarge", "gpu_type": "A10G", "gpus_per_instance": 1, "cost_per_hour": 1.212, "total_gpu_memory": "24 GB"},
        {"instance_type": "G5.4xlarge", "gpu_type": "A10G", "gpus_per_instance": 1, "cost_per_hour": 1.624, "total_gpu_memory": "24 GB"},
        {"instance_type": "G5.8xlarge", "gpu_type": "A10G", "gpus_per_instance": 1, "cost_per_hour": 2.448, "total_gpu_memory": "24 GB"},
        {"instance_type": "G5.16xlarge", "gpu_type": "A10G", "gpus_per_instance": 1, "cost_per_hour": 4.096, "total_gpu_memory": "24 GB"},
        {"instance_type": "G5.12xlarge", "gpu_type": "A10G", "gpus_per_instance": 4, "cost_per_hour": 5.672, "total_gpu_memory": "96 GB"},
        {"instance_type": "G5.24xlarge", "gpu_type": "A10G", "gpus_per_instance": 4, "cost_per_hour": 8.392, "total_gpu_memory": "96 GB"},
        {"instance_type": "G5.48xlarge", "gpu_type": "A10G", "gpus_per_instance": 8, "cost_per_hour": 15.896, "total_gpu_memory": "192 GB"},
        
        # L4 Instances
        {"instance_type": "G6.xlarge", "gpu_type": "L4", "gpus_per_instance": 1, "cost_per_hour": 1.212, "total_gpu_memory": "24 GB"},
        {"instance_type": "G6.2xlarge", "gpu_type": "L4", "gpus_per_instance": 1, "cost_per_hour": 1.424, "total_gpu_memory": "24 GB"},
        {"instance_type": "G6.4xlarge", "gpu_type": "L4", "gpus_per_instance": 1, "cost_per_hour": 1.848, "total_gpu_memory": "24 GB"},
        {"instance_type": "G6.8xlarge", "gpu_type": "L4", "gpus_per_instance": 1, "cost_per_hour": 2.696, "total_gpu_memory": "24 GB"},
        {"instance_type": "G6.12xlarge", "gpu_type": "L4", "gpus_per_instance": 4, "cost_per_hour": 6.784, "total_gpu_memory": "96 GB"},
        {"instance_type": "G6.16xlarge", "gpu_type": "L4", "gpus_per_instance": 1, "cost_per_hour": 4.392, "total_gpu_memory": "24 GB"},
        {"instance_type": "G6.48xlarge", "gpu_type": "L4", "gpus_per_instance": 8, "cost_per_hour": 17.568, "total_gpu_memory": "192 GB"},
        
        # L40S Instances (The main fix!)
        {"instance_type": "G6e.xlarge", "gpu_type": "L40S", "gpus_per_instance": 1, "cost_per_hour": 1.86, "total_gpu_memory": "48 GB"},
        {"instance_type": "G6e.2xlarge", "gpu_type": "L40S", "gpus_per_instance": 1, "cost_per_hour": 2.27, "total_gpu_memory": "48 GB"},
        {"instance_type": "G6e.4xlarge", "gpu_type": "L40S", "gpus_per_instance": 1, "cost_per_hour": 4.54, "total_gpu_memory": "48 GB"},
        {"instance_type": "G6e.8xlarge", "gpu_type": "L40S", "gpus_per_instance": 1, "cost_per_hour": 9.08, "total_gpu_memory": "48 GB"},
        {"instance_type": "G6e.12xlarge", "gpu_type": "L40S", "gpus_per_instance": 4, "cost_per_hour": 16.35, "total_gpu_memory": "192 GB"},
        {"instance_type": "G6e.16xlarge", "gpu_type": "L40S", "gpus_per_instance": 1, "cost_per_hour": 18.16, "total_gpu_memory": "48 GB"},
        {"instance_type": "G6e.24xlarge", "gpu_type": "L40S", "gpus_per_instance": 4, "cost_per_hour": 32.7, "total_gpu_memory": "192 GB"},
        {"instance_type": "G6e.48xlarge", "gpu_type": "L40S", "gpus_per_instance": 8, "cost_per_hour": 65.4, "total_gpu_memory": "384 GB"},
        
        # V100 Instances
        {"instance_type": "P3.2xlarge", "gpu_type": "V100", "gpus_per_instance": 1, "cost_per_hour": 3.06, "total_gpu_memory": "16 GB"},
        {"instance_type": "P3.8xlarge", "gpu_type": "V100", "gpus_per_instance": 4, "cost_per_hour": 12.24, "total_gpu_memory": "64 GB"},
        {"instance_type": "P3.16xlarge", "gpu_type": "V100", "gpus_per_instance": 8, "cost_per_hour": 24.48, "total_gpu_memory": "128 GB"},
        {"instance_type": "P3dn.24xlarge", "gpu_type": "V100", "gpus_per_instance": 8, "cost_per_hour": 31.212, "total_gpu_memory": "256 GB"},
        
        # A100 Instances  
        {"instance_type": "P4d.24xlarge", "gpu_type": "A100", "gpus_per_instance": 8, "cost_per_hour": 18.00, "total_gpu_memory": "320 GB"},
        {"instance_type": "P4de.24xlarge", "gpu_type": "A100", "gpus_per_instance": 8, "cost_per_hour": 22.50, "total_gpu_memory": "640 GB"},
        
        # H100 Instances
        {"instance_type": "P5.48xlarge", "gpu_type": "H100", "gpus_per_instance": 8, "cost_per_hour": 54.00, "total_gpu_memory": "640 GB"},
        
        # H200 Instances
        {"instance_type": "P5e.48xlarge", "gpu_type": "H200", "gpus_per_instance": 8, "cost_per_hour": 65.00, "total_gpu_memory": "1128 GB"},
        {"instance_type": "P5en.48xlarge", "gpu_type": "H200", "gpus_per_instance": 8, "cost_per_hour": 70.00, "total_gpu_memory": "1128 GB"},
        
        # B200 Instances
        {"instance_type": "P6-B200.48xlarge", "gpu_type": "B200", "gpus_per_instance": 8, "cost_per_hour": 120.00, "total_gpu_memory": "1440 GB"},
    ]
    return pd.DataFrame(aws_data)
def get_filtered_options(df: pd.DataFrame, model: str = None) -> Dict[str, List]:
    """Get filtered options based on model selection"""
    filtered_df = df.copy()
    
    if model:
        filtered_df = filtered_df[filtered_df['model'] == model]
    
    # Create complete configuration options for the selected model
    configurations = []
    if model:
        for _, row in filtered_df.iterrows():
            config_display = f"{row['strategy']} | {row['gpu_type']} x{row['gpu_count']} | {row['total_memory']}"
            configurations.append({
                'display': config_display,
                'data': row.to_dict()
            })
    
    return {
        'models': sorted(df['model'].unique().tolist()),
        'configurations': configurations
    }
def calculate_gpu_summary(selections: List[Dict]) -> Dict[str, int]:
    """Calculate total GPU requirements by type"""
    gpu_summary = {}
    for selection in selections:
        if selection and 'configuration' in selection and selection['configuration'] is not None:
            gpu_type = selection['configuration']['gpu_type']
            gpu_count = selection['configuration']['gpu_count']
            gpu_summary[gpu_type] = gpu_summary.get(gpu_type, 0) + gpu_count
    return gpu_summary

def generate_instance_recommendations(gpu_summary: Dict[str, int], aws_df: pd.DataFrame) -> List[Dict]:
    """Generate AWS instance recommendations based on GPU requirements"""
    recommendations = []
    
    for gpu_type, total_gpus in gpu_summary.items():
        if total_gpus > 0:
            # Find best matching instances
            matching_instances = aws_df[aws_df['gpu_type'] == gpu_type].copy()
            
            for _, instance in matching_instances.iterrows():
                instances_needed = int(np.ceil(total_gpus / instance['gpus_per_instance']))
                total_cost = instances_needed * instance['cost_per_hour']
                
                recommendations.append({
                    'gpu_type': gpu_type,
                    'total_gpus': total_gpus,
                    'instance_type': instance['instance_type'],
                    'gpus_per_instance': instance['gpus_per_instance'],
                    'instances_needed': instances_needed,
                    'cost_per_hour': instance['cost_per_hour'],
                    'total_cost': total_cost,
                    'monthly_cost': total_cost * 24 * 30
                })
    
    return sorted(recommendations, key=lambda x: x['total_cost'])

# Main application
# Main application
def main():
    st.title("🚀 NVIDIA NIM GPU Sizing Tool")
    st.markdown("**Optimize your GPU infrastructure for NVIDIA NIM model deployments**")
    
    # Load data
    nim_df = load_nim_data()
    aws_df = load_aws_instances()
    
    # Sidebar instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Select Model**: Choose from available NVIDIA NIM models
        2. **Choose Complete Configuration**: Pick the full setup (Strategy + GPU + Count)
        3. **Repeat**: Add up to 8 model configurations
        4. **Review**: Check GPU summary and AWS recommendations
        """)
        
        st.header("ℹ️ About")
        st.markdown("""
        This tool helps you:
        - Size GPU requirements for NVIDIA NIM deployments
        - Compare optimization strategies
        - Get AWS instance recommendations
        - Calculate infrastructure costs
        """)
# Model selection section
    st.header("🔧 Model Configuration (Select up to 8 models)")
    
    # Create columns for better layout
    cols = st.columns([1, 3, 4, 1])
    with cols[0]:
        st.write("**#**")
    with cols[1]:
        st.write("**Model**")
    with cols[2]:
        st.write("**Complete Configuration**")
    with cols[3]:
        st.write("**Clear**")
    
    # Model selection rows
    for i in range(8):
        cols = st.columns([1, 3, 4, 1])
        
        with cols[0]:
            st.write(f"{i+1}")
        
        # Get current selection
        current_selection = st.session_state.selections[i] if i < len(st.session_state.selections) else {}
        
        with cols[1]:
            # Model dropdown
            model_key = f"model_{i}"
            selected_model = st.selectbox(
                "Model",
                [""] + sorted(nim_df['model'].unique().tolist()),
                index=0 if not current_selection.get('model') else sorted(nim_df['model'].unique().tolist()).index(current_selection['model']) + 1,
                key=model_key,
                label_visibility="collapsed"
            )
        
        with cols[2]:
            # Complete Configuration dropdown
            if selected_model:
                # Get configurations for this model
                model_configs = nim_df[nim_df['model'] == selected_model]
                config_options = []
                
                for _, row in model_configs.iterrows():
                    config_display = f"{row['strategy']} | {row['gpu_type']} x{row['gpu_count']} | {row['total_memory']}"
                    config_options.append({
                        'display': config_display,
                        'data': row.to_dict()
                    })
                
                config_displays = [config['display'] for config in config_options]
                
                config_key = f"config_{i}"
                selected_config_display = st.selectbox(
                    "Complete Configuration",
                    [""] + config_displays,
                    key=config_key,
                    label_visibility="collapsed"
                )
                
                # Update session state when configuration is selected
                if selected_config_display:
                    # Find the matching configuration data
                    selected_config_data = None
                    for config in config_options:
                        if config['display'] == selected_config_display:
                            selected_config_data = config['data']
                            break
                    
                    if selected_config_data:
                        st.session_state.selections[i] = {
                            'model': selected_model,
                            'config_display': selected_config_display,
                            'strategy': selected_config_data['strategy'],
                            'gpu_type': selected_config_data['gpu_type'],
                            'gpu_count': selected_config_data['gpu_count'],
                            'configuration': selected_config_data
                        }
                else:
                    # Clear configuration but keep model
                    st.session_state.selections[i] = {'model': selected_model}
            else:
                # No model selected, show disabled dropdown
                st.selectbox(
                    "Complete Configuration",
                    ["Select a model first"],
                    disabled=True,
                    key=f"config_{i}_disabled",
                    label_visibility="collapsed"
                )
                # Clear this selection if model is cleared
                if current_selection:
                    st.session_state.selections[i] = {}
        
        with cols[3]:
            # Clear button
            if st.button("🗑️", key=f"clear_{i}", help="Clear this selection"):
                st.session_state.selections[i] = {}
        
        # Show configuration details
        if current_selection.get('configuration'):
            config = current_selection['configuration']
            st.markdown(f"<small>**{config['total_memory']}** total | **{config['use_case']}** | {config['notes']}</small>", unsafe_allow_html=True)
    
    # Quick Calculate Button
    st.divider()
    
    # Check if we have any valid selections
    valid_selections = [s for s in st.session_state.selections if s.get('configuration')]
    
    if valid_selections:
        col1, col2, col3 = st.columns([2, 2, 4])
        
        with col1:
            st.metric("Models Selected", len(valid_selections))
        
        with col2:
            if st.button("🚀 Calculate GPU Requirements", type="primary", use_container_width=True):
                st.session_state.show_results = True
        
        with col3:
            if st.button("🗑️ Clear All Selections", use_container_width=True):
                st.session_state.selections = [{}] * 8
                st.session_state.show_results = False
        
        # Show immediate results when button is clicked
        if st.session_state.get('show_results', False):
            st.header("⚡ Quick GPU Requirements & Cost Analysis")
            
            # Calculate GPU summary
            gpu_summary = calculate_gpu_summary(st.session_state.selections)
            
            if gpu_summary:
                # Show GPU summary in a compact format
                st.subheader("📊 GPU Requirements")
                
                gpu_cols = st.columns(len(gpu_summary))
                for i, (gpu_type, count) in enumerate(gpu_summary.items()):
                    with gpu_cols[i]:
                        st.metric(gpu_type, f"{count} GPUs")
                
                # Calculate and show instance recommendations
                st.subheader("💰 Recommended AWS Instances & Cost")
                
                recommendations = generate_instance_recommendations(gpu_summary, aws_df)
                
                if recommendations:
                    # Show best recommendation for each GPU type
                    for gpu_type in gpu_summary.keys():
                        gpu_recs = [r for r in recommendations if r['gpu_type'] == gpu_type]
                        if gpu_recs:
                            best_rec = min(gpu_recs, key=lambda x: x['total_cost'])
                            
                            st.info(f"""
                            **{gpu_type} Recommendation:**
                            - **Instance:** {best_rec['instance_type']} x{best_rec['instances_needed']}
                            - **Cost:** ${best_rec['total_cost']:.2f}/hour | ${best_rec['total_cost'] * 24:.2f}/day | ${best_rec['monthly_cost']:,.2f}/month
                            """)
                    
                    # Total cost summary
                    # FIXED - sums only BEST recommendation for each GPU type
                    best_recommendations = {}
                    for rec in recommendations:
                        gpu_type = rec['gpu_type']
                        if gpu_type in gpu_summary:  # Only include GPU types we actually need
                            if gpu_type not in best_recommendations or rec['total_cost'] < best_recommendations[gpu_type]['total_cost']:
                                best_recommendations[gpu_type] = rec
                    
                    total_hourly = sum(rec['total_cost'] for rec in best_recommendations.values())
                    #total_hourly = sum(rec['total_cost'] for rec in recommendations if any(rec['gpu_type'] == gpu_type for gpu_type in gpu_summary.keys()))
                    
                    st.success(f"""
                    ### 💵 Total Infrastructure Cost
                    - **Hourly:** ${total_hourly:.2f}
                    - **Daily (24/7):** ${total_hourly * 24:.2f}
                    - **Monthly (24/7):** ${total_hourly * 24 * 30:,.2f}
                    """)
                    
                else:
                    st.warning("No instance recommendations available for current GPU selection")
            
            # Show selected configurations
            st.subheader("📋 Your Selected Configurations")
            config_data = []
            for i, selection in enumerate(st.session_state.selections):
                if selection.get('configuration'):
                    config = selection['configuration']
                    config_data.append({
                        '#': i + 1,
                        'Model': config['model'],
                        'Strategy': config['strategy'],
                        'GPU': f"{config['gpu_type']} x{config['gpu_count']}",
                        'Memory': config['total_memory'],
                        'Use Case': config['use_case']
                    })
            
            if config_data:
                st.dataframe(pd.DataFrame(config_data), use_container_width=True)
    
    else:
        st.info("👆 Select models and configurations above to see GPU requirements and cost analysis")
    
    # Only show the full detailed sections if user wants to see them
    if st.session_state.get('show_results', False) and valid_selections:
        
        if st.button("📊 Show Detailed Analysis", type="secondary"):
            st.session_state.show_detailed = True
        
        # Rest of your existing detailed analysis code (GPU summary charts, etc.)
        if st.session_state.get('show_detailed', False):
            
            st.divider()
            
            # GPU Requirements Summary (your existing code)
            st.header("📊 Detailed GPU Analysis")
            
            gpu_summary = calculate_gpu_summary(st.session_state.selections)
            
            if gpu_summary:
                # Create visualization
                if any(gpu_summary.values()):
                    fig = px.bar(
                        x=list(gpu_summary.keys()),
                        y=list(gpu_summary.values()),
                        title="GPU Requirements by Type",
                        labels={'x': 'GPU Type', 'y': 'Number of GPUs'},
                        color=list(gpu_summary.values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Full recommendations table
                recommendations = generate_instance_recommendations(gpu_summary, aws_df)
                
                if recommendations:
                    rec_df = pd.DataFrame(recommendations)
                    display_df = rec_df.copy()
                    display_df['Cost/Hour'] = display_df['total_cost'].apply(lambda x: f"${x:.2f}")
                    display_df['Monthly Cost'] = display_df['monthly_cost'].apply(lambda x: f"${x:,.2f}")
                    display_df = display_df[['gpu_type', 'total_gpus', 'instance_type', 'gpus_per_instance', 'instances_needed', 'Cost/Hour', 'Monthly Cost']]
                    display_df.columns = ['GPU Type', 'Total GPUs', 'Instance Type', 'GPUs/Instance', 'Instances Needed', 'Cost/Hour', 'Monthly Cost']
                    
                    st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()