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
        {"model": "StarCoder2-7B", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Code completion", "notes": "Set NIM_MAX_MODEL_LEN=4096"},
        {"model": "StarCoder2-7B", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Batch code generation", "notes": "Multi-user development"},
        {"model": "Llama 3.2 Instruct", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Lightweight inference", "notes": "Small footprint"},
        {"model": "Llama 3.2 Instruct", "strategy": "Throughput", "gpu_type": "L4", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Edge computing", "notes": "Very efficient"},
        {"model": "Llama 3.1 Nemotron Nano 8b V1", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Efficient inference", "notes": "NVIDIA optimized"},
        {"model": "Llama 3.1 Nemotron Nano 8b V1", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Production serving", "notes": "Balanced cost/performance"},
        {"model": "Llama 3.2 Vision Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 2, "memory_per_gpu": "80 GB", "total_memory": "160 GB", "use_case": "Multimodal real-time", "notes": "Image + text processing"},
        {"model": "Llama 3.2 Vision Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 4, "memory_per_gpu": "24 GB", "total_memory": "96 GB", "use_case": "Batch vision tasks", "notes": "Document processing"},
        {"model": "Llama 3.1 Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Fast text generation", "notes": "General purpose"},
        {"model": "Llama 3.1 Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Production serving", "notes": "Scalable deployment"},
        {"model": "Mistral Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 1, "memory_per_gpu": "80 GB", "total_memory": "80 GB", "use_case": "Fast multilingual", "notes": "European LLM"},
        {"model": "Mistral Instruct", "strategy": "Throughput", "gpu_type": "A10G", "gpu_count": 2, "memory_per_gpu": "24 GB", "total_memory": "48 GB", "use_case": "Production multilingual", "notes": "Good for European languages"},
        {"model": "Mixtral Instruct", "strategy": "Latency", "gpu_type": "H100", "gpu_count": 2, "memory_per_gpu": "80 GB", "total_memory": "160 GB", "use_case": "MoE fast inference", "notes": "Mixture of Experts"},
        {"model": "Mixtral Instruct", "strategy": "Throughput", "gpu_type": "H100", "gpu_count": 4, "memory_per_gpu": "80 GB", "total_memory": "320 GB", "use_case": "MoE batch processing", "notes": "High-volume MoE"},
        {"model": "NeMo Retriever-Parse", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Document parsing", "notes": "Text extraction"},
        {"model": "Llama 3.2 NV EmbedQA 1b V2", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Question answering", "notes": "Lightweight QA"},
        {"model": "Llama 3.2 NV RerankQA 1b V2", "strategy": "Latency", "gpu_type": "A10G", "gpu_count": 1, "memory_per_gpu": "24 GB", "total_memory": "24 GB", "use_case": "Search reranking", "notes": "Lightweight reranking"},
    ]
    return pd.DataFrame(nim_data)

@st.cache_data
def load_aws_instances():
    """Load AWS instance pricing data"""
    aws_data = [
        {"instance_type": "P5.48xlarge", "gpu_type": "H100", "gpus_per_instance": 8, "cost_per_hour": 54.00, "total_gpu_memory": "640 GB"},
        {"instance_type": "P4d.24xlarge", "gpu_type": "A100", "gpus_per_instance": 8, "cost_per_hour": 18.00, "total_gpu_memory": "320 GB"},
        {"instance_type": "P4de.24xlarge", "gpu_type": "A100", "gpus_per_instance": 8, "cost_per_hour": 22.50, "total_gpu_memory": "640 GB"},
        {"instance_type": "G5.12xlarge", "gpu_type": "A10G", "gpus_per_instance": 4, "cost_per_hour": 5.67, "total_gpu_memory": "96 GB"},
        {"instance_type": "G5.48xlarge", "gpu_type": "A10G", "gpus_per_instance": 8, "cost_per_hour": 15.90, "total_gpu_memory": "192 GB"},
        {"instance_type": "G6e.48xlarge", "gpu_type": "L40S", "gpus_per_instance": 8, "cost_per_hour": 65.40, "total_gpu_memory": "384 GB"},
        {"instance_type": "G6.48xlarge", "gpu_type": "L4", "gpus_per_instance": 8, "cost_per_hour": 17.57, "total_gpu_memory": "192 GB"},
    ]
    return pd.DataFrame(aws_data)

def get_filtered_options(df: pd.DataFrame, model: str = None, strategy: str = None, gpu_type: str = None) -> Dict[str, List]:
    """Get filtered options based on previous selections"""
    filtered_df = df.copy()
    
    if model:
        filtered_df = filtered_df[filtered_df['model'] == model]
    if strategy:
        filtered_df = filtered_df[filtered_df['strategy'] == strategy]
    if gpu_type:
        filtered_df = filtered_df[filtered_df['gpu_type'] == gpu_type]
    
    return {
        'models': sorted(df['model'].unique().tolist()),
        'strategies': sorted(filtered_df['strategy'].unique().tolist()) if model else [],
        'gpu_types': sorted(filtered_df['gpu_type'].unique().tolist()) if model and strategy else [],
        'gpu_counts': sorted(filtered_df['gpu_count'].unique().tolist()) if model and strategy and gpu_type else []
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
        2. **Choose Strategy**: Pick optimization strategy (Latency/Throughput)  
        3. **Select GPU Type**: Choose GPU hardware
        4. **Pick GPU Count**: Select number of GPUs needed
        5. **Repeat**: Add up to 8 model configurations
        6. **Review**: Check GPU summary and AWS recommendations
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
    cols = st.columns([1, 3, 2, 2, 1, 1])
    with cols[0]:
        st.write("**#**")
    with cols[1]:
        st.write("**Model**")
    with cols[2]:
        st.write("**Strategy**")
    with cols[3]:
        st.write("**GPU Type**")
    with cols[4]:
        st.write("**Count**")
    with cols[5]:
        st.write("**Clear**")
    
    # Model selection rows
    for i in range(8):
        cols = st.columns([1, 3, 2, 2, 1, 1])
        
        with cols[0]:
            st.write(f"{i+1}")
        
        # Get current selection
        current_selection = st.session_state.selections[i] if i < len(st.session_state.selections) else {}
        
        # Get filtered options
        options = get_filtered_options(
            nim_df, 
            current_selection.get('model'),
            current_selection.get('strategy'),
            current_selection.get('gpu_type')
        )
        
        with cols[1]:
            # Model dropdown
            model_key = f"model_{i}"
            selected_model = st.selectbox(
                "Model",
                [""] + options['models'],
                index=0 if not current_selection.get('model') else options['models'].index(current_selection['model']) + 1,
                key=model_key,
                label_visibility="collapsed"
            )
            
            # Update selection when model changes
            if selected_model != current_selection.get('model'):
                st.session_state.selections[i] = {'model': selected_model} if selected_model else {}
                st.rerun()
        
        with cols[2]:
            # Strategy dropdown
            strategy_key = f"strategy_{i}"
            selected_strategy = st.selectbox(
                "Strategy",
                [""] + options['strategies'],
                index=0 if not current_selection.get('strategy') else (options['strategies'].index(current_selection['strategy']) + 1 if current_selection.get('strategy') in options['strategies'] else 0),
                key=strategy_key,
                disabled=not selected_model,
                label_visibility="collapsed"
            )
            
            # Update selection when strategy changes
            if selected_strategy != current_selection.get('strategy'):
                updated_selection = st.session_state.selections[i].copy()
                updated_selection['strategy'] = selected_strategy
                if 'gpu_type' in updated_selection:
                    del updated_selection['gpu_type']
                if 'gpu_count' in updated_selection:
                    del updated_selection['gpu_count']
                if 'configuration' in updated_selection:
                    del updated_selection['configuration']
                st.session_state.selections[i] = updated_selection
                st.rerun()
        
        with cols[3]:
            # GPU Type dropdown
            gpu_type_key = f"gpu_type_{i}"
            selected_gpu_type = st.selectbox(
                "GPU Type",
                [""] + options['gpu_types'],
                index=0 if not current_selection.get('gpu_type') else (options['gpu_types'].index(current_selection['gpu_type']) + 1 if current_selection.get('gpu_type') in options['gpu_types'] else 0),
                key=gpu_type_key,
                disabled=not selected_strategy,
                label_visibility="collapsed"
            )
            
            # Update selection when GPU type changes
            if selected_gpu_type != current_selection.get('gpu_type'):
                updated_selection = st.session_state.selections[i].copy()
                updated_selection['gpu_type'] = selected_gpu_type
                if 'gpu_count' in updated_selection:
                    del updated_selection['gpu_count']
                if 'configuration' in updated_selection:
                    del updated_selection['configuration']
                st.session_state.selections[i] = updated_selection
                st.rerun()
        
        with cols[4]:
            # GPU Count dropdown
            gpu_count_key = f"gpu_count_{i}"
            selected_gpu_count = st.selectbox(
                "Count",
                [""] + [str(x) for x in options['gpu_counts']],
                index=0 if not current_selection.get('gpu_count') else ([str(x) for x in options['gpu_counts']].index(str(current_selection['gpu_count'])) + 1 if current_selection.get('gpu_count') in options['gpu_counts'] else 0),
                key=gpu_count_key,
                disabled=not selected_gpu_type,
                label_visibility="collapsed"
            )
            
            # Update configuration when GPU count changes
            if selected_gpu_count and selected_gpu_count != str(current_selection.get('gpu_count', '')):
                # Find the exact configuration
                config = nim_df[
                    (nim_df['model'] == selected_model) &
                    (nim_df['strategy'] == selected_strategy) &
                    (nim_df['gpu_type'] == selected_gpu_type) &
                    (nim_df['gpu_count'] == int(selected_gpu_count))
                ].iloc[0].to_dict()
                
                updated_selection = st.session_state.selections[i].copy()
                updated_selection.update({
                    'gpu_count': int(selected_gpu_count),
                    'configuration': config
                })
                st.session_state.selections[i] = updated_selection
                st.rerun()
        
        with cols[5]:
            # Clear button
            if st.button("🗑️", key=f"clear_{i}", help="Clear this selection"):
                st.session_state.selections[i] = {}
                st.rerun()
        
        # Show configuration details
        if current_selection.get('configuration'):
            config = current_selection['configuration']
            st.markdown(f"<small>**{config['total_memory']}** total | **{config['use_case']}** | {config['notes']}</small>", unsafe_allow_html=True)
    
    st.divider()
    
    # GPU Requirements Summary
    st.header("📊 GPU Requirements Summary")
    
    gpu_summary = calculate_gpu_summary(st.session_state.selections)
    
    if gpu_summary:
        # Create metrics columns
        metric_cols = st.columns(len(['H100', 'A100', 'A10G', 'L40S', 'L4']))
        gpu_types = ['H100', 'A100', 'A10G', 'L40S', 'L4']
        
        for i, gpu_type in enumerate(gpu_types):
            count = gpu_summary.get(gpu_type, 0)
            with metric_cols[i]:
                st.metric(
                    label=gpu_type,
                    value=f"{count} GPUs",
                    delta=None
                )
        
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
    else:
        st.info("Select models above to see GPU requirements summary")
    
    st.divider()
    
    # Instance Recommendations
    st.header("💰 AWS Instance Recommendations")
    
    if gpu_summary:
        recommendations = generate_instance_recommendations(gpu_summary, aws_df)
        
        if recommendations:
            # Create recommendations table
            rec_df = pd.DataFrame(recommendations)
            
            # Format the dataframe for display
            display_df = rec_df.copy()
            display_df['Cost/Hour'] = display_df['total_cost'].apply(lambda x: f"${x:.2f}")
            display_df['Monthly Cost'] = display_df['monthly_cost'].apply(lambda x: f"${x:,.2f}")
            display_df = display_df[['gpu_type', 'total_gpus', 'instance_type', 'gpus_per_instance', 'instances_needed', 'Cost/Hour', 'Monthly Cost']]
            display_df.columns = ['GPU Type', 'Total GPUs', 'Instance Type', 'GPUs/Instance', 'Instances Needed', 'Cost/Hour', 'Monthly Cost']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Calculate totals
            total_hourly = sum(rec['total_cost'] for rec in recommendations)
            total_monthly = total_hourly * 24 * 30
            
            # Display cost summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Hourly Cost", f"${total_hourly:.2f}")
            with col2:
                st.metric("Daily Cost (24/7)", f"${total_hourly * 24:.2f}")
            with col3:
                st.metric("Monthly Cost (24/7)", f"${total_monthly:,.2f}")
            
            # Cost breakdown chart
            if len(recommendations) > 1:
                fig_cost = px.pie(
                    values=[rec['total_cost'] for rec in recommendations],
                    names=[f"{rec['instance_type']}" for rec in recommendations],
                    title="Cost Breakdown by Instance Type"
                )
                st.plotly_chart(fig_cost, use_container_width=True)
            
        else:
            st.warning("No instance recommendations available for current GPU selection")
    else:
        st.info("Select models above to see AWS instance recommendations")
    
    # Export functionality
    st.divider()
    st.header("📤 Export Configuration")
    
    if gpu_summary:
        # Create export data
        export_data = []
        for i, selection in enumerate(st.session_state.selections):
            if selection.get('configuration'):
                config = selection['configuration']
                export_data.append({
                    'Selection': i + 1,
                    'Model': config['model'],
                    'Strategy': config['strategy'],
                    'GPU Type': config['gpu_type'],
                    'GPU Count': config['gpu_count'],
                    'Total Memory': config['total_memory'],
                    'Use Case': config['use_case'],
                    'Notes': config['notes']
                })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            
            # Display export preview
            st.subheader("Configuration Summary")
            st.dataframe(export_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name="nim_gpu_sizing_configuration.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create detailed report
                report_data = {
                    'Configuration': export_df.to_dict('records'),
                    'GPU Summary': gpu_summary,
                    'Recommendations': recommendations if gpu_summary else []
                }
                
                import json
                json_data = json.dumps(report_data, indent=2)
                st.download_button(
                    label="📊 Download Full Report (JSON)",
                    data=json_data,
                    file_name="nim_gpu_sizing_report.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()